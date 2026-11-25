"""Recursive dynamics and context-adaptation training for WorldActionModel."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import random
import shutil
import subprocess
import time
from typing import Any

import torch

from browser_world.dataset import schema
from browser_world.interaction import executor
from browser_world.wam2 import losses
from browser_world.wam2 import model
from browser_world.wam2 import observation
from browser_world.wam2 import semantic
from browser_world.wam2 import training_data

ARCHITECTURE_VERSION = 'wam2_structured_feedback_v1'
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = _REPO_ROOT / 'artifacts' / 'wam2' / 'dynamics_v1'


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Configuration for Step 3 dynamics training."""

    seed: int = 0
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    trajectory_batch_size: int = 4
    grad_accum: int = 4
    epochs: int = 20
    raw_dir: pathlib.Path = training_data.DEFAULT_RAW_DIR
    vocab_path: pathlib.Path = observation.DEFAULT_VOCAB_PATH
    out_dir: pathlib.Path = DEFAULT_OUT_DIR


class CachedTextEncoder:
    """CPU-cached wrapper around SemanticEncoder for fast epoch iteration."""

    def __init__(
        self,
        cache: dict[str, torch.Tensor],
        device: torch.device,
    ) -> None:
        self._cache = cache
        self.device = device

    def encode(
        self, texts: Sequence[str], max_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del max_tokens
        batch = len(texts)
        hidden = torch.zeros(
            (batch, 1, semantic.TEXT_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.zeros((batch, 1), dtype=torch.bool, device=self.device)
        offsets = torch.zeros((batch, 1, 2), dtype=torch.long)
        for idx, text in enumerate(texts):
            if text:
                hidden[idx, 0] = self._cache[text].to(self.device)
                mask[idx, 0] = True
                offsets[idx, 0, 1] = len(text)
        return hidden, mask, offsets


@dataclasses.dataclass(frozen=True)
class EncodedWindow:
    """Pre-aligned observations and action metadata for one ExperienceWindow."""

    aligned_obs: tuple[observation.AlignedObservation, ...]
    actions: tuple[executor.Action, ...]
    pointers: tuple[tuple[float, float], ...]
    grounded_slots: tuple[int | None, ...]
    step0_action: observation.ActionInputs


def build_split_text_cache(
    loaded: training_data.LoadedExperiences,
    encoder: observation.TextEncoder,
    batch_size: int = 32,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Extracts and caches pooled FP32 semantic features on CPU."""
    unique: list[str] = []
    seen: set[str] = set()
    for win in loaded.windows[1]:
        for obs in win.observations:
            for elem in obs.elements:
                for txt in (elem.name or '', elem.value or ''):
                    if txt and txt not in seen:
                        seen.add(txt)
                        unique.append(txt)
        for act in win.actions:
            if act.text and act.text not in seen:
                seen.add(act.text)
                unique.append(act.text)

    cache: dict[str, torch.Tensor] = {}
    if unique:
        feats = observation.encode_texts(
            tuple(unique), encoder, batch_size=batch_size
        ).cpu()
        for txt, vec in zip(unique, feats):
            cache[txt] = vec

    meta = {
        'split': loaded.split,
        'unique_strings': len(cache),
        'model_id': semantic.MODEL_ID,
        'revision': semantic.REVISION,
        'max_tokens': 128,
        'pooling': 'masked_mean',
    }
    return cache, meta


def encode_windows(
    windows: Sequence[training_data.ExperienceWindow],
    cached_encoder: observation.TextEncoder,
    role_to_index: dict[str, int],
    slot_count: int = model.SLOT_COUNT,
) -> tuple[EncodedWindow, ...]:
    """Pre-aligns all observations in each window and resolves groundings."""
    encoded: list[EncodedWindow] = []
    for win in windows:
        aligned: list[observation.AlignedObservation] = []
        prev: observation.AlignedObservation | None = None
        for obs in win.observations:
            prev = observation.encode_observation(
                obs, cached_encoder, role_to_index,
                slot_count=slot_count, previous=prev,
            )
            aligned.append(prev)

        grounded: list[int | None] = []
        for step_idx in range(len(win.actions)):
            slot = training_data.trace_target_to_initial_slot(
                win, step_idx, aligned[0]
            )
            grounded.append(slot)

        step0_act = observation.encode_action(
            win.actions[0], aligned[0], cached_encoder, win.pointers[0]
        )
        encoded.append(
            EncodedWindow(
                aligned_obs=tuple(aligned),
                actions=win.actions,
                pointers=win.pointers,
                grounded_slots=tuple(grounded),
                step0_action=step0_act,
            )
        )
    return tuple(encoded)


def _imagined_action_inputs(
    win: EncodedWindow,
    step_idx: int,
    pred_belief: model.Belief,
    last_pointer: tuple[float, float],
    cached_encoder: observation.TextEncoder,
) -> tuple[observation.ActionInputs, tuple[float, float]]:
    """Builds ActionInputs at step >= 1 from predicted belief geometry."""
    act = win.actions[step_idx]
    slot = win.grounded_slots[step_idx]
    if slot is None:
        raise RuntimeError('Cannot encode ungrounded future action.')

    obs0 = win.aligned_obs[0].observation
    vw = max(float(obs0.motion.viewport_width), 1.0)
    vh = max(float(obs0.motion.viewport_height), 1.0)
    if slot >= 0:
        bx, by, bw, bh = pred_belief.boxes[0, slot].detach().tolist()
        ptr = ((bx + 0.5 * bw) * vw, (by + 0.5 * bh) * vh)
    else:
        ptr = last_pointer

    device = pred_belief.latent.device
    kind_idx = schema.ACTION_KINDS.index(act.kind)
    kind = torch.tensor([kind_idx], dtype=torch.long, device=device)
    target = torch.tensor([slot], dtype=torch.long, device=device)
    text = (
        observation.encode_texts((act.text,), cached_encoder).to(device)
        if act.text
        else torch.zeros(
            (1, semantic.TEXT_DIM), dtype=torch.float32, device=device
        )
    )
    checked = torch.zeros((1, 3), dtype=torch.float32, device=device)
    if act.checked is None:
        checked[0, 0] = 1.0
    elif act.checked is False:
        checked[0, 1] = 1.0
    else:
        checked[0, 2] = 1.0
    scroll = torch.tensor(
        [[float(act.scroll_y or 0.0) / vh]],
        dtype=torch.float32,
        device=device,
    )
    ptr_t = torch.tensor(
        [[ptr[0] / vw, ptr[1] / vh]], dtype=torch.float32, device=device
    )
    return (
        observation.ActionInputs(kind, target, text, checked, scroll, ptr_t),
        ptr,
    )


@dataclasses.dataclass(frozen=True)
class WindowLossResult:
    """Loss tensor and component telemetry for one training window."""

    loss: torch.Tensor
    total_val: float
    raw_components: dict[str, float]
    weighted_components: dict[str, float]
    supported_steps: int
    unsupported_continuations: int


def _window_training_loss(
    world: model.WorldActionModel,
    win: EncodedWindow,
    cached_encoder: observation.TextEncoder,
) -> WindowLossResult:
    """Computes rollout + inverse + context-adaptation loss on one window."""
    belief = observation.observe(world, win.aligned_obs[0], previous=None)
    actual_belief = belief
    total_loss = (belief.latent * 0.0).sum()
    last_ptr = win.pointers[0]
    first_pred: model.Prediction | None = None
    first_act_lat: torch.Tensor | None = None
    actual_1_belief: model.Belief | None = None
    raw_sums: dict[str, float] = {k: 0.0 for k in losses.LOSS_WEIGHTS}
    weighted_sums: dict[str, float] = {k: 0.0 for k in losses.LOSS_WEIGHTS}
    raw_sums['adaptation'] = 0.0
    weighted_sums['adaptation'] = 0.0
    supported_steps = 0
    unsupported_conts = 0

    for step_idx in range(len(win.actions)):
        if win.grounded_slots[step_idx] is None:
            unsupported_conts += 1
            break
        if step_idx == 0:
            act_in = win.step0_action
        else:
            assert first_pred is not None
            act_in, last_ptr = _imagined_action_inputs(
                win, step_idx, belief, last_ptr, cached_encoder
            )

        act_lat = world.encode_action(
            belief, act_in.kind, act_in.target, act_in.text,
            act_in.checked, act_in.scroll, act_in.pointer,
        )
        pred = world.imagine(belief, act_lat, act_in.target)
        cur_obs = win.aligned_obs[step_idx]
        next_obs = win.aligned_obs[step_idx + 1]
        prev_actual = actual_belief
        actual_belief = observation.observe(
            world, next_obs, previous=prev_actual
        )
        step_l = losses.step_dynamics_losses(
            pred, cur_obs.inputs.present, cur_obs.inputs.states,
            next_obs.inputs, actual_belief.latent,
        )
        total_loss = total_loss + step_l.total
        supported_steps += 1
        for k, v in step_l.components.items():
            val = float(v.item())
            raw_sums[k] += val
            weighted_sums[k] += losses.LOSS_WEIGHTS[k] * val

        if step_idx == 0:
            first_pred = pred
            first_act_lat = act_lat
            actual_1_belief = actual_belief
            inv_l, _, _ = losses.inverse_dynamics_loss(
                world, prev_actual, actual_belief, act_in
            )
            inv_val = float(inv_l.item())
            raw_sums['inverse_dynamics'] += inv_val
            weighted_sums['inverse_dynamics'] += (
                losses.LOSS_WEIGHTS['inverse_dynamics'] * inv_val
            )
            total_loss = (
                total_loss + losses.LOSS_WEIGHTS['inverse_dynamics'] * inv_l
            )
        belief = pred.belief

    # Experience-conditioned context adaptation (separate from rollout).
    if (
        len(win.actions) >= 2
        and first_pred is not None
        and first_act_lat is not None
        and actual_1_belief is not None
        and win.grounded_slots[1] is not None
    ):
        obs1 = win.aligned_obs[1]
        known_1 = (obs1.inputs.present > 0.5) & obs1.inputs.existence_known
        corrected_1 = world.correct_context(
            actual_1_belief, first_pred.belief, first_act_lat, known_1
        )
        act1_in = observation.encode_action(
            win.actions[1], obs1, cached_encoder, win.pointers[1]
        )
        act1_lat = world.encode_action(
            corrected_1, act1_in.kind, act1_in.target, act1_in.text,
            act1_in.checked, act1_in.scroll, act1_in.pointer,
        )
        pred_adapt = world.imagine(corrected_1, act1_lat, act1_in.target)
        obs2 = win.aligned_obs[2]
        actual_2 = observation.observe(world, obs2, previous=actual_1_belief)
        adapt_l = losses.step_dynamics_losses(
            pred_adapt, obs1.inputs.present, obs1.inputs.states,
            obs2.inputs, actual_2.latent,
        )
        adapt_val = float(adapt_l.total.item())
        raw_sums['adaptation'] += adapt_val
        weighted_sums['adaptation'] += adapt_val
        total_loss = total_loss + adapt_l.total

    return WindowLossResult(
        loss=total_loss,
        total_val=float(total_loss.item()),
        raw_components=raw_sums,
        weighted_components=weighted_sums,
        supported_steps=supported_steps,
        unsupported_continuations=unsupported_conts,
    )


def evaluate_validation(
    world: model.WorldActionModel,
    val_by_horizon: dict[int, tuple[EncodedWindow, ...]],
    cached_encoder: observation.TextEncoder,
) -> dict[str, Any]:
    """Evaluates validation horizons H=1,2,3 and returns selection metric."""
    world.eval()
    horizon_reports: dict[str, Any] = {}
    selection_losses: list[float] = []

    with torch.no_grad():
        for horizon, windows in sorted(val_by_horizon.items()):
            sel_vals: list[float] = []
            tot_vals: list[float] = []
            latent_l1s: list[float] = []
            geom_l1s: list[float] = []
            dir_counts = {k: [0, 0] for k in losses.DIRECTION_STRATA}
            ex_counts = {k: [0, 0] for k in losses.EXISTENCE_STRATA}
            brier_sum = 0.0
            brier_cnt = 0
            persist_brier_sum = 0.0
            inv_kind_ok = 0
            inv_tgt_ok = 0
            inv_tot = 0
            terminated = 0

            for win in windows:
                belief = observation.observe(
                    world, win.aligned_obs[0], previous=None
                )
                actual_belief = belief
                last_ptr = win.pointers[0]
                stopped = False
                for step_idx in range(horizon):
                    if win.grounded_slots[step_idx] is None:
                        terminated += 1
                        stopped = True
                        break
                    if step_idx == 0:
                        act_in = win.step0_action
                    else:
                        act_in, last_ptr = _imagined_action_inputs(
                            win, step_idx, belief, last_ptr, cached_encoder
                        )
                    act_lat = world.encode_action(
                        belief, act_in.kind, act_in.target, act_in.text,
                        act_in.checked, act_in.scroll, act_in.pointer,
                    )
                    pred = world.imagine(belief, act_lat, act_in.target)
                    cur = win.aligned_obs[step_idx]
                    nxt = win.aligned_obs[step_idx + 1]
                    prev_actual = actual_belief
                    actual_belief = observation.observe(
                        world, nxt, previous=prev_actual
                    )
                    sl = losses.step_dynamics_losses(
                        pred, cur.inputs.present, cur.inputs.states,
                        nxt.inputs, actual_belief.latent,
                    )
                    if step_idx == horizon - 1:
                        sel_vals.append(float(sl.selection_loss.item()))
                        tot_vals.append(float(sl.total.item()))
                        latent_l1s.append(
                            float(sl.components['latent_consistency'].item())
                        )
                        geom_l1s.append(float(sl.components['geometry'].item()))

                        # Directional state & Brier evaluation on final step.
                        pmask = (
                            (belief.present > 0.5)
                            & (nxt.inputs.present > 0.5)
                            & nxt.inputs.existence_known
                        )
                        if pmask.any():
                            p_s = pred.belief.states[pmask]
                            t_s = nxt.inputs.states[pmask]
                            prev_s = (
                                win.aligned_obs[step_idx].inputs.states[pmask]
                            )
                            brier_sum += float(((p_s - t_s) ** 2).sum().item())
                            persist_brier_sum += float(
                                ((prev_s - t_s) ** 2).sum().item()
                            )
                            brier_cnt += int(p_s.numel())

                            for s_name, cond in (
                                ('0->0', (prev_s < 0.5) & (t_s < 0.5)),
                                ('0->1', (prev_s < 0.5) & (t_s >= 0.5)),
                                ('1->0', (prev_s >= 0.5) & (t_s < 0.5)),
                                ('1->1', (prev_s >= 0.5) & (t_s >= 0.5)),
                            ):
                                dir_counts[s_name][1] += int(cond.sum().item())
                                dir_counts[s_name][0] += int(
                                    ((p_s >= 0.5) == (t_s >= 0.5))[cond]
                                    .sum()
                                    .item()
                                )

                        # Existence accuracy by stratum.
                        kmask = nxt.inputs.existence_known
                        prev_on = (
                            win.aligned_obs[step_idx].inputs.present > 0.5
                        )
                        nxt_on = nxt.inputs.present > 0.5
                        pred_on = pred.belief.present >= 0.5
                        for e_name, cond in (
                            ('persistence', kmask & prev_on & nxt_on),
                            ('removal', kmask & prev_on & ~nxt_on),
                            ('birth', kmask & ~prev_on & nxt_on),
                            ('empty', kmask & ~prev_on & ~nxt_on),
                        ):
                            ex_counts[e_name][1] += int(cond.sum().item())
                            ex_counts[e_name][0] += int(
                                (pred_on == nxt_on)[cond].sum().item()
                            )

                    if step_idx == 0 and horizon == 1:
                        _, k_ok, t_ok = losses.inverse_dynamics_loss(
                            world, prev_actual, actual_belief, act_in
                        )
                        inv_kind_ok += int(k_ok)
                        inv_tgt_ok += int(t_ok)
                        inv_tot += 1
                    belief = pred.belief
                if stopped:
                    continue

            mean_sel = (
                sum(sel_vals) / len(sel_vals) if sel_vals else 0.0
            )
            if sel_vals:
                selection_losses.append(mean_sel)

            dir_acc = {
                k: {
                    'acc': round(ok / tot, 4) if tot else None,
                    'support': tot,
                }
                for k, (ok, tot) in dir_counts.items()
            }
            ex_acc = {
                k: {
                    'acc': round(ok / tot, 4) if tot else None,
                    'support': tot,
                }
                for k, (ok, tot) in ex_counts.items()
            }
            horizon_reports[f'H={horizon}'] = {
                'evaluated_windows': len(sel_vals),
                'terminated_unknown_future_entity': terminated,
                'selection_loss': round(mean_sel, 4),
                'total_loss': (
                    round(sum(tot_vals) / len(tot_vals), 4) if tot_vals else 0.0
                ),
                'latent_l1': (
                    round(sum(latent_l1s) / len(latent_l1s), 4)
                    if latent_l1s else 0.0
                ),
                'geometry_l1': (
                    round(sum(geom_l1s) / len(geom_l1s), 4)
                    if geom_l1s else 0.0
                ),
                'state_brier': (
                    round(brier_sum / brier_cnt, 4) if brier_cnt else None
                ),
                'persist_baseline_brier': (
                    round(persist_brier_sum / brier_cnt, 4)
                    if brier_cnt else None
                ),
                'directional_states': dir_acc,
                'existence': ex_acc,
                'inverse_kind_acc': (
                    round(inv_kind_ok / inv_tot, 4) if inv_tot else None
                ),
                'inverse_target_acc': (
                    round(inv_tgt_ok / inv_tot, 4) if inv_tot else None
                ),
            }

    mean_selection = (
        sum(selection_losses) / len(selection_losses)
        if selection_losses else float('inf')
    )
    return {
        'mean_selection_loss': round(mean_selection, 4),
        'horizons': horizon_reports,
    }


def validate_train_config(
    cfg: TrainConfig,
    roles: tuple[str, ...],
    train_count: int,
    val_count: int,
) -> None:
    """Validates training hyperparameters, output directory, and readiness."""
    if cfg.epochs < 1:
        raise ValueError(f'epochs must be >= 1, got {cfg.epochs}.')
    if cfg.trajectory_batch_size < 1:
        raise ValueError(
            f'trajectory_batch_size must be >= 1, '
            f'got {cfg.trajectory_batch_size}.'
        )
    if cfg.grad_accum < 1:
        raise ValueError(f'grad_accum must be >= 1, got {cfg.grad_accum}.')
    if cfg.lr <= 0:
        raise ValueError(f'lr must be > 0, got {cfg.lr}.')
    if cfg.grad_clip <= 0:
        raise ValueError(f'grad_clip must be > 0, got {cfg.grad_clip}.')
    if len(roles) < 2:
        raise ValueError(
            f'Role vocabulary must have >= 2 roles, got {len(roles)}.'
        )
    if train_count < 1:
        raise ValueError('Training windows cannot be empty.')
    if val_count < 1:
        raise ValueError('Validation windows cannot be empty.')
    if cfg.out_dir.exists() and any(cfg.out_dir.iterdir()):
        raise ValueError(
            f'Output directory {cfg.out_dir} already contains artifacts.'
        )


def _state_dict_sha256(world: model.WorldActionModel) -> str:
    h = hashlib.sha256()
    for name in sorted(world.state_dict()):
        tensor = world.state_dict()[name].detach().cpu().contiguous()
        h.update(name.encode('utf-8'))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _split_hashes(raw_dir: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for split in training_data.ALLOWED_SPLITS:
        p = raw_dir / split / 'transitions.jsonl'
        if p.exists():
            out[split] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _write_source_provenance(out_dir: pathlib.Path) -> dict[str, Any]:
    """Saves source_manifest.json and reviewable source snapshot."""
    source_files = [
        'browser_world/wam2/__init__.py',
        'browser_world/wam2/model.py',
        'browser_world/wam2/semantic.py',
        'browser_world/wam2/observation.py',
        'browser_world/wam2/training_data.py',
        'browser_world/wam2/losses.py',
        'browser_world/wam2/train.py',
        'browser_world/wam2/smoke_test.py',
        'browser_world/dataset/schema.py',
        'browser_world/dataset/curate.py',
        'browser_world/interaction/observer.py',
        'browser_world/interaction/executor.py',
        'browser_world/collection/recorder.py',
        'requirements.txt',
        'requirements-dev.txt',
        'pytest.ini',
    ]
    snap_dir = out_dir / 'source_snapshot'
    entries: list[dict[str, Any]] = []
    for rel in source_files:
        src = _REPO_ROOT / rel
        if not src.exists():
            continue
        dst = snap_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        data = src.read_bytes()
        entries.append({
            'relative_path': rel,
            'size_bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(),
        })
    git_head = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    manifest = {'git_head': git_head, 'files': entries}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'source_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )
    return manifest


def save_checkpoint(
    path: pathlib.Path,
    world: model.WorldActionModel,
    roles: tuple[str, ...],
    cfg: TrainConfig,
    dataset_hashes: dict[str, str],
    source_provenance: dict[str, Any],
    epoch: int,
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> str:
    """Atomically saves weights-only checkpoint with unified schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'architecture_version': ARCHITECTURE_VERSION,
        'architecture_config': {
            'role_count': len(roles),
            'slot_count': world.slot_count,
            'slot_dim': model.SLOT_DIM,
            'mechanism_dim': model.MECHANISM_DIM,
            'state_dim': model.STATE_DIM,
            'content_dim': model.CONTENT_DIM,
            'context_dim': model.CONTEXT_DIM,
        },
        'param_count': sum(p.numel() for p in world.parameters()),
        'role_vocabulary': list(roles),
        'slot_count': world.slot_count,
        'semantic_encoder_id': semantic.MODEL_ID,
        'semantic_encoder_revision': semantic.REVISION,
        'state_names': list(schema.STATE_NAMES),
        'action_kinds': list(schema.ACTION_KINDS),
        'loss_convention': 'scalar_position_masked_mean_history_conditioned_targets',
        'loss_weights': dict(losses.LOSS_WEIGHTS),
        'train_config': {
            'seed': cfg.seed,
            'lr': cfg.lr,
            'weight_decay': cfg.weight_decay,
            'grad_clip': cfg.grad_clip,
            'trajectory_batch_size': cfg.trajectory_batch_size,
            'grad_accum': cfg.grad_accum,
            'epochs': cfg.epochs,
        },
        'dataset_hashes': dataset_hashes,
        'source_provenance': {'git_head': source_provenance.get('git_head', '')},
        'trained_capabilities': [
            'one_step_dynamics',
            'recursive_dynamics',
            'directional_boolean_states',
            'existence_and_birth_removal',
            'inverse_dynamics',
            'context_adaptation',
        ],
        'untrained_capabilities': [
            'goal_evaluation',
            'return_estimation',
            'action_proposal',
            'text_copy',
        ],
        'epoch': epoch,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'model_state_dict': world.state_dict(),
    }
    tmp_path = path.with_suffix('.pt.tmp')
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checkpoint_reload(
    path: pathlib.Path,
    device: torch.device,
) -> dict[str, Any]:
    """Verifies strict weights-only checkpoint loading."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    reloaded = model.WorldActionModel(
        role_count=len(ckpt['role_vocabulary']),
        slot_count=int(ckpt['slot_count']),
    ).to(device)
    reloaded.load_state_dict(ckpt['model_state_dict'], strict=True)
    return {
        'param_count': sum(p.numel() for p in reloaded.parameters()),
        'epoch': int(ckpt['epoch']),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _dataset_hash(raw_dir: pathlib.Path) -> str:
    h = hashlib.sha256()
    for split in training_data.ALLOWED_SPLITS:
        p = raw_dir / split / 'transitions.jsonl'
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def run_training(cfg: TrainConfig) -> dict[str, Any]:
    """Executes cached semantic pre-encoding and 20-epoch dynamics training."""
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.reset_peak_memory_stats()

    role_map, roles = observation.load_role_vocab(cfg.vocab_path)
    train_raw = training_data.load_experience_windows(
        cfg.raw_dir, split='train', max_horizon=3
    )
    val_raw = training_data.load_experience_windows(
        cfg.raw_dir, split='validation', max_horizon=3
    )

    sem_enc = semantic.SemanticEncoder(device)
    tr_cache, tr_cache_meta = build_split_text_cache(train_raw, sem_enc)
    va_cache, va_cache_meta = build_split_text_cache(val_raw, sem_enc)
    del sem_enc
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    tr_cached_enc = CachedTextEncoder(tr_cache, device)
    va_cached_enc = CachedTextEncoder(va_cache, device)

    train_encoded = tuple(
        w
        for h in (1, 2, 3)
        for w in encode_windows(train_raw.windows[h], tr_cached_enc, role_map)
    )
    val_by_h = {
        h: encode_windows(val_raw.windows[h], va_cached_enc, role_map)
        for h in (1, 2, 3)
    }
    val_count = sum(len(v) for v in val_by_h.values())
    validate_train_config(cfg, roles, len(train_encoded), val_count)

    world = model.WorldActionModel(
        role_count=len(roles), slot_count=model.SLOT_COUNT
    ).to(device)
    init_sha = _state_dict_sha256(world)
    opt = torch.optim.AdamW(
        world.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    d_hashes = _split_hashes(cfg.raw_dir)
    provenance = _write_source_provenance(cfg.out_dir)
    history: list[dict[str, Any]] = []
    best_sel = float('inf')
    best_epoch = -1
    last_val_metrics: dict[str, Any] = {}
    t_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        t_ep = time.perf_counter()
        world.train()
        indices = list(range(len(train_encoded)))
        random.Random(cfg.seed + epoch).shuffle(indices)

        opt.zero_grad(set_to_none=True)
        epoch_losses: list[float] = []
        raw_acc: dict[str, float] = {
            **{k: 0.0 for k in losses.LOSS_WEIGHTS}, 'adaptation': 0.0
        }
        weighted_acc = dict(raw_acc)
        sup_steps_tot = 0
        unsup_cont_tot = 0
        grad_norms: list[float] = []
        micro_step = 0

        for start in range(0, len(indices), cfg.trajectory_batch_size):
            batch_ids = indices[start:start + cfg.trajectory_batch_size]
            batch_loss = torch.zeros((), device=device)
            for idx in batch_ids:
                res = _window_training_loss(
                    world, train_encoded[idx], tr_cached_enc
                )
                if not torch.isfinite(res.loss):
                    raise RuntimeError(
                        f'Non-finite training loss at epoch={epoch} '
                        f'window_idx={idx}.'
                    )
                batch_loss = batch_loss + (res.loss / len(batch_ids))
                epoch_losses.append(res.total_val)
                sup_steps_tot += res.supported_steps
                unsup_cont_tot += res.unsupported_continuations
                for k, v in res.raw_components.items():
                    raw_acc[k] += v
                for k, v in res.weighted_components.items():
                    weighted_acc[k] += v

            (batch_loss / cfg.grad_accum).backward()
            micro_step += 1
            if micro_step % cfg.grad_accum == 0 or (
                start + cfg.trajectory_batch_size >= len(indices)
            ):
                gn = float(
                    torch.nn.utils.clip_grad_norm_(
                        world.parameters(), cfg.grad_clip
                    ).item()
                )
                if not math.isfinite(gn):
                    raise RuntimeError(
                        f'Non-finite gradient norm at epoch={epoch}.'
                    )
                grad_norms.append(gn)
                opt.step()
                opt.zero_grad(set_to_none=True)

        val_metrics = evaluate_validation(world, val_by_h, va_cached_enc)
        last_val_metrics = val_metrics
        val_sel = float(val_metrics['mean_selection_loss'])
        if not math.isfinite(val_sel):
            raise RuntimeError(
                f'Non-finite validation selection loss at epoch={epoch}.'
            )
        n_win = max(len(epoch_losses), 1)
        tr_mean = sum(epoch_losses) / n_win
        ep_elapsed = time.perf_counter() - t_ep
        train_metrics = {
            'train_loss': round(tr_mean, 4),
            'raw_components': {
                k: round(v / n_win, 4) for k, v in raw_acc.items()
            },
            'weighted_components': {
                k: round(v / n_win, 4) for k, v in weighted_acc.items()
            },
            'lr': cfg.lr,
            'mean_grad_norm': round(
                sum(grad_norms) / max(len(grad_norms), 1), 4
            ),
            'max_grad_norm': round(max(grad_norms) if grad_norms else 0.0, 4),
            'supported_rollout_steps': sup_steps_tot,
            'unsupported_continuations': unsup_cont_tot,
            'elapsed_s': round(ep_elapsed, 2),
        }
        record = {
            'epoch': epoch,
            **train_metrics,
            'val_selection_loss': val_sel,
            'val_horizons': val_metrics['horizons'],
        }
        history.append(record)
        print(
            f'Epoch {epoch:02d}/{cfg.epochs} | '
            f'train={tr_mean:.4f} | val_sel={val_sel:.4f} | '
            f'gn={train_metrics["mean_grad_norm"]:.3f} | '
            f'{ep_elapsed:.1f}s',
            flush=True,
        )

        if val_sel < best_sel:
            best_sel = val_sel
            best_epoch = epoch
            save_checkpoint(
                cfg.out_dir / 'best.pt', world, roles, cfg,
                d_hashes, provenance, epoch, train_metrics, val_metrics,
            )

    if best_epoch < 1 or not (cfg.out_dir / 'best.pt').exists():
        raise RuntimeError('No valid best checkpoint was saved during training.')

    last_sha = save_checkpoint(
        cfg.out_dir / 'last.pt', world, roles, cfg,
        d_hashes, provenance, cfg.epochs,
        {k: v for k, v in history[-1].items() if k != 'val_horizons'},
        last_val_metrics,
    )
    best_reload = verify_checkpoint_reload(cfg.out_dir / 'best.pt', device)
    duration_s = time.perf_counter() - t_start
    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if device.type == 'cuda' else 0.0
    )

    summary = {
        'architecture_version': ARCHITECTURE_VERSION,
        'param_count': sum(p.numel() for p in world.parameters()),
        'init_state_sha256': init_sha,
        'duration_s': round(duration_s, 2),
        'peak_cuda_mb': round(peak_mb, 2),
        'unique_transitions': {
            'train': train_raw.transition_count,
            'validation': val_raw.transition_count,
        },
        'train_windows_total': len(train_encoded),
        'best_epoch': best_epoch,
        'best_selection_loss': best_sel,
        'best_checkpoint': best_reload,
        'last_checkpoint_sha256': last_sha,
        'dataset_hashes': d_hashes,
        'caches': {'train': tr_cache_meta, 'validation': va_cache_meta},
        'history': history,
    }
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / 'config.json').write_text(
        json.dumps(
            {
                'seed': cfg.seed,
                'lr': cfg.lr,
                'weight_decay': cfg.weight_decay,
                'grad_clip': cfg.grad_clip,
                'trajectory_batch_size': cfg.trajectory_batch_size,
                'grad_accum': cfg.grad_accum,
                'epochs': cfg.epochs,
                'loss_weights': losses.LOSS_WEIGHTS,
                'dataset_hashes': d_hashes,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    (cfg.out_dir / 'metrics.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--out-dir', default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    cfg = TrainConfig(epochs=args.epochs, out_dir=pathlib.Path(args.out_dir))
    summary = run_training(cfg)
    print(
        f'Done in {summary["duration_s"]}s | '
        f'best_epoch={summary["best_epoch"]} | '
        f'best_sel={summary["best_selection_loss"]:.4f} | '
        f'sha256={summary["best_checkpoint"]["sha256"]}'
    )


if __name__ == '__main__':
    main()
