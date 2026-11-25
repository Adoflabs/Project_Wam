"""Recorded raw experience windows and recursive rollout grounding."""

from __future__ import annotations

import collections
import dataclasses
import json
import pathlib
from typing import Any

from browser_world.collection import recorder
from browser_world.dataset import curate
from browser_world.interaction import executor
from browser_world.interaction import observer
from browser_world.wam2 import model
from browser_world.wam2 import observation

ALLOWED_SPLITS = ('train', 'validation')
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = _REPO_ROOT / 'data' / 'raw' / 'e04' / 'main'


@dataclasses.dataclass(frozen=True)
class ExperienceWindow:
    """An ordered sequence of actual observations and proposed actions."""

    observations: tuple[observer.Observation, ...]
    actions: tuple[executor.Action, ...]
    pointers: tuple[tuple[float, float], ...]
    run_id: str
    steps: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class LoadedExperiences:
    """Contiguous H=1,2,3 windows, exclusions, and branch metadata."""

    split: str
    transition_count: int
    windows: dict[int, tuple[ExperienceWindow, ...]]
    exclusions: tuple[dict[str, Any], ...]
    branch_groups: dict[str, str | None]


@dataclasses.dataclass(frozen=True)
class RolloutCheckResult:
    """Result of verifying teacher-forced H-step imagination grounding."""

    completed_steps: int
    terminated_at_step: int | None
    termination_reason: str | None
    predictions: tuple[model.Prediction, ...]


def _action_from_record(raw_action: dict[str, Any]) -> executor.Action:
    return executor.Action(
        kind=str(raw_action['kind']),
        target=raw_action.get('target'),
        text=raw_action.get('text'),
        checked=raw_action.get('checked'),
        scroll_y=raw_action.get('scroll_y'),
    )


def _pre_action_pointer(record: dict[str, Any]) -> tuple[float, float] | None:
    path = record.get('result', {}).get('pointer_path')
    if not path:
        return None
    first = path[0]
    if 'x' not in first or 'y' not in first:
        return None
    return (float(first['x']), float(first['y']))


def load_experience_windows(
    raw_dir: pathlib.Path = DEFAULT_RAW_DIR,
    split: str = 'train',
    max_horizon: int = 3,
) -> LoadedExperiences:
    """Loads train/validation raw transitions into H=1..max_horizon windows."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f'Split {split!r} is not allowed; expected one of {ALLOWED_SPLITS}.'
        )
    if max_horizon < 1 or max_horizon > 3:
        raise ValueError(f'max_horizon must be in [1, 3], got {max_horizon}.')

    t_path = raw_dir / split / recorder.TRANSITIONS_FILE
    raw_records = [
        json.loads(line)
        for line in t_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    branch_groups: dict[str, str | None] = {}
    for rec in raw_records:
        ep_id = str(rec['episode_id'])
        run_id = str(rec['run_id'])
        grouped[(run_id, ep_id)].append(rec)
        branch = (rec.get('scenario') or {}).get('branch')
        branch_groups[run_id] = branch.get('group') if branch else None

    exclusions: list[dict[str, Any]] = []
    windows_by_h: dict[int, list[ExperienceWindow]] = {
        h: [] for h in range(1, max_horizon + 1)
    }

    for (run_id, ep_id), recs in sorted(grouped.items()):
        recs.sort(key=lambda r: int(r['step']))
        valid_recs: list[
            tuple[dict[str, Any], observer.Observation, observer.Observation,
                  executor.Action, tuple[float, float]]
        ] = []
        for rec in recs:
            step = int(rec['step'])
            if not rec.get('settled'):
                exclusions.append({
                    'run_id': run_id, 'episode_id': ep_id, 'step': step,
                    'reason': 'unsettled',
                })
                continue
            if not rec.get('result', {}).get('executed'):
                exclusions.append({
                    'run_id': run_id, 'episode_id': ep_id, 'step': step,
                    'reason': 'not_executed',
                })
                continue
            ptr = _pre_action_pointer(rec)
            if ptr is None:
                exclusions.append({
                    'run_id': run_id, 'episode_id': ep_id, 'step': step,
                    'reason': 'missing_pre_action_pointer',
                })
                continue
            b_obs = observer.observation_from_dict(rec['before'])
            a_obs = observer.observation_from_dict(rec['after'])
            act = _action_from_record(rec['action'])
            valid_recs.append((rec, b_obs, a_obs, act, ptr))

        # Split into maximally contiguous sub-runs with verified continuity.
        segments: list[list[Any]] = []
        for item in valid_recs:
            rec = item[0]
            if not segments:
                segments.append([item])
                continue
            prev_rec, _, prev_after, _, _ = segments[-1][-1]
            cont = rec.get('continuity') or {}
            is_contiguous = (
                int(rec['step']) == int(prev_rec['step']) + 1
                and bool(cont.get('continuous'))
                and cont.get('previous_step') == prev_rec['step']
                and not recorder.differences(
                    recorder.canonical(prev_after),
                    recorder.canonical(item[1]),
                )
            )
            if is_contiguous:
                segments[-1].append(item)
            else:
                exclusions.append({
                    'run_id': run_id, 'episode_id': ep_id,
                    'step': int(rec['step']),
                    'reason': 'discontinuous_transition',
                })
                segments.append([item])

        for seg in segments:
            for start in range(len(seg)):
                for horizon in range(1, max_horizon + 1):
                    end = start + horizon
                    if end > len(seg):
                        break
                    chunk = seg[start:end]
                    obs_seq = [chunk[0][1]] + [c[2] for c in chunk]
                    act_seq = [c[3] for c in chunk]
                    ptr_seq = [c[4] for c in chunk]
                    step_seq = [int(c[0]['step']) for c in chunk]
                    windows_by_h[horizon].append(
                        ExperienceWindow(
                            observations=tuple(obs_seq),
                            actions=tuple(act_seq),
                            pointers=tuple(ptr_seq),
                            run_id=run_id,
                            steps=tuple(step_seq),
                        )
                    )

    return LoadedExperiences(
        split=split,
        transition_count=len(raw_records),
        windows={h: tuple(v) for h, v in windows_by_h.items()},
        exclusions=tuple(exclusions),
        branch_groups=branch_groups,
    )


def resolve_observed_node_id(
    slot: int,
    aligned: observation.AlignedObservation,
) -> int:
    """Resolves a slot to a real observed node_id; rejects imagined slots."""
    if slot < 0 or slot >= len(aligned.inputs.slot_to_element):
        raise ValueError(f'Invalid slot index {slot}.')
    elem_idx = aligned.inputs.slot_to_element[slot]
    if elem_idx is None:
        raise ValueError(
            f'Imagined or unoccupied slot {slot} cannot be used as a real '
            f'browser execution target.'
        )
    return aligned.observation.elements[elem_idx].node_id


def trace_target_to_initial_slot(
    window: ExperienceWindow,
    step_idx: int,
    initial_aligned: observation.AlignedObservation,
) -> int | None:
    """Traces action[step_idx].target back to its slot in initial_aligned."""
    act = window.actions[step_idx]
    if act.kind == executor.SCROLL:
        return -1
    if act.target is None:
        return None
    obs_t = window.observations[step_idx]
    cur_elem_idx = next(
        (i for i, e in enumerate(obs_t.elements) if e.node_id == act.target),
        None,
    )
    if cur_elem_idx is None:
        return None
    for k in range(step_idx, 0, -1):
        align = curate.align(
            window.observations[k - 1], window.observations[k]
        )
        prev_idx = align.after_to_before[cur_elem_idx]
        if prev_idx < 0:
            return None
        cur_elem_idx = prev_idx
    return initial_aligned.inputs.element_to_slot[cur_elem_idx]


def rollout_imagined_window(
    world: model.WorldActionModel,
    window: ExperienceWindow,
    encoder: observation.TextEncoder,
    role_to_index: dict[str, int],
    slot_count: int = model.SLOT_COUNT,
) -> RolloutCheckResult:
    """Runs recursive imagination on a window without future observe() leakage.

    Stops cleanly if a future action targets an entity not present in the
    initial observation.
    """
    if not window.actions:
        return RolloutCheckResult(0, None, None, ())

    initial_aligned = observation.encode_observation(
        window.observations[0],
        encoder,
        role_to_index,
        slot_count=slot_count,
        previous=None,
    )
    belief = observation.observe(world, initial_aligned, previous=None)
    preds: list[model.Prediction] = []

    for step_idx, (act, ptr) in enumerate(zip(window.actions, window.pointers)):
        slot = trace_target_to_initial_slot(
            window, step_idx, initial_aligned
        )
        if slot is None:
            return RolloutCheckResult(
                completed_steps=step_idx,
                terminated_at_step=step_idx,
                termination_reason='unknown_future_entity',
                predictions=tuple(preds),
            )

        # Build action inputs using only initial observation geometry/text
        # and the imagined belief's slot representation.
        if step_idx == 0 or slot < 0:
            act_for_step = act
        else:
            init_elem_idx = initial_aligned.inputs.slot_to_element[slot]
            if init_elem_idx is None:
                raise RuntimeError(
                    f'Initial slot {slot} has no mapped element.'
                )
            act_for_step = dataclasses.replace(
                act,
                target=initial_aligned.observation.elements[
                    init_elem_idx
                ].node_id,
            )
        act_inputs = observation.encode_action(
            act_for_step, initial_aligned, encoder, ptr
        )
        action_latent = world.encode_action(
            belief,
            act_inputs.kind,
            act_inputs.target,
            act_inputs.text,
            act_inputs.checked,
            act_inputs.scroll,
            act_inputs.pointer,
        )
        pred = world.imagine(belief, action_latent, act_inputs.target)
        preds.append(pred)
        belief = pred.belief

    return RolloutCheckResult(
        completed_steps=len(preds),
        terminated_at_step=None,
        termination_reason=None,
        predictions=tuple(preds),
    )
