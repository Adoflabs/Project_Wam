"""Post-training capability and action-dependence evaluation for WAM."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import pathlib
from typing import Any

import torch
from torch.nn import functional

from browser_world.dataset import schema
from browser_world.interaction import observer
from browser_world.wam2 import losses
from browser_world.wam2 import model
from browser_world.wam2 import observation
from browser_world.wam2 import semantic
from browser_world.wam2 import train
from browser_world.wam2 import training_data


def _prf(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        (2 * tp / (2 * tp + fp + fn))
        if (2 * tp + fp + fn) else None
    )
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'support': tp + fn,
        'precision': round(prec, 4) if prec is not None else None,
        'recall': round(rec, 4) if rec is not None else None,
        'f1': round(f1, 4) if f1 is not None else None,
    }


def _load_model_strict(
    ckpt_path: pathlib.Path,
    device: torch.device,
) -> tuple[model.WorldActionModel, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    world = model.WorldActionModel(
        role_count=len(ckpt['role_vocabulary']),
        slot_count=int(ckpt['slot_count']),
    ).to(device)
    world.load_state_dict(ckpt['model_state_dict'], strict=True)
    world.eval()
    return world, ckpt


def evaluate_horizon(
    world: model.WorldActionModel,
    windows: Sequence[train.EncodedWindow],
    horizon: int,
    cached_encoder: observation.TextEncoder,
    *,
    reobserve_intermediate: bool = False,
) -> dict[str, Any]:
    """Evaluates a list of windows at target horizon using threshold 0.5."""
    n_states = len(schema.STATE_NAMES)
    dir_counts = {
        d: [[0, 0, 0, 0] for _ in range(n_states)]
        for d in ('0->1', '1->0')
    }
    brier_prop_sum = [0.0] * n_states
    brier_prop_cnt = [0] * n_states
    persist_brier_sum = 0.0
    persist_brier_cnt = 0

    ex_tp = {'birth': 0, 'removal': 0, 'persistence': 0, 'empty': 0}
    ex_fp = {'birth': 0, 'removal': 0}
    ex_fn = {'birth': 0, 'removal': 0}
    ex_sup = {'birth': 0, 'removal': 0, 'persistence': 0, 'empty': 0}

    aff_bce_sum = 0.0
    aff_tp = aff_fp = aff_fn = 0
    flag_bce_sum = 0.0
    role_ok = role_tot = 0
    parent_ok = parent_tot = 0
    geom_l1_sum = 0.0
    geom_cnt = 0
    latent_l1_sum = 0.0
    latent_cnt = 0
    inv_k_ok = inv_t_ok = inv_tot = 0
    terminated = 0
    evaluated = 0

    with torch.no_grad():
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
                    if reobserve_intermediate:
                        act_in = observation.encode_action(
                            win.actions[step_idx],
                            win.aligned_obs[step_idx],
                            cached_encoder,
                            win.pointers[step_idx],
                        )
                        belief = actual_belief
                    else:
                        act_in, last_ptr = train._imagined_action_inputs(
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

                if step_idx == 0 and horizon == 1:
                    _, k_ok, t_ok = losses.inverse_dynamics_loss(
                        world, prev_actual, actual_belief, act_in
                    )
                    inv_k_ok += int(k_ok)
                    inv_t_ok += int(t_ok)
                    inv_tot += 1

                if step_idx == horizon - 1:
                    evaluated += 1
                    kmask = nxt.inputs.existence_known
                    prev_on = cur.inputs.present > 0.5
                    nxt_on = nxt.inputs.present > 0.5
                    pred_on = pred.belief.present >= 0.5

                    b_true = kmask & ~prev_on & nxt_on
                    b_pred = kmask & ~prev_on & pred_on
                    ex_sup['birth'] += int(b_true.sum().item())
                    ex_tp['birth'] += int((b_true & b_pred).sum().item())
                    ex_fp['birth'] += int((~b_true & b_pred).sum().item())
                    ex_fn['birth'] += int((b_true & ~b_pred).sum().item())

                    r_true = kmask & prev_on & ~nxt_on
                    r_pred = kmask & prev_on & ~pred_on
                    ex_sup['removal'] += int(r_true.sum().item())
                    ex_tp['removal'] += int((r_true & r_pred).sum().item())
                    ex_fp['removal'] += int((~r_true & r_pred).sum().item())
                    ex_fn['removal'] += int((r_true & ~r_pred).sum().item())

                    p_true = kmask & prev_on & nxt_on
                    ex_sup['persistence'] += int(p_true.sum().item())
                    ex_tp['persistence'] += int((p_true & pred_on).sum().item())

                    e_true = kmask & ~prev_on & ~nxt_on
                    ex_sup['empty'] += int(e_true.sum().item())
                    ex_tp['empty'] += int((e_true & ~pred_on).sum().item())

                    # Directional states & Brier on persistent matched entities.
                    if p_true.any():
                        p_s = pred.belief.states[p_true]
                        t_s = nxt.inputs.states[p_true]
                        c_s = cur.inputs.states[p_true]
                        init_s = win.aligned_obs[0].inputs.states[p_true]
                        persist_brier_sum += float(
                            ((init_s - t_s) ** 2).sum().item()
                        )
                        persist_brier_cnt += int(t_s.numel())

                        pred_bin = p_s >= 0.5
                        cur_bin = c_s >= 0.5
                        nxt_bin = t_s >= 0.5
                        for s_i in range(n_states):
                            sq_err = (p_s[:, s_i] - t_s[:, s_i]) ** 2
                            brier_prop_sum[s_i] += float(sq_err.sum().item())
                            brier_prop_cnt[s_i] += int(sq_err.numel())

                            # 0 -> 1
                            elig_01 = ~cur_bin[:, s_i]
                            if elig_01.any():
                                gt = nxt_bin[elig_01, s_i]
                                pr = pred_bin[elig_01, s_i]
                                d = dir_counts['0->1'][s_i]
                                d[0] += int((gt & pr).sum().item())
                                d[1] += int((~gt & pr).sum().item())
                                d[2] += int((gt & ~pr).sum().item())
                                d[3] += int((~gt & ~pr).sum().item())
                            # 1 -> 0
                            elig_10 = cur_bin[:, s_i]
                            if elig_10.any():
                                gt = ~nxt_bin[elig_10, s_i]
                                pr = ~pred_bin[elig_10, s_i]
                                d = dir_counts['1->0'][s_i]
                                d[0] += int((gt & pr).sum().item())
                                d[1] += int((~gt & pr).sum().item())
                                d[2] += int((gt & ~pr).sum().item())
                                d[3] += int((~gt & ~pr).sum().item())

                    pmask = nxt_on & kmask
                    if pmask.any():
                        aff_bce_sum += float(
                            functional.binary_cross_entropy_with_logits(
                                pred.affordance_logits[pmask],
                                nxt.inputs.affordances[pmask],
                            ).item()
                        )
                        ap = pred.belief.affordances[pmask] >= 0.5
                        at = nxt.inputs.affordances[pmask] >= 0.5
                        aff_tp += int((ap & at).sum().item())
                        aff_fp += int((ap & ~at).sum().item())
                        aff_fn += int((~ap & at).sum().item())

                        flag_bce_sum += float(
                            functional.binary_cross_entropy_with_logits(
                                pred.flag_logits[pmask],
                                nxt.inputs.flags[pmask],
                            ).item()
                        )
                        role_pred = pred.role_logits[pmask].argmax(dim=-1)
                        role_true = nxt.inputs.roles[pmask].argmax(dim=-1)
                        role_ok += int((role_pred == role_true).sum().item())
                        role_tot += int(role_true.numel())

                        lat_l1 = functional.smooth_l1_loss(
                            pred.belief.latent[pmask],
                            actual_belief.latent[pmask],
                        )
                        latent_l1_sum += float(lat_l1.item())
                        latent_cnt += 1

                    par_mask = pmask & nxt.inputs.parent_known
                    if par_mask.any():
                        pp = pred.parent_logits[par_mask].argmax(dim=-1)
                        pt = nxt.inputs.parents[par_mask].argmax(dim=-1)
                        parent_ok += int((pp == pt).sum().item())
                        parent_tot += int(pt.numel())

                    box_mask = pmask & nxt.inputs.box_known
                    if box_mask.any():
                        g_l1 = functional.smooth_l1_loss(
                            pred.belief.boxes[box_mask],
                            nxt.inputs.boxes[box_mask],
                        )
                        geom_l1_sum += float(g_l1.item())
                        geom_cnt += 1

                belief = pred.belief
            if stopped:
                continue

    per_prop: dict[str, Any] = {}
    micro = {'0->1': [0, 0, 0, 0], '1->0': [0, 0, 0, 0]}
    for s_i, name in enumerate(schema.STATE_NAMES):
        per_prop[name] = {
            '0->1': _prf(*dir_counts['0->1'][s_i]),
            '1->0': _prf(*dir_counts['1->0'][s_i]),
            'brier': (
                round(brier_prop_sum[s_i] / brier_prop_cnt[s_i], 4)
                if brier_prop_cnt[s_i] else None
            ),
            'brier_support': brier_prop_cnt[s_i],
        }
        for d in ('0->1', '1->0'):
            for k in range(4):
                micro[d][k] += dir_counts[d][s_i][k]

    tot_b_sum = sum(brier_prop_sum)
    tot_b_cnt = sum(brier_prop_cnt)
    return {
        'horizon': horizon,
        'threshold': 0.5,
        'reobserve_intermediate': reobserve_intermediate,
        'total_windows': len(windows),
        'evaluated_windows': evaluated,
        'terminated_unknown_future_entity': terminated,
        'brier_all': round(tot_b_sum / tot_b_cnt, 4) if tot_b_cnt else None,
        'brier_persist_baseline': (
            round(persist_brier_sum / persist_brier_cnt, 4)
            if persist_brier_cnt else None
        ),
        'brier_support': tot_b_cnt,
        'micro_directional': {
            '0->1': _prf(*micro['0->1']),
            '1->0': _prf(*micro['1->0']),
        },
        'per_property': per_prop,
        'existence': {
            'birth': _prf(
                ex_tp['birth'], ex_fp['birth'], ex_fn['birth'], 0
            ),
            'removal': _prf(
                ex_tp['removal'], ex_fp['removal'], ex_fn['removal'], 0
            ),
            'persistence_acc': (
                round(ex_tp['persistence'] / ex_sup['persistence'], 4)
                if ex_sup['persistence'] else None
            ),
            'persistence_support': ex_sup['persistence'],
            'empty_acc': (
                round(ex_tp['empty'] / ex_sup['empty'], 4)
                if ex_sup['empty'] else None
            ),
            'empty_support': ex_sup['empty'],
        },
        'other_properties': {
            'affordance_bce': (
                round(aff_bce_sum / evaluated, 4) if evaluated else None
            ),
            'affordance_f1': _prf(aff_tp, aff_fp, aff_fn, 0)['f1'],
            'flag_bce': (
                round(flag_bce_sum / evaluated, 4) if evaluated else None
            ),
            'role_acc': round(role_ok / role_tot, 4) if role_tot else None,
            'parent_acc': (
                round(parent_ok / parent_tot, 4) if parent_tot else None
            ),
            'geometry_l1': (
                round(geom_l1_sum / geom_cnt, 4) if geom_cnt else None
            ),
            'latent_l1': (
                round(latent_l1_sum / latent_cnt, 4) if latent_cnt else None
            ),
            'inverse_kind_acc': (
                round(inv_k_ok / inv_tot, 4) if inv_tot else None
            ),
            'inverse_target_acc': (
                round(inv_t_ok / inv_tot, 4) if inv_tot else None
            ),
        },
    }


def evaluate_context_adaptation(
    world: model.WorldActionModel,
    val_h2: Sequence[train.EncodedWindow],
    cached_encoder: observation.TextEncoder,
) -> dict[str, Any]:
    """Compares step-1 predictions with correct_context enabled vs disabled."""
    brier_on = brier_off = 0.0
    lat_on = lat_off = 0.0
    count = lat_cnt = 0

    with torch.no_grad():
        for win in val_h2:
            if win.grounded_slots[1] is None:
                continue
            obs0, obs1, obs2 = (
                win.aligned_obs[0], win.aligned_obs[1], win.aligned_obs[2]
            )
            b0 = observation.observe(world, obs0, previous=None)
            act0_lat = world.encode_action(
                b0, win.step0_action.kind, win.step0_action.target,
                win.step0_action.text, win.step0_action.checked,
                win.step0_action.scroll, win.step0_action.pointer,
            )
            p0 = world.imagine(b0, act0_lat, win.step0_action.target)
            b1 = observation.observe(world, obs1, previous=b0)
            known1 = (obs1.inputs.present > 0.5) & obs1.inputs.existence_known
            b1_corr = world.correct_context(b1, p0.belief, act0_lat, known1)
            b2 = observation.observe(world, obs2, previous=b1)

            act1_in = observation.encode_action(
                win.actions[1], obs1, cached_encoder, win.pointers[1]
            )
            act1_on = world.encode_action(
                b1_corr, act1_in.kind, act1_in.target, act1_in.text,
                act1_in.checked, act1_in.scroll, act1_in.pointer,
            )
            act1_off = world.encode_action(
                b1, act1_in.kind, act1_in.target, act1_in.text,
                act1_in.checked, act1_in.scroll, act1_in.pointer,
            )
            p1_on = world.imagine(b1_corr, act1_on, act1_in.target)
            p1_off = world.imagine(b1, act1_off, act1_in.target)

            pmask = (
                (obs1.inputs.present > 0.5)
                & (obs2.inputs.present > 0.5)
                & obs2.inputs.existence_known
            )
            if pmask.any():
                t_s = obs2.inputs.states[pmask]
                brier_on += float(
                    ((p1_on.belief.states[pmask] - t_s) ** 2).sum().item()
                )
                brier_off += float(
                    ((p1_off.belief.states[pmask] - t_s) ** 2).sum().item()
                )
                count += int(t_s.numel())
                lat_on += float(
                    functional.smooth_l1_loss(
                        p1_on.belief.latent[pmask], b2.latent[pmask]
                    ).item()
                )
                lat_off += float(
                    functional.smooth_l1_loss(
                        p1_off.belief.latent[pmask], b2.latent[pmask]
                    ).item()
                )
                lat_cnt += 1

    return {
        'evaluated_windows': lat_cnt,
        'brier_with_context': round(brier_on / count, 4) if count else None,
        'brier_without_context': round(brier_off / count, 4) if count else None,
        'latent_l1_with_context': (
            round(lat_on / lat_cnt, 4) if lat_cnt else None
        ),
        'latent_l1_without_context': (
            round(lat_off / lat_cnt, 4) if lat_cnt else None
        ),
    }


def evaluate_counterfactual_control(
    world: model.WorldActionModel,
    val_h1_raw: Sequence[training_data.ExperienceWindow],
    val_h1_enc: Sequence[train.EncodedWindow],
    branch_groups: dict[str, str | None],
) -> dict[str, Any]:
    """Evaluates action dependence on validation counterfactual branch."""
    dis_idx = schema.STATE_NAMES.index('disabled')
    sel_idx = schema.STATE_NAMES.index('selected')
    candidates: list[dict[str, Any]] = []

    with torch.no_grad():
        for raw_w, enc_w in zip(val_h1_raw, val_h1_enc):
            if branch_groups.get(raw_w.run_id) is None:
                continue
            b0 = observation.observe(world, enc_w.aligned_obs[0], previous=None)
            act_lat = world.encode_action(
                b0, enc_w.step0_action.kind, enc_w.step0_action.target,
                enc_w.step0_action.text, enc_w.step0_action.checked,
                enc_w.step0_action.scroll, enc_w.step0_action.pointer,
            )
            pred = world.imagine(b0, act_lat, enc_w.step0_action.target)
            obs0, obs1 = enc_w.aligned_obs[0], enc_w.aligned_obs[1]
            day_slot = next(
                (
                    obs0.inputs.element_to_slot[i]
                    for i, e in enumerate(obs0.observation.elements)
                    if e.role == 'combobox' and e.name == 'Day'
                ),
                None,
            )
            act = raw_w.actions[0]
            t_elem = next(
                (
                    e for e in obs0.observation.elements
                    if e.node_id == act.target
                ),
                None,
            )
            day_pred = (
                float(pred.belief.states[0, day_slot, dis_idx].item())
                if day_slot is not None else None
            )
            day_actual = (
                bool(obs1.inputs.states[0, day_slot, dis_idx].item() >= 0.5)
                if day_slot is not None else None
            )
            mean_sel = float(
                pred.belief.states[0, :len(obs0.observation.elements), sel_idx]
                .mean()
                .item()
            )
            candidates.append({
                'run_id': raw_w.run_id,
                'action': (
                    f'{act.kind} '
                    f'{t_elem.role if t_elem else ""}:'
                    f'{t_elem.name if t_elem else ""}'
                ),
                'day_disabled_pred_prob': (
                    round(day_pred, 4) if day_pred is not None else None
                ),
                'day_disabled_actual': day_actual,
                'mean_selected_prob': round(mean_sel, 4),
            })
    return {'group': 'branch-harbor-combined-30020', 'candidates': candidates}


def collect_concrete_examples(
    world: model.WorldActionModel,
    val_raw_by_h: dict[int, tuple[training_data.ExperienceWindow, ...]],
    val_enc_by_h: dict[int, tuple[train.EncodedWindow, ...]],
    cached_encoder: observation.TextEncoder,
) -> dict[str, Any]:
    """Collects actual validation prediction examples for Section 17."""
    examples: dict[str, Any] = {}
    with torch.no_grad():
        for raw_w, enc_w in zip(val_raw_by_h[1], val_enc_by_h[1]):
            b0 = observation.observe(world, enc_w.aligned_obs[0], previous=None)
            act_lat = world.encode_action(
                b0, enc_w.step0_action.kind, enc_w.step0_action.target,
                enc_w.step0_action.text, enc_w.step0_action.checked,
                enc_w.step0_action.scroll, enc_w.step0_action.pointer,
            )
            pred = world.imagine(b0, act_lat, enc_w.step0_action.target)
            obs0, obs1 = enc_w.aligned_obs[0], enc_w.aligned_obs[1]
            act = raw_w.actions[0]
            t_elem = next(
                (e for e in obs0.observation.elements if e.node_id == act.target),
                None,
            )
            act_desc = (
                f'{act.kind} {t_elem.role}:{t_elem.name}'
                if t_elem else act.kind
            )
            pmask = (
                (obs0.inputs.present[0] > 0.5)
                & (obs1.inputs.present[0] > 0.5)
                & obs1.inputs.existence_known[0]
            )
            for slot in torch.nonzero(pmask, as_tuple=False).flatten().tolist():
                el_i = obs0.inputs.slot_to_element[slot]
                if el_i is None:
                    continue
                el = obs0.observation.elements[el_i]
                for s_i, s_name in enumerate(schema.STATE_NAMES):
                    c_v = bool(obs0.inputs.states[0, slot, s_i].item() >= 0.5)
                    a_v = bool(obs1.inputs.states[0, slot, s_i].item() >= 0.5)
                    p_prob = float(pred.belief.states[0, slot, s_i].item())
                    p_v = p_prob >= 0.5
                    item = {
                        'run_id': raw_w.run_id,
                        'step': raw_w.steps[0],
                        'action': act_desc,
                        'entity': f'{el.role}:{el.name}',
                        'property': s_name,
                        'initial': c_v,
                        'predicted_prob': round(p_prob, 4),
                        'predicted': p_v,
                        'actual': a_v,
                    }
                    if not c_v and a_v and p_v and 'activation_hit' not in examples:
                        examples['activation_hit'] = item
                    elif c_v and not a_v and not p_v and 'invalidation_hit' not in examples:
                        examples['invalidation_hit'] = item
                    elif not c_v and not a_v and not p_v and 'preserved_unrelated' not in examples:
                        examples['preserved_unrelated'] = item
                    elif c_v == a_v and p_v != a_v and 'false_change' not in examples:
                        examples['false_change'] = item
                    elif c_v != a_v and p_v != a_v and 'missed_change' not in examples:
                        examples['missed_change'] = item

            # Birth prediction check
            birth_mask = (
                (obs0.inputs.present[0] <= 0.5)
                & (obs1.inputs.present[0] > 0.5)
                & obs1.inputs.existence_known[0]
            )
            if birth_mask.any() and 'birth_example' not in examples:
                b_slot = int(torch.nonzero(birth_mask, as_tuple=False)[0].item())
                b_prob = float(pred.belief.present[0, b_slot].item())
                examples['birth_example'] = {
                    'run_id': raw_w.run_id,
                    'step': raw_w.steps[0],
                    'action': act_desc,
                    'slot': b_slot,
                    'predicted_present_prob': round(b_prob, 4),
                    'actual_present': True,
                }

        # Recursive H=3 error example
        for raw_w, enc_w in zip(val_raw_by_h[3], val_enc_by_h[3]):
            if all(s is not None for s in enc_w.grounded_slots):
                b = observation.observe(world, enc_w.aligned_obs[0], previous=None)
                last_ptr = raw_w.pointers[0]
                for st in range(3):
                    if st == 0:
                        ai = enc_w.step0_action
                    else:
                        ai, last_ptr = train._imagined_action_inputs(
                            enc_w, st, b, last_ptr, cached_encoder
                        )
                    al = world.encode_action(
                        b, ai.kind, ai.target, ai.text, ai.checked,
                        ai.scroll, ai.pointer,
                    )
                    pr = world.imagine(b, al, ai.target)
                    b = pr.belief
                o3 = enc_w.aligned_obs[3]
                examples['recursive_h3_example'] = {
                    'run_id': raw_w.run_id,
                    'steps': list(raw_w.steps),
                    'h3_state_brier': round(float((
                        (b.states[0, :len(o3.observation.elements)]
                         - o3.inputs.states[0, :len(o3.observation.elements)])
                        ** 2
                    ).mean().item()), 4),
                }
                break
    return examples


def evaluate_checkpoint_suite(
    ckpt_path: pathlib.Path,
    device: torch.device,
) -> dict[str, Any]:
    """Runs full Sections 13-17 validation evaluation for one checkpoint."""
    world, ckpt = _load_model_strict(ckpt_path, device)
    role_map = {r: i for i, r in enumerate(ckpt['role_vocabulary'])}
    val_raw = training_data.load_experience_windows(
        split='validation', max_horizon=3
    )
    sem_enc = semantic.SemanticEncoder(device)
    va_cache, _ = train.build_split_text_cache(val_raw, sem_enc)
    del sem_enc
    va_enc = train.CachedTextEncoder(va_cache, device)

    val_enc_by_h = {
        h: train.encode_windows(val_raw.windows[h], va_enc, role_map)
        for h in (1, 2, 3)
    }

    horizons = {
        f'H={h}': evaluate_horizon(world, val_enc_by_h[h], h, va_enc)
        for h in (1, 2, 3)
    }

    # Common trajectory cohort supporting H=1, 2, and 3.
    h3_keys = {
        (r.run_id, r.steps[0])
        for r, e in zip(val_raw.windows[3], val_enc_by_h[3])
        if all(s is not None for s in e.grounded_slots)
    }
    common_cohort: dict[str, Any] = {'cohort_count': len(h3_keys)}
    for h in (1, 2, 3):
        sub = [
            e for r, e in zip(val_raw.windows[h], val_enc_by_h[h])
            if (r.run_id, r.steps[0]) in h3_keys
        ]
        common_cohort[f'H={h}_open_loop'] = evaluate_horizon(
            world, sub, h, va_enc, reobserve_intermediate=False
        )
        if h >= 2:
            common_cohort[f'H={h}_reobserved'] = evaluate_horizon(
                world, sub, h, va_enc, reobserve_intermediate=True
            )

    return {
        'checkpoint': str(ckpt_path),
        'epoch': int(ckpt['epoch']),
        'loss_convention_eval': train.evaluate_validation(
            world, val_enc_by_h, va_enc
        ),
        'horizons': horizons,
        'common_cohort': common_cohort,
        'counterfactual_control': evaluate_counterfactual_control(
            world, val_raw.windows[1], val_enc_by_h[1], val_raw.branch_groups
        ),
        'context_adaptation': evaluate_context_adaptation(
            world, val_enc_by_h[2], va_enc
        ),
        'concrete_examples': collect_concrete_examples(
            world, val_raw.windows, val_enc_by_h, va_enc
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--v2-ckpt',
        default='artifacts/wam2/dynamics_v2_corrected/best.pt',
    )
    parser.add_argument(
        '--v1-ckpt',
        default='artifacts/wam2/dynamics_v1/last.pt',
    )
    parser.add_argument(
        '--out',
        default='artifacts/wam2/dynamics_v2_corrected/evaluation.json',
    )
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report = {
        'v2_corrected_best': evaluate_checkpoint_suite(
            pathlib.Path(args.v2_ckpt), device
        ),
        'v1_historical_last': evaluate_checkpoint_suite(
            pathlib.Path(args.v1_ckpt), device
        ),
    }
    out_path = pathlib.Path(args.out)
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved evaluation to', out_path)


if __name__ == '__main__':
    main()
