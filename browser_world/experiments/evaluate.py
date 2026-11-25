"""Validation decoding and detailed metrics for WorldModel."""

from collections.abc import Sequence
from typing import Any

import torch
from torch.utils import data as torch_data

from browser_world.dataset import schema
from browser_world.dataset import training_data
from browser_world.experiments import baselines
from browser_world.model import losses
from browser_world.model import world_model


def _f1_stats(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denominator = 2 * tp + fp + fn
    f1 = 2 * tp / denominator if denominator else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def model_inputs(
    batch: dict[str, torch.Tensor],
    *,
    action_blind: bool = False,
) -> dict[str, torch.Tensor]:
    """Selects observable inputs for the requested experiment."""
    inputs = {
        key: batch[key]
        for key in world_model.INPUT_KEYS
    }
    if action_blind:
        inputs['action_kind'] = torch.full_like(
            inputs['action_kind'],
            schema.ACTION_KINDS.index('scroll'),
        )
        inputs['action_target'] = torch.full_like(
            inputs['action_target'],
            -1,
        )
        for key in (
            'action_text_bytes',
            'action_checked',
            'action_scroll',
            'action_pointer',
        ):
            inputs[key] = torch.zeros_like(inputs[key])
    return inputs


def decode_predictions(
    predictions: dict[str, torch.Tensor],
    element_mask: torch.Tensor,
) -> list[dict[str, object]]:
    """Decodes a K=1 batch of model logits into baseline-format dicts."""
    if element_mask.ndim != 3 or element_mask.shape[1] != 1:
        raise ValueError('decode_predictions requires [B, 1, N] element_mask.')

    mask_cpu = element_mask[:, 0].detach().cpu()
    changed = (predictions['changed_logits'][:, 0] >= 0).detach().cpu()
    removed = (predictions['removed_logits'][:, 0] >= 0).detach().cpu()
    next_states = (
        predictions['next_state_logits'][:, 0] >= 0
    ).detach().cpu()
    value_source = (
        predictions['value_logits'][:, 0].argmax(dim=-1)
    ).detach().cpu()
    box_delta = predictions['box_delta'][:, 0].detach().cpu()
    outcomes = (predictions['outcome_logits'][:, 0] >= 0).detach().cpu()
    new_count = predictions['new_count'][:, 0].detach().cpu()

    decoded: list[dict[str, object]] = []
    for batch_idx in range(mask_cpu.shape[0]):
        real = mask_cpu[batch_idx]
        decoded.append({
            'changed': changed[batch_idx][real].to(torch.int64).tolist(),
            'removed': removed[batch_idx][real].to(torch.int64).tolist(),
            'next_states': (
                next_states[batch_idx][real].to(torch.int64).tolist()
            ),
            'value_source': value_source[batch_idx][real].tolist(),
            'box_delta': box_delta[batch_idx][real].tolist(),
            'new_count': float(new_count[batch_idx].item()),
            'outcomes': outcomes[batch_idx].to(torch.int64).tolist(),
        })
    return decoded


def evaluate_model(
    model: world_model.WorldModel,
    loader: torch_data.DataLoader,
    dataset: training_data.Windows,
    examples: Sequence[dict[str, Any]],
    device: torch.device,
    role_names: Sequence[str] | None = None,
    *,
    action_blind: bool = False,
) -> dict[str, Any]:
    """Evaluates WorldModel on a validation/test split without gradients."""
    model.eval()
    loss_totals: dict[str, float] = {}
    batch_count = 0
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    role_preds: dict[tuple[str, int], list[float]] = {}
    offset = 0

    with torch.inference_mode():
        for cpu_batch in loader:
            batch = {
                key: value.to(device)
                for key, value in cpu_batch.items()
            }
            inputs = model_inputs(batch, action_blind=action_blind)
            predictions = model(**inputs)
            total_loss, terms = losses.prediction_loss(predictions, batch)

            loss_totals['total'] = (
                loss_totals.get('total', 0.0) + float(total_loss.item())
            )
            for name, val in terms.items():
                loss_totals[name] = (
                    loss_totals.get(name, 0.0) + float(val.item())
                )
            batch_count += 1

            decoded = decode_predictions(predictions, batch['element_mask'])
            batch_roles = (
                predictions['new_role_counts'][:, 0].detach().cpu().tolist()
            )
            for item_idx, item_pred in enumerate(decoded):
                meta = dataset.meta[offset + item_idx][0]
                key = (str(meta['run_id']), int(meta['step']))
                if key in by_key:
                    raise ValueError(f'Duplicate prediction key: {key}')
                by_key[key] = item_pred
                role_preds[key] = batch_roles[item_idx]
            offset += len(decoded)

    if batch_count == 0:
        raise ValueError('Evaluation loader is empty.')
    if offset != len(dataset) or len(by_key) != len(examples):
        raise ValueError(
            f'Prediction count mismatch: got {len(by_key)}, '
            f'expected {len(examples)}.'
        )

    def predict_fn(example: dict[str, Any]) -> dict[str, object]:
        meta = example['meta']
        key = (str(meta['run_id']), int(meta['step']))
        return by_key[key]

    base_metrics = baselines.evaluate(examples, predict_fn)

    nt_tp = nt_fp = nt_fn = 0
    surv_nt_tp = surv_nt_fp = surv_nt_fn = 0
    bac_tp = bac_fp = bac_fn = 0
    removed_not_changed = 0
    outcome_counts = [[0, 0, 0] for _ in schema.OUTCOMES]
    state_counts = [[0, 0, 0] for _ in schema.STATE_NAMES]
    action_counts: dict[str, dict[str, list[int] | int]] = {
        kind: {
            'transitions': 0,
            'chg': [0, 0, 0],
            'nt': [0, 0, 0],
            'surv_nt': [0, 0, 0],
            'rem': [0, 0, 0],
        }
        for kind in schema.ACTION_KINDS
    }
    run_counts: dict[str, dict[str, list[int] | int]] = {}
    num_non_pad_roles = len(next(iter(role_preds.values())))
    role_abs_errors = [0.0] * num_non_pad_roles

    for example in examples:
        meta = example['meta']
        run_id = str(meta['run_id'])
        key = (run_id, int(meta['step']))
        pred = by_key[key]
        action_target = int(example['inputs']['action']['target'])
        action_kind = schema.ACTION_KINDS[
            int(example['inputs']['action']['kind'])
        ]
        copy_pred = baselines.predict_action_copy(example)

        act_bucket = action_counts[action_kind]
        act_bucket['transitions'] = int(act_bucket['transitions']) + 1
        run_bucket = run_counts.setdefault(
            run_id, {'transitions': 0, 'chg': [0, 0, 0], 'nt': [0, 0, 0]},
        )
        run_bucket['transitions'] = int(run_bucket['transitions']) + 1

        for idx, target in enumerate(example['targets']['elements']):
            if not target['known']:
                continue
            pred_changed = bool(pred['changed'][idx])  # type: ignore[index]
            pred_removed = bool(pred['removed'][idx])  # type: ignore[index]
            true_changed = bool(target['changed'])
            true_removed = bool(target['removed'])

            act_chg = act_bucket['chg']
            act_rem = act_bucket['rem']
            run_chg = run_bucket['chg']
            assert isinstance(act_chg, list) and isinstance(act_rem, list)
            assert isinstance(run_chg, list)
            act_chg[0] += int(true_changed and pred_changed)
            act_chg[1] += int(pred_changed and not true_changed)
            act_chg[2] += int(true_changed and not pred_changed)
            act_rem[0] += int(true_removed and pred_removed)
            act_rem[1] += int(pred_removed and not true_removed)
            act_rem[2] += int(true_removed and not pred_removed)
            run_chg[0] += int(true_changed and pred_changed)
            run_chg[1] += int(pred_changed and not true_changed)
            run_chg[2] += int(true_changed and not pred_changed)

            if pred_removed and not pred_changed:
                removed_not_changed += 1

            if target['matched']:
                cur_states = example['inputs']['state']['elements'][idx]['states']
                true_next = target['next_states']
                pred_next = pred['next_states'][idx]  # type: ignore[index]
                for s_idx in range(len(schema.STATE_NAMES)):
                    t_sc = bool(true_next[s_idx] != cur_states[s_idx])
                    p_sc = bool(pred_next[s_idx] != cur_states[s_idx])
                    state_counts[s_idx][0] += int(t_sc and p_sc)
                    state_counts[s_idx][1] += int(p_sc and not t_sc)
                    state_counts[s_idx][2] += int(t_sc and not p_sc)

            if idx != action_target:
                nt_tp += int(true_changed and pred_changed)
                nt_fp += int(pred_changed and not true_changed)
                nt_fn += int(true_changed and not pred_changed)

                act_nt = act_bucket['nt']
                run_nt = run_bucket['nt']
                assert isinstance(act_nt, list) and isinstance(run_nt, list)
                act_nt[0] += int(true_changed and pred_changed)
                act_nt[1] += int(pred_changed and not true_changed)
                act_nt[2] += int(true_changed and not pred_changed)
                run_nt[0] += int(true_changed and pred_changed)
                run_nt[1] += int(pred_changed and not true_changed)
                run_nt[2] += int(true_changed and not pred_changed)

                if target['matched']:
                    surv_nt_tp += int(true_changed and pred_changed)
                    surv_nt_fp += int(pred_changed and not true_changed)
                    surv_nt_fn += int(true_changed and not pred_changed)
                    act_snt = act_bucket['surv_nt']
                    assert isinstance(act_snt, list)
                    act_snt[0] += int(true_changed and pred_changed)
                    act_snt[1] += int(pred_changed and not true_changed)
                    act_snt[2] += int(true_changed and not pred_changed)

                if (action_kind != 'scroll'
                        and not bool(copy_pred['changed'][idx])):
                    bac_tp += int(true_changed and pred_changed)
                    bac_fp += int(pred_changed and not true_changed)
                    bac_fn += int(true_changed and not pred_changed)

        for out_idx, (truth, guess) in enumerate(zip(
            example['targets']['outcomes'],
            pred['outcomes'],  # type: ignore[arg-type]
        )):
            t_bool, g_bool = bool(truth), bool(guess)
            outcome_counts[out_idx][0] += int(t_bool and g_bool)
            outcome_counts[out_idx][1] += int(g_bool and not t_bool)
            outcome_counts[out_idx][2] += int(t_bool and not g_bool)

        true_role_counts = [0.0] * num_non_pad_roles
        for new_elem in example['targets']['new']:
            role_id = int(new_elem['role'])
            if 1 <= role_id <= num_non_pad_roles:
                true_role_counts[role_id - 1] += 1.0
        for r_idx, (p_val, t_val) in enumerate(
            zip(role_preds[key], true_role_counts)
        ):
            role_abs_errors[r_idx] += abs(p_val - t_val)

    per_outcome: dict[str, dict[str, float | int]] = {}
    for out_idx, name in enumerate(schema.OUTCOMES):
        tp, fp, fn = outcome_counts[out_idx]
        stats = _f1_stats(tp, fp, fn)
        stats['support'] = tp + fn
        per_outcome[name] = stats

    state_changes: dict[str, dict[str, float | int]] = {}
    for s_idx, name in enumerate(schema.STATE_NAMES):
        tp, fp, fn = state_counts[s_idx]
        stats = _f1_stats(tp, fp, fn)
        stats['support'] = tp + fn
        state_changes[name] = stats

    per_action: dict[str, dict[str, float | int]] = {}
    for kind, bkt in action_counts.items():
        chg_s = _f1_stats(*bkt['chg'])  # type: ignore[misc]
        nt_s = _f1_stats(*bkt['nt'])  # type: ignore[misc]
        snt_s = _f1_stats(*bkt['surv_nt'])  # type: ignore[misc]
        rem_s = _f1_stats(*bkt['rem'])  # type: ignore[misc]
        per_action[kind] = {
            'transitions': int(bkt['transitions']),
            'changed_f1': chg_s['f1'],
            'non_target_changed_f1': nt_s['f1'],
            'surviving_non_target_f1': snt_s['f1'],
            'removed_f1': rem_s['f1'],
            'changed_support': chg_s['tp'] + chg_s['fn'],
            'non_target_support': nt_s['tp'] + nt_s['fn'],
            'surviving_non_target_support': snt_s['tp'] + snt_s['fn'],
            'removed_support': rem_s['tp'] + rem_s['fn'],
        }

    per_run: dict[str, dict[str, float | int]] = {}
    for rid, bkt in run_counts.items():
        chg_s = _f1_stats(*bkt['chg'])  # type: ignore[misc]
        nt_s = _f1_stats(*bkt['nt'])  # type: ignore[misc]
        per_run[rid] = {
            'transitions': int(bkt['transitions']),
            'changed_f1': chg_s['f1'],
            'non_target_changed_f1': nt_s['f1'],
            'positive_support': chg_s['tp'] + chg_s['fn'],
        }

    labels = (
        list(role_names[1:])
        if role_names and len(role_names) - 1 == num_non_pad_roles
        else [f'role_{i}' for i in range(1, num_non_pad_roles + 1)]
    )
    n_examples = len(examples)
    per_role_mae = {
        label: err / n_examples
        for label, err in zip(labels, role_abs_errors)
    }

    surv_nt_stats = _f1_stats(surv_nt_tp, surv_nt_fp, surv_nt_fn)
    surv_nt_stats['support'] = surv_nt_tp + surv_nt_fn

    return {
        'loss': {name: val / batch_count for name, val in loss_totals.items()},
        'baseline_format': base_metrics,
        'non_target_changed': _f1_stats(nt_tp, nt_fp, nt_fn),
        'surviving_non_target_changed': surv_nt_stats,
        'beyond_action_copy': _f1_stats(bac_tp, bac_fp, bac_fn),
        'removed_not_changed_count': removed_not_changed,
        'state_changes': state_changes,
        'per_action': per_action,
        'per_run': per_run,
        'per_outcome': per_outcome,
        'per_role_new_count_mae': per_role_mae,
    }
