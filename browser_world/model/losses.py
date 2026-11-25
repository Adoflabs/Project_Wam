"""Masked supervision of observable world-model predictions."""

from collections.abc import Mapping

import torch
from torch.nn import functional

from browser_world.model import heads


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Averages valid entries and returns zero for an empty mask."""
    selected = values.masked_select(mask)
    return selected.sum() / max(selected.numel(), 1)


def prediction_loss(
    predictions: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Returns the total loss and individual supervision losses.

    Args:
      predictions: Outputs from WorldModel.
      batch: Collated current inputs and supervised targets.

    Returns:
      A differentiable scalar loss and named scalar loss terms.

    Targets are used only here, after prediction has completed.
    """
    known = batch['element_mask'] & batch['t_known']
    matched = known & batch['t_matched']
    value_mask = matched & batch['t_value_mask']
    box_mask = matched & batch['t_box_mask']

    changed_loss = _masked_mean(
        functional.binary_cross_entropy_with_logits(
            predictions['changed_logits'],
            batch['t_changed'],
            reduction='none',
        ),
        known,
    )
    removed_loss = _masked_mean(
        functional.binary_cross_entropy_with_logits(
            predictions['removed_logits'],
            batch['t_removed'],
            reduction='none',
        ),
        known,
    )
    states_loss = _masked_mean(
        functional.binary_cross_entropy_with_logits(
            predictions['next_state_logits'],
            batch['t_next_states'],
            reduction='none',
        ),
        matched.unsqueeze(-1),
    )
    value_targets = batch['t_value_source'][value_mask]
    value_loss = functional.cross_entropy(
        predictions['value_logits'][value_mask],
        value_targets,
        reduction='sum',
    ) / max(value_targets.numel(), 1)
    box_loss = _masked_mean(
        functional.smooth_l1_loss(
            predictions['box_delta'] / heads.BOX_DELTA_SCALE,
            batch['t_box_delta'] / heads.BOX_DELTA_SCALE,
            reduction='none',
        ),
        box_mask.unsqueeze(-1),
    )
    outcome_loss = functional.binary_cross_entropy_with_logits(
        predictions['outcome_logits'],
        batch['t_outcomes'],
    )
    role_targets = torch.zeros_like(predictions['new_role_counts'])
    role_indices = (batch['t_new_role'] - 1).clamp_min(0)
    role_targets.scatter_add_(
        dim=-1,
        index=role_indices,
        src=batch['t_new_mask'].to(role_targets.dtype),
    )
    new_roles_loss = functional.smooth_l1_loss(
        torch.log1p(predictions['new_role_counts']),
        torch.log1p(role_targets),
    )
    new_count_loss = functional.smooth_l1_loss(
        torch.log1p(predictions['new_count']),
        torch.log1p(batch['t_new_count']),
    )

    terms = {
        'changed': changed_loss,
        'removed': removed_loss,
        'states': states_loss,
        'value': value_loss,
        'box': box_loss,
        'outcomes': outcome_loss,
        'new_roles': new_roles_loss,
        'new_count': new_count_loss,
    }
    total = torch.stack(tuple(terms.values())).sum()
    return total, terms
