"""Direction-aware dynamics, existence, and inverse losses for WAM."""

from __future__ import annotations

import dataclasses

import torch
from torch.nn import functional

from browser_world.wam2 import model
from browser_world.wam2 import observation

LOSS_WEIGHTS = {
    'existence': 1.0,
    'directional_states': 2.0,
    'birth_states': 1.0,
    'roles': 0.5,
    'flags': 0.5,
    'geometry': 0.25,
    'affordances': 1.0,
    'parents': 0.5,
    'latent_consistency': 1.0,
    'inverse_dynamics': 0.25,
}

DIRECTION_STRATA = ('0->0', '0->1', '1->0', '1->1')
EXISTENCE_STRATA = ('persistence', 'removal', 'birth', 'empty')


@dataclasses.dataclass(frozen=True)
class StepLosses:
    """Loss breakdown and stratum support for one imagined transition."""

    total: torch.Tensor
    selection_loss: torch.Tensor
    components: dict[str, torch.Tensor]
    direction_support: dict[str, int]
    existence_support: dict[str, int]


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Returns the mean over valid supervised scalar positions."""
    if mask.dtype != torch.bool:
        raise TypeError(f'Expected bool mask, got {mask.dtype}.')
    if mask.device != values.device:
        raise ValueError(
            f'Mask device {mask.device} differs from values {values.device}.'
        )
    expanded_mask = mask.expand_as(values)
    masked_values = torch.where(
        expanded_mask,
        values,
        torch.zeros_like(values),
    )
    valid_count = expanded_mask.sum()
    return masked_values.sum() / valid_count.clamp_min(1)


def directional_state_loss(
    pred: model.Prediction,
    prev_states: torch.Tensor,
    prev_present: torch.Tensor,
    next_inputs: observation.ObservationInputs,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Computes balanced BCE across 0->0, 0->1, 1->0, 1->1 state strata."""
    persist = (
        (prev_present > 0.5)
        & (next_inputs.present > 0.5)
        & next_inputs.existence_known
    ).unsqueeze(-1)
    prev_zero = prev_states < 0.5
    prev_one = prev_states >= 0.5
    nxt = next_inputs.states
    next_zero = nxt < 0.5
    next_one = nxt >= 0.5

    bce_z2o = functional.binary_cross_entropy_with_logits(
        pred.zero_to_one_logits, nxt, reduction='none'
    )
    bce_o2z = functional.binary_cross_entropy_with_logits(
        pred.one_to_zero_logits, 1.0 - nxt, reduction='none'
    )

    strata = (
        ('0->0', persist & prev_zero & next_zero, bce_z2o),
        ('0->1', persist & prev_zero & next_one, bce_z2o),
        ('1->0', persist & prev_one & next_zero, bce_o2z),
        ('1->1', persist & prev_one & next_one, bce_o2z),
    )
    support: dict[str, int] = {}
    terms: list[torch.Tensor] = []
    for name, mask, loss_tensor in strata:
        cnt = int(mask.sum().item())
        support[name] = cnt
        if cnt > 0:
            terms.append(masked_mean(loss_tensor, mask))

    if not terms:
        zero = (pred.zero_to_one_logits * 0.0).sum()
        return zero, support
    return torch.stack(terms).mean(), support


def existence_loss(
    pred: model.Prediction,
    prev_present: torch.Tensor,
    next_inputs: observation.ObservationInputs,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Computes balanced BCE across persistence, removal, birth, and empty."""
    known = next_inputs.existence_known
    prev_on = prev_present > 0.5
    next_on = next_inputs.present > 0.5
    bce = functional.binary_cross_entropy_with_logits(
        pred.present_logits, next_inputs.present, reduction='none'
    )

    strata = (
        ('persistence', known & prev_on & next_on),
        ('removal', known & prev_on & ~next_on),
        ('birth', known & ~prev_on & next_on),
        ('empty', known & ~prev_on & ~next_on),
    )
    support: dict[str, int] = {}
    terms: list[torch.Tensor] = []
    for name, mask in strata:
        cnt = int(mask.sum().item())
        support[name] = cnt
        if cnt > 0:
            terms.append(masked_mean(bce, mask))

    if not terms:
        zero = (pred.present_logits * 0.0).sum()
        return zero, support
    return torch.stack(terms).mean(), support


def _masked_categorical_ce(
    logits: torch.Tensor,
    one_hot_targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Computes cross-entropy from one-hot targets over masked [B,N] slots."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = one_hot_targets.argmax(dim=-1).reshape(-1)
    ce = functional.cross_entropy(flat_logits, flat_targets, reduction='none')
    return masked_mean(ce.reshape(mask.shape), mask)


def step_dynamics_losses(
    pred: model.Prediction,
    prev_present: torch.Tensor,
    prev_states: torch.Tensor,
    next_inputs: observation.ObservationInputs,
    target_latent: torch.Tensor,
) -> StepLosses:
    """Computes all weighted dynamics losses for one predicted step."""
    ex_loss, ex_sup = existence_loss(pred, prev_present, next_inputs)
    dir_loss, dir_sup = directional_state_loss(
        pred, prev_states, prev_present, next_inputs
    )

    known = next_inputs.existence_known
    present_mask = (next_inputs.present > 0.5) & known
    birth_mask = ((prev_present <= 0.5) & present_mask).unsqueeze(-1)
    birth_bce = functional.binary_cross_entropy_with_logits(
        pred.birth_state_logits, next_inputs.states, reduction='none'
    )
    birth_loss = masked_mean(birth_bce, birth_mask)

    role_loss = _masked_categorical_ce(
        pred.role_logits, next_inputs.roles, present_mask
    )

    flag_bce = functional.binary_cross_entropy_with_logits(
        pred.flag_logits, next_inputs.flags, reduction='none'
    )
    flag_loss = masked_mean(flag_bce, present_mask.unsqueeze(-1))

    box_l1 = functional.smooth_l1_loss(
        pred.belief.boxes, next_inputs.boxes, reduction='none'
    )
    box_loss = masked_mean(
        box_l1, (present_mask & next_inputs.box_known).unsqueeze(-1)
    )

    aff_bce = functional.binary_cross_entropy_with_logits(
        pred.affordance_logits, next_inputs.affordances, reduction='none'
    )
    aff_loss = masked_mean(aff_bce, present_mask.unsqueeze(-1))

    parent_loss = _masked_categorical_ce(
        pred.parent_logits,
        next_inputs.parents,
        present_mask & next_inputs.parent_known,
    )

    latent_l1 = functional.smooth_l1_loss(
        pred.belief.latent, target_latent.detach(), reduction='none'
    )
    latent_loss = masked_mean(latent_l1, present_mask.unsqueeze(-1))

    components = {
        'existence': ex_loss,
        'directional_states': dir_loss,
        'birth_states': birth_loss,
        'roles': role_loss,
        'flags': flag_loss,
        'geometry': box_loss,
        'affordances': aff_loss,
        'parents': parent_loss,
        'latent_consistency': latent_loss,
    }
    total = sum(LOSS_WEIGHTS[k] * v for k, v in components.items())
    selection_loss = dir_loss + ex_loss + aff_loss
    return StepLosses(
        total=total,
        selection_loss=selection_loss,
        components=components,
        direction_support=dir_sup,
        existence_support=ex_sup,
    )


def inverse_dynamics_loss(
    world: model.WorldActionModel,
    before_belief: model.Belief,
    after_belief: model.Belief,
    action_inputs: observation.ActionInputs,
) -> tuple[torch.Tensor, bool, bool]:
    """Computes inverse action kind and target cross-entropy."""
    kind_logits, target_logits = world.infer_action(before_belief, after_belief)
    kind_ce = functional.cross_entropy(kind_logits, action_inputs.kind)
    target_cls = torch.where(
        action_inputs.target < 0,
        torch.zeros_like(action_inputs.target),
        action_inputs.target + 1,
    )
    target_ce = functional.cross_entropy(target_logits, target_cls)
    loss = 0.5 * (kind_ce + target_ce)
    kind_ok = bool(kind_logits.argmax(dim=-1).eq(action_inputs.kind).all())
    target_ok = bool(target_logits.argmax(dim=-1).eq(target_cls).all())
    return loss, kind_ok, target_ok


def set_valued_content_loss(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    target_mask: torch.Tensor,
    field_eligible: torch.Tensor,
) -> torch.Tensor:
    """Computes logsumexp(legal) - logsumexp(valid_targets) over fields."""
    if legal_mask.dtype != torch.bool or target_mask.dtype != torch.bool:
        raise TypeError('legal_mask and target_mask must be bool tensors.')
    if field_eligible.dtype != torch.bool:
        raise TypeError('field_eligible must be a bool tensor.')
    effective_targets = target_mask & legal_mask
    if (field_eligible & ~effective_targets.any(dim=-1)).any():
        raise ValueError(
            'Eligible content field has no legal target source.'
        )

    neg_inf = -1e9
    legal_logits = logits.masked_fill(~legal_mask, neg_inf)
    target_logits = logits.masked_fill(~effective_targets, neg_inf)
    per_field = (
        torch.logsumexp(legal_logits, dim=-1)
        - torch.logsumexp(target_logits, dim=-1)
    )
    return masked_mean(per_field, field_eligible)
