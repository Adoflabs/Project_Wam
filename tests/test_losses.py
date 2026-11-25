"""Unit tests for masked world-model losses."""

# pylint: disable=missing-function-docstring

import math

import torch

from browser_world.model import heads
from browser_world.model import losses


def _zero_predictions(batch=1, steps=1, count=2, num_roles=16):
    return {
        'changed_logits': torch.zeros((batch, steps, count), requires_grad=True),
        'removed_logits': torch.zeros((batch, steps, count), requires_grad=True),
        'next_state_logits': torch.zeros(
            (batch, steps, count, 8), requires_grad=True,
        ),
        'value_logits': torch.zeros(
            (batch, steps, count, 5), requires_grad=True,
        ),
        'box_delta': torch.zeros((batch, steps, count, 4), requires_grad=True),
        'outcome_logits': torch.zeros((batch, steps, 8), requires_grad=True),
        'new_role_counts': torch.zeros(
            (batch, steps, num_roles - 1), requires_grad=True,
        ),
        'new_count': torch.zeros((batch, steps), requires_grad=True),
    }


def _basic_batch(batch=1, steps=1, count=2, new_slots=2):
    return {
        'element_mask': torch.ones((batch, steps, count), dtype=torch.bool),
        't_known': torch.ones((batch, steps, count), dtype=torch.bool),
        't_matched': torch.ones((batch, steps, count), dtype=torch.bool),
        't_value_mask': torch.ones((batch, steps, count), dtype=torch.bool),
        't_box_mask': torch.ones((batch, steps, count), dtype=torch.bool),
        't_changed': torch.zeros((batch, steps, count), dtype=torch.float32),
        't_removed': torch.zeros((batch, steps, count), dtype=torch.float32),
        't_next_states': torch.zeros(
            (batch, steps, count, 8), dtype=torch.float32,
        ),
        't_value_source': torch.zeros((batch, steps, count), dtype=torch.long),
        't_box_delta': torch.zeros(
            (batch, steps, count, 4), dtype=torch.float32,
        ),
        't_outcomes': torch.zeros((batch, steps, 8), dtype=torch.float32),
        't_new_role': torch.zeros((batch, steps, new_slots), dtype=torch.long),
        't_new_mask': torch.zeros((batch, steps, new_slots), dtype=torch.bool),
        't_new_count': torch.zeros((batch, steps), dtype=torch.float32),
    }


def test_exact_hand_computable_losses_and_immutability():
    preds = _zero_predictions()
    batch = _basic_batch()
    pred_snapshot = {k: v.clone().detach() for k, v in preds.items()}
    batch_snapshot = {k: v.clone() for k, v in batch.items()}

    total, terms = losses.prediction_loss(preds, batch)

    assert math.isclose(terms['changed'].item(), math.log(2), rel_tol=1e-6)
    assert math.isclose(terms['removed'].item(), math.log(2), rel_tol=1e-6)
    assert math.isclose(terms['states'].item(), math.log(2), rel_tol=1e-6)
    assert math.isclose(terms['outcomes'].item(), math.log(2), rel_tol=1e-6)
    assert math.isclose(terms['value'].item(), math.log(5), rel_tol=1e-6)
    assert math.isclose(terms['box'].item(), 0.0, abs_tol=1e-7)
    assert math.isclose(terms['new_roles'].item(), 0.0, abs_tol=1e-7)
    assert math.isclose(terms['new_count'].item(), 0.0, abs_tol=1e-7)
    assert torch.isfinite(total)

    total.backward()
    for key, tensor in preds.items():
        if key not in ('box_delta', 'new_role_counts', 'new_count'):
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all()

    for k, v in preds.items():
        assert torch.equal(v.detach(), pred_snapshot[k])
    for k, v in batch.items():
        assert torch.equal(v, batch_snapshot[k])


def test_mask_exclusions_removed_padding_and_empty():
    preds = _zero_predictions(batch=1, steps=1, count=3)
    batch = _basic_batch(batch=1, steps=1, count=3, new_slots=0)

    # Entity 0: valid & matched. Entity 1: removed (t_matched=False) with
    # huge bogus next_states/value_source/box_delta. Entity 2: padded/unknown.
    batch['t_matched'][0, 0, 1] = False
    batch['t_next_states'][0, 0, 1] = 1.0
    batch['t_value_source'][0, 0, 1] = 4
    batch['t_box_delta'][0, 0, 1] = 100.0 * heads.BOX_DELTA_SCALE

    batch['element_mask'][0, 0, 2] = False
    batch['t_known'][0, 0, 2] = False
    batch['t_changed'][0, 0, 2] = 1.0
    batch['t_box_delta'][0, 0, 2] = 50.0 * heads.BOX_DELTA_SCALE

    with torch.no_grad():
        preds['changed_logits'][0, 0, 0] = 0.0
        preds['changed_logits'][0, 0, 1] = 0.0
        preds['changed_logits'][0, 0, 2] = 100.0

    total, terms = losses.prediction_loss(preds, batch)
    assert math.isclose(terms['changed'].item(), math.log(2), rel_tol=1e-6)
    assert math.isclose(terms['box'].item(), 0.0, abs_tol=1e-7)
    assert math.isclose(terms['value'].item(), math.log(5), rel_tol=1e-6)
    assert torch.isfinite(total)

    # Excluding entity 0 from t_known leaves only entity 1 (matched=False).
    batch['t_known'][0, 0, 0] = False
    _, terms_excl = losses.prediction_loss(preds, batch)
    assert math.isclose(terms_excl['states'].item(), 0.0, abs_tol=1e-7)
    assert math.isclose(terms_excl['value'].item(), 0.0, abs_tol=1e-7)
    assert math.isclose(terms_excl['box'].item(), 0.0, abs_tol=1e-7)

    # All-empty element mask produces zero entity losses and finite total.
    batch['element_mask'].zero_()
    total_empty, terms_empty = losses.prediction_loss(preds, batch)
    for k in ('changed', 'removed', 'states', 'value', 'box'):
        assert math.isclose(terms_empty[k].item(), 0.0, abs_tol=1e-7)
    assert torch.isfinite(total_empty)


def test_repeated_new_roles_histogram_accumulation():
    preds = _zero_predictions(batch=1, steps=1, count=1, num_roles=16)
    batch = _basic_batch(batch=1, steps=1, count=1, new_slots=4)

    # Roles 3, 3, 7 are active; slot 3 is masked (role 0) and ignored.
    batch['t_new_role'] = torch.tensor([[[3, 3, 7, 0]]], dtype=torch.long)
    batch['t_new_mask'] = torch.tensor([[[True, True, True, False]]])
    batch['t_new_count'] = torch.tensor([[3.0]], dtype=torch.float32)

    with torch.no_grad():
        preds['new_role_counts'][0, 0, 2] = 2.0  # role 3 -> index 2
        preds['new_role_counts'][0, 0, 6] = 1.0  # role 7 -> index 6
        preds['new_count'][0, 0] = 3.0

    _, terms = losses.prediction_loss(preds, batch)
    assert math.isclose(terms['new_roles'].item(), 0.0, abs_tol=1e-7)
    assert math.isclose(terms['new_count'].item(), 0.0, abs_tol=1e-7)
