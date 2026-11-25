"""Unit tests for PredictionHeads."""

# pylint: disable=missing-function-docstring

import pytest
import torch

from browser_world.model import encoding
from browser_world.model import heads


def _synthetic_head_inputs(batch=2, steps=3, count=5):
    torch.manual_seed(0)
    predicted = torch.randn(batch, steps, count, encoding.MODEL_DIM)
    actions = torch.randn(batch, steps, encoding.MODEL_DIM)
    page = torch.rand(batch, steps, 4)
    element_mask = torch.ones(batch, steps, count, dtype=torch.bool)
    return predicted, actions, page, element_mask


def test_heads_shapes_parameter_count_and_counts():
    torch.manual_seed(0)
    head = heads.PredictionHeads(num_roles=16)
    assert sum(p.numel() for p in head.parameters()) == 38826

    with pytest.raises(ValueError, match='roles'):
        heads.PredictionHeads(num_roles=1)

    predicted, actions, page, element_mask = _synthetic_head_inputs()
    out = head(predicted, actions, page, element_mask)

    assert out['changed_logits'].shape == (2, 3, 5)
    assert out['removed_logits'].shape == (2, 3, 5)
    assert out['next_state_logits'].shape == (2, 3, 5, 8)
    assert out['value_logits'].shape == (2, 3, 5, 5)
    assert out['box_delta'].shape == (2, 3, 5, 4)
    assert out['outcome_logits'].shape == (2, 3, 8)
    assert out['new_role_counts'].shape == (2, 3, 15)
    assert out['new_count'].shape == (2, 3)

    for tensor in out.values():
        assert torch.isfinite(tensor).all()
    assert (out['new_role_counts'] >= 0.0).all()
    assert (out['new_count'] >= 0.0).all()
    torch.testing.assert_close(
        out['new_role_counts'].sum(dim=-1),
        out['new_count'],
    )


def test_heads_padding_invariance_and_entirely_masked():
    torch.manual_seed(0)
    head = heads.PredictionHeads(num_roles=16)
    predicted, actions, page, element_mask = _synthetic_head_inputs(
        batch=1, steps=2, count=4,
    )
    out = head(predicted, actions, page, element_mask)

    pad_pred = torch.cat(
        (predicted, torch.full((1, 2, 3, encoding.MODEL_DIM), 42.0)),
        dim=2,
    )
    pad_mask = torch.cat(
        (element_mask, torch.zeros((1, 2, 3), dtype=torch.bool)),
        dim=2,
    )
    pad_out = head(pad_pred, actions, page, pad_mask)

    for key in ('changed_logits', 'removed_logits', 'next_state_logits',
                'value_logits', 'box_delta'):
        torch.testing.assert_close(pad_out[key][:, :, :4], out[key])
    for key in ('outcome_logits', 'new_role_counts', 'new_count'):
        torch.testing.assert_close(pad_out[key], out[key])

    all_masked = torch.zeros_like(element_mask)
    out_masked = head(predicted, actions, page, all_masked)
    for key in ('outcome_logits', 'new_role_counts', 'new_count'):
        assert torch.isfinite(out_masked[key]).all()


def test_heads_permutation_equivariance():
    torch.manual_seed(0)
    head = heads.PredictionHeads(num_roles=16)
    predicted, actions, page, element_mask = _synthetic_head_inputs(
        batch=1, steps=1, count=5,
    )
    out = head(predicted, actions, page, element_mask)

    perm = torch.tensor([3, 0, 4, 1, 2])
    perm_out = head(
        predicted[:, :, perm], actions, page, element_mask[:, :, perm],
    )
    for key in ('changed_logits', 'removed_logits', 'next_state_logits',
                'value_logits', 'box_delta'):
        torch.testing.assert_close(perm_out[key], out[key][:, :, perm])
    for key in ('outcome_logits', 'new_role_counts', 'new_count'):
        torch.testing.assert_close(perm_out[key], out[key])
