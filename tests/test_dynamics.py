"""Unit tests for ActionConditionedDynamics."""

# pylint: disable=missing-function-docstring

import pytest
import torch

from browser_world.model import attention
from browser_world.model import dynamics
from browser_world.model import encoding


def _synthetic_dynamics_inputs(batch=2, steps=3, count=5):
    torch.manual_seed(0)
    elements = torch.randn(batch, steps, count, encoding.MODEL_DIM)
    actions = torch.randn(batch, steps, encoding.MODEL_DIM)
    page = torch.rand(batch, steps, 4)
    action_target = torch.tensor([[1, 2, -1]] * batch, dtype=torch.long)
    action_target = action_target[:, :steps].clone()
    base_parent = [-1, 0, 0, 1, -1][:count]
    parent = torch.tensor([[base_parent]] * batch, dtype=torch.long)
    parent = parent.expand(batch, steps, count).clone()
    box = torch.rand(batch, steps, count, 4)
    box_mask = torch.ones(batch, steps, count, dtype=torch.bool)
    element_mask = torch.ones(batch, steps, count, dtype=torch.bool)
    relations = attention.relation_features(
        parent, box, box_mask, element_mask,
    )
    return (
        elements, actions, page, action_target,
        parent, box, box_mask, relations, element_mask,
    )


def test_output_contract_and_parameter_count():
    torch.manual_seed(0)
    model = dynamics.ActionConditionedDynamics()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 315336

    (
        elements, actions, page, action_target,
        _, _, _, relations, element_mask,
    ) = _synthetic_dynamics_inputs(batch=2, steps=3, count=5)
    out = model(
        elements, actions, page, action_target, relations, element_mask,
    )
    assert out.shape == (2, 3, 5, encoding.MODEL_DIM)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_independent_parameter_ownership():
    model = dynamics.ActionConditionedDynamics()
    block_a, block_b = model.blocks
    assert block_a is not block_b
    ptrs_a = {p.data_ptr() for p in block_a.parameters()}
    ptrs_b = {p.data_ptr() for p in block_b.parameters()}
    assert ptrs_a.isdisjoint(ptrs_b)


def test_action_target_and_page_sensitivities():
    torch.manual_seed(0)
    model = dynamics.ActionConditionedDynamics()
    (
        elements, actions, page, action_target,
        _, _, _, relations, element_mask,
    ) = _synthetic_dynamics_inputs(batch=1, steps=1, count=5)
    out_base = model(
        elements, actions, page, action_target, relations, element_mask,
    )

    # C. Action sensitivity.
    out_action = model(
        elements, actions + 1.5, page, action_target, relations, element_mask,
    )
    assert not torch.equal(out_base, out_action)

    # D. Target sensitivity (1 vs 3 vs -1).
    out_target_3 = model(
        elements, actions, page, torch.tensor([[3]]), relations, element_mask,
    )
    out_target_none = model(
        elements, actions, page, torch.tensor([[-1]]), relations, element_mask,
    )
    assert not torch.equal(out_base, out_target_3)
    assert not torch.equal(out_base, out_target_none)

    # E. Page-context sensitivity.
    alt_page = page.clone()
    alt_page[0, 0, 1] += 0.5
    out_page = model(
        elements, actions, alt_page, action_target, relations, element_mask,
    )
    assert not torch.equal(out_base, out_page)


def test_residual_identity():
    torch.manual_seed(0)
    model = dynamics.ActionConditionedDynamics()
    (
        elements, actions, page, action_target,
        _, _, _, relations, element_mask,
    ) = _synthetic_dynamics_inputs(batch=1, steps=2, count=5)
    element_mask[0, 1, 3:] = False

    with torch.no_grad():
        model.delta_projection.weight.zero_()
        model.delta_projection.bias.zero_()
        out = model(
            elements, actions, page, action_target, relations, element_mask,
        )

    torch.testing.assert_close(out[0, 0], elements[0, 0])
    torch.testing.assert_close(out[0, 1, :3], elements[0, 1, :3])
    assert torch.equal(out[0, 1, 3:], torch.zeros((2, encoding.MODEL_DIM)))


def test_permutation_equivariance():
    torch.manual_seed(0)
    model = dynamics.ActionConditionedDynamics()
    (
        elements, actions, page, action_target,
        parent, box, box_mask, relations, element_mask,
    ) = _synthetic_dynamics_inputs(batch=1, steps=2, count=5)
    action_target = torch.tensor([[1, -1]], dtype=torch.long)

    out = model(
        elements, actions, page, action_target, relations, element_mask,
    )

    perm = torch.tensor([3, 0, 4, 1, 2])
    inv_perm = torch.argsort(perm)
    perm_parent = parent[:, :, perm]
    remapped_parent = torch.where(
        perm_parent >= 0,
        inv_perm[perm_parent.clamp_min(0)],
        perm_parent,
    )
    remapped_target = torch.where(
        action_target >= 0,
        inv_perm[action_target.clamp_min(0)],
        action_target,
    )
    perm_relations = attention.relation_features(
        remapped_parent,
        box[:, :, perm],
        box_mask[:, :, perm],
        element_mask[:, :, perm],
    )
    perm_out = model(
        elements[:, :, perm],
        actions,
        page,
        remapped_target,
        perm_relations,
        element_mask[:, :, perm],
    )
    torch.testing.assert_close(perm_out, out[:, :, perm], rtol=1e-5, atol=1e-5)


def test_padding_invariance_and_empty_observation():
    torch.manual_seed(0)
    model = dynamics.ActionConditionedDynamics()
    (
        elements, actions, page, action_target,
        parent, box, box_mask, relations, element_mask,
    ) = _synthetic_dynamics_inputs(batch=1, steps=1, count=4)
    out = model(
        elements, actions, page, action_target, relations, element_mask,
    )

    pad_elements = torch.cat(
        (elements, torch.full((1, 1, 2, encoding.MODEL_DIM), 25.0)),
        dim=2,
    )
    pad_parent = torch.cat(
        (parent, torch.full((1, 1, 2), -1, dtype=torch.long)),
        dim=2,
    )
    pad_box = torch.cat((box, torch.ones((1, 1, 2, 4))), dim=2)
    pad_box_mask = torch.cat(
        (box_mask, torch.ones((1, 1, 2), dtype=torch.bool)),
        dim=2,
    )
    pad_mask = torch.cat(
        (element_mask, torch.zeros((1, 1, 2), dtype=torch.bool)),
        dim=2,
    )
    pad_relations = attention.relation_features(
        pad_parent, pad_box, pad_box_mask, pad_mask,
    )
    pad_out = model(
        pad_elements, actions, page, action_target, pad_relations, pad_mask,
    )
    torch.testing.assert_close(pad_out[:, :, :4], out, rtol=1e-5, atol=1e-5)
    assert torch.equal(
        pad_out[:, :, 4:],
        torch.zeros((1, 1, 2, encoding.MODEL_DIM)),
    )

    all_masked = torch.zeros_like(element_mask)
    rel_masked = attention.relation_features(
        parent, box, box_mask, all_masked,
    )
    out_masked = model(
        elements, actions, page, torch.tensor([[-1]]), rel_masked, all_masked,
    )
    assert torch.isfinite(out_masked).all()
    assert torch.equal(out_masked, torch.zeros_like(elements))


def test_gradient_propagation():
    torch.manual_seed(0)
    model = dynamics.ActionConditionedDynamics()
    (
        elements, actions, page, action_target,
        _, _, _, relations, element_mask,
    ) = _synthetic_dynamics_inputs(batch=1, steps=1, count=4)
    elements = elements.detach().requires_grad_(True)
    actions = actions.detach().requires_grad_(True)

    out = model(
        elements, actions, page, action_target, relations, element_mask,
    )
    loss = (out * torch.randn_like(out)).sum()
    loss.backward()

    assert elements.grad.abs().sum() > 0
    assert actions.grad.abs().sum() > 0
    assert model.conditioning.weight.grad.abs().sum() > 0
    assert model.blocks[0].qkv.weight.grad.abs().sum() > 0
    assert model.blocks[0].relation_bias.weight.grad.abs().sum() > 0
    assert model.blocks[1].qkv.weight.grad.abs().sum() > 0
    assert model.blocks[1].relation_bias.weight.grad.abs().sum() > 0
    assert model.delta_projection.weight.grad.abs().sum() > 0
    assert torch.isfinite(elements.grad).all()
    assert torch.isfinite(actions.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA device is not available.',
)
def test_cuda_compatibility():
    torch.manual_seed(0)
    cpu_model = dynamics.ActionConditionedDynamics()
    cuda_model = dynamics.ActionConditionedDynamics().to('cuda')
    cuda_model.load_state_dict(cpu_model.state_dict())

    (
        elements, actions, page, action_target,
        parent, box, box_mask, cpu_rel, element_mask,
    ) = _synthetic_dynamics_inputs(batch=1, steps=2, count=5)
    cuda_rel = attention.relation_features(
        parent.to('cuda'),
        box.to('cuda'),
        box_mask.to('cuda'),
        element_mask.to('cuda'),
    )
    cpu_out = cpu_model(
        elements, actions, page, action_target, cpu_rel, element_mask,
    )
    cuda_out = cuda_model(
        elements.to('cuda'),
        actions.to('cuda'),
        page.to('cuda'),
        action_target.to('cuda'),
        cuda_rel,
        element_mask.to('cuda'),
    )
    assert cuda_out.device.type == 'cuda'
    assert torch.isfinite(cuda_out).all()
    torch.testing.assert_close(
        cuda_out.cpu(), cpu_out, rtol=1e-3, atol=1e-3,
    )

    loss = (cuda_out * torch.randn_like(cuda_out)).sum()
    loss.backward()
    assert cuda_model.conditioning.weight.grad.device.type == 'cuda'
    assert cuda_model.delta_projection.weight.grad.device.type == 'cuda'
    assert torch.isfinite(cuda_model.delta_projection.weight.grad).all()
