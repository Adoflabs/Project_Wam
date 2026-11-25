"""Unit tests for relation_features and RelationalAttentionBlock."""

# pylint: disable=missing-function-docstring

import pytest
import torch
from torch.nn import functional

from browser_world.model import attention


def test_exact_relationship_values():
    # Entities: 0=root, 1=child of 0, 2=child of 0, 3=root without box, 4=pad
    parent = torch.tensor([[[-1, 0, 0, -1, -1]]], dtype=torch.long)
    box = torch.tensor([[[
        [0.0, 0.0, 0.2, 0.2],   # center (0.1, 0.1)
        [0.4, 0.1, 0.2, 0.2],   # center (0.5, 0.2)
        [0.0, 0.6, 0.4, 0.2],   # center (0.2, 0.7)
        [0.9, 0.9, 0.1, 0.1],   # invalid geometry
        [0.5, 0.5, 0.1, 0.1],   # padded entity
    ]]], dtype=torch.float32)
    box_mask = torch.tensor([[[True, True, True, False, True]]])
    element_mask = torch.tensor([[[True, True, True, True, False]]])

    rel = attention.relation_features(parent, box, box_mask, element_mask)
    assert rel.shape == (1, 1, 5, 5, 9)

    # Self relations for valid entities 0..3.
    for i in range(4):
        assert rel[0, 0, i, i, 0] == 1.0
        assert rel[0, 0, i, i, 3] == 0.0  # Not its own sibling.

    # Entity 0 is parent of 1 and 2; direction reverses for (0, 1) vs (1, 0).
    assert rel[0, 0, 1, 0, :4].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert rel[0, 0, 0, 1, :4].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert rel[0, 0, 2, 0, :4].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert rel[0, 0, 0, 2, :4].tolist() == [0.0, 0.0, 1.0, 0.0]

    # Entities 1 and 2 are siblings; roots 0 and 3 are not siblings.
    assert rel[0, 0, 1, 2, :4].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert rel[0, 0, 2, 1, :4].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert rel[0, 0, 0, 3, :4].tolist() == [0.0, 0.0, 0.0, 0.0]

    # Signed offset for (0, 1) is center_1 - center_0 = (0.4, 0.1).
    torch.testing.assert_close(
        rel[0, 0, 0, 1, 4:],
        torch.tensor([0.4, 0.1, 0.4, 0.1, 1.0]),
    )
    # Signed offset for (1, 0) is center_0 - center_1 = (-0.4, -0.1).
    torch.testing.assert_close(
        rel[0, 0, 1, 0, 4:],
        torch.tensor([-0.4, -0.1, 0.4, 0.1, 1.0]),
    )
    assert (rel[..., 6:8] >= 0.0).all()

    # Missing geometry for entity 3 yields zero offsets and availability 0.
    assert rel[0, 0, 0, 3, 4:].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert rel[0, 0, 3, 3, 4:].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]

    # Padded entity 4 has zero relations in both query and key axes.
    assert torch.equal(rel[0, 0, 4, :], torch.zeros((5, 9)))
    assert torch.equal(rel[0, 0, :, 4], torch.zeros((5, 9)))


def _synthetic_batch(batch=2, steps=3, count=5):
    torch.manual_seed(0)
    elements = torch.randn(batch, steps, count, 128)
    base_parent = [-1, 0, 0, 1, -1][:count]
    parent = torch.tensor([[base_parent]] * batch, dtype=torch.long)
    parent = parent.expand(batch, steps, count).clone()
    box = torch.rand(batch, steps, count, 4)
    box_mask = torch.ones(batch, steps, count, dtype=torch.bool)
    element_mask = torch.ones(batch, steps, count, dtype=torch.bool)
    return elements, parent, box, box_mask, element_mask


def test_attention_shape_and_parameter_count():
    torch.manual_seed(0)
    block = attention.RelationalAttentionBlock()
    total_params = sum(p.numel() for p in block.parameters())
    assert total_params == 132516

    elements, parent, box, box_mask, element_mask = _synthetic_batch()
    relations = attention.relation_features(
        parent, box, box_mask, element_mask,
    )
    out = block(elements, relations, element_mask)
    assert out.shape == (2, 3, 5, 128)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_entity_permutation_equivariance():
    torch.manual_seed(0)
    block = attention.RelationalAttentionBlock()
    elements, parent, box, box_mask, element_mask = _synthetic_batch(
        batch=1, steps=1, count=5,
    )
    relations = attention.relation_features(
        parent, box, box_mask, element_mask,
    )
    out = block(elements, relations, element_mask)

    perm = torch.tensor([3, 0, 4, 1, 2])
    inv_perm = torch.argsort(perm)
    perm_parent = parent[:, :, perm]
    remapped_parent = torch.where(
        perm_parent >= 0,
        inv_perm[perm_parent.clamp_min(0)],
        perm_parent,
    )
    perm_relations = attention.relation_features(
        remapped_parent,
        box[:, :, perm],
        box_mask[:, :, perm],
        element_mask[:, :, perm],
    )
    perm_out = block(
        elements[:, :, perm],
        perm_relations,
        element_mask[:, :, perm],
    )
    torch.testing.assert_close(perm_out, out[:, :, perm], rtol=1e-5, atol=1e-5)


def test_padding_invariance_and_entirely_masked():
    torch.manual_seed(0)
    block = attention.RelationalAttentionBlock()
    elements, parent, box, box_mask, element_mask = _synthetic_batch(
        batch=1, steps=1, count=4,
    )
    rel = attention.relation_features(parent, box, box_mask, element_mask)
    out = block(elements, rel, element_mask)

    pad_elements = torch.cat(
        (elements, torch.full((1, 1, 3, 128), 50.0)),
        dim=2,
    )
    pad_parent = torch.cat(
        (parent, torch.full((1, 1, 3), -1, dtype=torch.long)),
        dim=2,
    )
    pad_box = torch.cat((box, torch.ones((1, 1, 3, 4))), dim=2)
    pad_box_mask = torch.cat(
        (box_mask, torch.ones((1, 1, 3), dtype=torch.bool)),
        dim=2,
    )
    pad_mask = torch.cat(
        (element_mask, torch.zeros((1, 1, 3), dtype=torch.bool)),
        dim=2,
    )
    pad_rel = attention.relation_features(
        pad_parent, pad_box, pad_box_mask, pad_mask,
    )
    pad_out = block(pad_elements, pad_rel, pad_mask)
    torch.testing.assert_close(pad_out[:, :, :4], out, rtol=1e-5, atol=1e-5)
    assert torch.equal(pad_out[:, :, 4:], torch.zeros((1, 1, 3, 128)))

    all_masked = torch.zeros_like(element_mask)
    rel_masked = attention.relation_features(
        parent, box, box_mask, all_masked,
    )
    out_masked = block(elements, rel_masked, all_masked)
    assert torch.isfinite(out_masked).all()
    assert torch.equal(out_masked, torch.zeros_like(elements))


def test_missing_geometry_and_relationship_sensitivity():
    torch.manual_seed(0)
    block = attention.RelationalAttentionBlock()
    elements, parent, box, box_mask, element_mask = _synthetic_batch(
        batch=1, steps=1, count=4,
    )
    box_mask[0, 0, 2] = False
    rel_a = attention.relation_features(parent, box, box_mask, element_mask)
    out_a = block(elements, rel_a, element_mask)

    box_changed = box.clone()
    box_changed[0, 0, 2] = torch.tensor([99.0, -40.0, 5.0, 2.0])
    rel_b = attention.relation_features(
        parent, box_changed, box_mask, element_mask,
    )
    torch.testing.assert_close(rel_a, rel_b)
    out_b = block(elements, rel_b, element_mask)
    torch.testing.assert_close(out_a, out_b)

    # Change parent-child relationship while holding entity features constant.
    alt_parent = parent.clone()
    alt_parent[0, 0, 3] = 2
    rel_c = attention.relation_features(
        alt_parent, box, box_mask, element_mask,
    )
    out_c = block(elements, rel_c, element_mask)
    assert not torch.equal(out_a, out_c)


def test_gradient_propagation():
    torch.manual_seed(0)
    block = attention.RelationalAttentionBlock()
    elements, parent, box, box_mask, element_mask = _synthetic_batch(
        batch=1, steps=1, count=4,
    )
    elements = elements.detach().requires_grad_(True)
    relations = attention.relation_features(
        parent, box, box_mask, element_mask,
    )
    out = block(elements, relations, element_mask)
    weights = torch.randn_like(out)
    loss = (out * weights).sum()
    loss.backward()

    assert block.qkv.weight.grad.abs().sum() > 0
    assert block.relation_bias.weight.grad.abs().sum() > 0
    assert block.output_projection.weight.grad.abs().sum() > 0
    assert block.feedforward[0].weight.grad.abs().sum() > 0
    assert block.feedforward[2].weight.grad.abs().sum() > 0
    assert elements.grad.abs().sum() > 0
    assert torch.isfinite(elements.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA device is not available.',
)
def test_cuda_compatibility():
    torch.manual_seed(0)
    cpu_block = attention.RelationalAttentionBlock()
    cuda_block = attention.RelationalAttentionBlock().to('cuda')
    cuda_block.load_state_dict(cpu_block.state_dict())

    elements, parent, box, box_mask, element_mask = _synthetic_batch(
        batch=1, steps=2, count=5,
    )
    cpu_rel = attention.relation_features(
        parent, box, box_mask, element_mask,
    )
    cuda_rel = attention.relation_features(
        parent.to('cuda'),
        box.to('cuda'),
        box_mask.to('cuda'),
        element_mask.to('cuda'),
    )
    cpu_out = cpu_block(elements, cpu_rel, element_mask)
    cuda_out = cuda_block(
        elements.to('cuda'), cuda_rel, element_mask.to('cuda'),
    )
    assert cuda_out.device.type == 'cuda'
    assert torch.isfinite(cuda_out).all()
    torch.testing.assert_close(
        cuda_out.cpu(), cpu_out, rtol=1e-3, atol=1e-3,
    )

    loss = (cuda_out * torch.ones_like(cuda_out)).sum()
    loss.backward()
    assert cuda_block.qkv.weight.grad.device.type == 'cuda'
    assert cuda_block.relation_bias.weight.grad.device.type == 'cuda'
    assert torch.isfinite(cuda_block.relation_bias.weight.grad).all()
