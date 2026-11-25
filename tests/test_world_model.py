"""Unit tests for the composed WorldModel."""

# pylint: disable=missing-function-docstring

import inspect

import pytest
import torch

from browser_world.model import world_model


def _synthetic_model_inputs(batch=2, steps=1, count=4, byte_len=6):
    torch.manual_seed(0)
    parent = torch.tensor([[[-1, 0, 0, -1][:count]]] * batch, dtype=torch.long)
    parent = parent.expand(batch, steps, count).clone()
    return {
        'role': torch.randint(1, 16, (batch, steps, count)),
        'name_bytes': torch.randint(1, 257, (batch, steps, count, byte_len)),
        'value_bytes': torch.randint(1, 257, (batch, steps, count, byte_len)),
        'has_value': torch.randint(
            0, 2, (batch, steps, count), dtype=torch.bool,
        ),
        'states': torch.randint(
            0, 2, (batch, steps, count, 8), dtype=torch.float32,
        ),
        'flags': torch.randint(
            0, 2, (batch, steps, count, 4), dtype=torch.float32,
        ),
        'box': torch.rand(batch, steps, count, 4),
        'box_mask': torch.ones((batch, steps, count), dtype=torch.bool),
        'parent': parent,
        'page': torch.rand(batch, steps, 4),
        'action_kind': torch.tensor([[2], [4]][:batch], dtype=torch.long),
        'action_target': torch.tensor([[1], [-1]][:batch], dtype=torch.long),
        'action_text_bytes': torch.randint(1, 257, (batch, steps, 4)),
        'action_checked': torch.zeros((batch, steps), dtype=torch.long),
        'action_scroll': torch.zeros((batch, steps), dtype=torch.float32),
        'action_pointer': torch.rand(batch, steps, 2),
        'element_mask': torch.ones((batch, steps, count), dtype=torch.bool),
    }


def test_world_model_structure_params_and_whitelist():
    torch.manual_seed(0)
    model = world_model.WorldModel(num_roles=16)
    assert sum(p.numel() for p in model.parameters()) == 675802

    # Four independent attention blocks (2 observation + 2 dynamics).
    blocks = [*model.observation_blocks, *model.dynamics.blocks]
    assert len(blocks) == 4
    seen_ptrs = set()
    for block in blocks:
        ptrs = {p.data_ptr() for p in block.parameters()}
        assert seen_ptrs.isdisjoint(ptrs)
        seen_ptrs.update(ptrs)

    # Whitelist & no target leakage in signature or INPUT_KEYS.
    sig_params = set(inspect.signature(model.forward).parameters)
    assert sig_params == set(world_model.INPUT_KEYS)
    assert not any(k.startswith('t_') for k in world_model.INPUT_KEYS)

    inputs = _synthetic_model_inputs(batch=2, steps=1, count=4)
    out = model(**inputs)
    assert out['changed_logits'].shape == (2, 1, 4)
    assert out['removed_logits'].shape == (2, 1, 4)
    assert out['next_state_logits'].shape == (2, 1, 4, 8)
    assert out['value_logits'].shape == (2, 1, 4, 5)
    assert out['box_delta'].shape == (2, 1, 4, 4)
    assert out['outcome_logits'].shape == (2, 1, 8)
    assert out['new_role_counts'].shape == (2, 1, 15)
    assert out['new_count'].shape == (2, 1)


def test_world_model_permutation_and_padding_invariance():
    torch.manual_seed(0)
    model = world_model.WorldModel(num_roles=16)
    inputs = _synthetic_model_inputs(batch=1, steps=1, count=4)
    out = model(**inputs)

    perm = torch.tensor([2, 0, 3, 1])
    inv_perm = torch.argsort(perm)
    perm_inputs = dict(inputs)
    for k in ('role', 'name_bytes', 'value_bytes', 'has_value', 'states',
              'flags', 'box', 'box_mask', 'element_mask'):
        perm_inputs[k] = inputs[k][:, :, perm]
    perm_parent = inputs['parent'][:, :, perm]
    perm_inputs['parent'] = torch.where(
        perm_parent >= 0,
        inv_perm[perm_parent.clamp_min(0)],
        perm_parent,
    )
    perm_inputs['action_target'] = torch.where(
        inputs['action_target'] >= 0,
        inv_perm[inputs['action_target'].clamp_min(0)],
        inputs['action_target'],
    )
    perm_out = model(**perm_inputs)
    for k in ('changed_logits', 'removed_logits', 'next_state_logits',
              'value_logits', 'box_delta'):
        torch.testing.assert_close(
            perm_out[k], out[k][:, :, perm], rtol=1e-5, atol=1e-5,
        )
    for k in ('outcome_logits', 'new_role_counts', 'new_count'):
        torch.testing.assert_close(perm_out[k], out[k], rtol=1e-5, atol=1e-5)

    # Padding invariance.
    pad_inputs = dict(inputs)
    for k in ('role', 'name_bytes', 'value_bytes', 'has_value', 'states',
              'flags', 'box', 'box_mask'):
        pad_shape = list(inputs[k].shape)
        pad_shape[2] = 2
        pad_inputs[k] = torch.cat(
            (inputs[k], torch.zeros(pad_shape, dtype=inputs[k].dtype)),
            dim=2,
        )
    pad_inputs['parent'] = torch.cat(
        (inputs['parent'], torch.full((1, 1, 2), -1, dtype=torch.long)),
        dim=2,
    )
    pad_inputs['element_mask'] = torch.cat(
        (inputs['element_mask'], torch.zeros((1, 1, 2), dtype=torch.bool)),
        dim=2,
    )
    pad_out = model(**pad_inputs)
    for k in ('changed_logits', 'removed_logits', 'next_state_logits',
              'value_logits', 'box_delta'):
        torch.testing.assert_close(
            pad_out[k][:, :, :4], out[k], rtol=1e-5, atol=1e-5,
        )
    for k in ('outcome_logits', 'new_role_counts', 'new_count'):
        torch.testing.assert_close(pad_out[k], out[k], rtol=1e-5, atol=1e-5)


def test_world_model_gradients_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = world_model.WorldModel(num_roles=16)
    inputs = _synthetic_model_inputs(batch=2, steps=1, count=4)
    out = model(**inputs)
    loss = sum((v * torch.randn_like(v)).sum() for v in out.values())
    loss.backward()

    assert model.byte_encoder.embedding.weight.grad[1:].abs().sum() > 0
    assert model.element_encoder.projection.weight.grad.abs().sum() > 0
    assert model.observation_blocks[0].qkv.weight.grad.abs().sum() > 0
    assert model.action_encoder.projection.weight.grad.abs().sum() > 0
    assert model.dynamics.conditioning.weight.grad.abs().sum() > 0
    assert model.heads.changed.weight.grad.abs().sum() > 0
    assert model.heads.new_roles.weight.grad.abs().sum() > 0

    ckpt = tmp_path / 'model.pt'
    torch.save(model.state_dict(), ckpt)
    reloaded = world_model.WorldModel(num_roles=16)
    reloaded.load_state_dict(torch.load(ckpt, weights_only=True))
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        out_a = model(**inputs)
        out_b = reloaded(**inputs)
    for key in out_a:
        torch.testing.assert_close(out_a[key], out_b[key])


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA device is not available.',
)
def test_world_model_cuda_compatibility():
    torch.manual_seed(0)
    cpu_model = world_model.WorldModel(num_roles=16)
    cuda_model = world_model.WorldModel(num_roles=16).to('cuda')
    cuda_model.load_state_dict(cpu_model.state_dict())

    cpu_inputs = _synthetic_model_inputs(batch=2, steps=1, count=4)
    cuda_inputs = {k: v.to('cuda') for k, v in cpu_inputs.items()}
    cpu_out = cpu_model(**cpu_inputs)
    cuda_out = cuda_model(**cuda_inputs)

    for key in cpu_out:
        assert cuda_out[key].device.type == 'cuda'
        assert torch.isfinite(cuda_out[key]).all()
        torch.testing.assert_close(
            cuda_out[key].cpu(), cpu_out[key], rtol=1e-2, atol=1e-2,
        )
