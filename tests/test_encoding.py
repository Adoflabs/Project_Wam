"""Unit tests for the ByteEncoder neural component."""

# pylint: disable=missing-function-docstring

import pytest
import torch

from browser_world.dataset import byte_encoding
from browser_world.model import encoding


def test_parameter_count_and_shapes():
    encoder = encoding.ByteEncoder()
    assert encoder.embedding.weight.shape == (257, 16)
    assert encoder.convolution.weight.shape == (64, 16, 3)
    assert encoder.convolution.bias.shape == (64,)
    total_params = sum(
        parameter.numel() for parameter in encoder.parameters()
    )
    assert total_params == 7248


@pytest.mark.parametrize(
    ('input_shape', 'expected_shape'),
    [
        ((6,), (64,)),
        ((2, 6), (2, 64)),
        ((2, 3, 6), (2, 3, 64)),
        ((2, 3, 4, 6), (2, 3, 4, 64)),
    ],
)
def test_shape_preservation(input_shape, expected_shape):
    encoder = encoding.ByteEncoder()
    byte_ids = torch.randint(1, 257, input_shape, dtype=torch.long)
    features = encoder(byte_ids)
    assert features.shape == expected_shape
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()


def test_right_padding_invariance():
    encoder = encoding.ByteEncoder()
    doctor_ids = byte_encoding.encode_bytes('Doctor')
    exact = torch.tensor(doctor_ids, dtype=torch.long)
    padded = torch.tensor(
        doctor_ids + [byte_encoding.BYTE_PAD_ID] * 5,
        dtype=torch.long,
    )
    exact_features = encoder(exact)
    padded_features = encoder(padded)
    torch.testing.assert_close(
        exact_features,
        padded_features,
        rtol=1e-5,
        atol=1e-6,
    )


def test_empty_and_padded_sequences():
    encoder = encoding.ByteEncoder()
    all_padded = torch.zeros((2, 8), dtype=torch.long)
    empty_features = encoder(all_padded)
    assert empty_features.shape == (2, 64)
    assert torch.isfinite(empty_features).all()
    assert torch.equal(empty_features, torch.zeros((2, 64)))

    doctor_ids = byte_encoding.encode_bytes('Doctor')
    mixed = torch.tensor(
        [doctor_ids, [byte_encoding.BYTE_PAD_ID] * len(doctor_ids)],
        dtype=torch.long,
    )
    mixed_features = encoder(mixed)
    assert mixed_features.shape == (2, 64)
    assert torch.isfinite(mixed_features).all()
    assert torch.equal(mixed_features[1], torch.zeros((64,)))
    assert not torch.equal(mixed_features[0], torch.zeros((64,)))


@pytest.mark.parametrize('text', ['₹1250', 'नमस्ते दुनिया', '🚀'])
def test_unicode_input(text):
    encoder = encoding.ByteEncoder()
    byte_ids = torch.tensor(
        byte_encoding.encode_bytes(text),
        dtype=torch.long,
    )
    features = encoder(byte_ids)
    assert features.shape == (64,)
    assert features.dtype == torch.float32
    assert torch.isfinite(features).all()


def test_backward_computation():
    torch.manual_seed(0)
    encoder = encoding.ByteEncoder()
    byte_ids = torch.tensor(
        [byte_encoding.encode_bytes('Doctor') + [0, 0]],
        dtype=torch.long,
    )
    features = encoder(byte_ids)
    loss = features.square().mean()
    loss.backward()

    emb_grad = encoder.embedding.weight.grad
    conv_w_grad = encoder.convolution.weight.grad
    conv_b_grad = encoder.convolution.bias.grad
    assert emb_grad is not None
    assert conv_w_grad is not None
    assert conv_b_grad is not None
    assert torch.isfinite(emb_grad).all()
    assert torch.isfinite(conv_w_grad).all()
    assert torch.isfinite(conv_b_grad).all()
    assert torch.equal(
        emb_grad[byte_encoding.BYTE_PAD_ID],
        torch.zeros(16),
    )
    assert emb_grad[1:].abs().sum() > 0
    assert conv_w_grad.abs().sum() > 0


def test_invalid_input_contracts():
    encoder = encoding.ByteEncoder()
    with pytest.raises(ValueError, match='nonempty'):
        encoder(torch.tensor(5, dtype=torch.long))
    with pytest.raises(ValueError, match='nonempty'):
        encoder(torch.empty((2, 0), dtype=torch.long))
    with pytest.raises(TypeError, match='torch.long'):
        encoder(torch.ones((2, 4), dtype=torch.float32))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA device is not available.',
)
def test_cuda_compatibility():
    torch.manual_seed(0)
    cpu_encoder = encoding.ByteEncoder()
    cuda_encoder = encoding.ByteEncoder().to('cuda')
    cuda_encoder.load_state_dict(cpu_encoder.state_dict())

    cpu_ids = torch.tensor(
        [byte_encoding.encode_bytes('Doctor') + [0, 0]],
        dtype=torch.long,
    )
    cuda_ids = cpu_ids.to('cuda')

    cpu_features = cpu_encoder(cpu_ids)
    cuda_features = cuda_encoder(cuda_ids)
    assert cuda_features.device.type == 'cuda'
    assert torch.isfinite(cuda_features).all()
    torch.testing.assert_close(
        cuda_features.cpu(),
        cpu_features,
        rtol=1e-3,
        atol=1e-3,
    )

    loss = cuda_features.square().mean()
    loss.backward()
    assert cuda_encoder.embedding.weight.grad.device.type == 'cuda'
    assert cuda_encoder.convolution.weight.grad.device.type == 'cuda'
    assert cuda_encoder.convolution.bias.grad.device.type == 'cuda'
    assert torch.isfinite(cuda_encoder.embedding.weight.grad).all()
    assert torch.isfinite(cuda_encoder.convolution.weight.grad).all()
    assert torch.isfinite(cuda_encoder.convolution.bias.grad).all()


def _synthetic_element_inputs(batch_size=2, steps=1, num_elements=4):
    torch.manual_seed(0)
    role = torch.randint(1, 16, (batch_size, steps, num_elements))
    name_features = torch.randn(
        batch_size, steps, num_elements, encoding.BYTE_FEATURE_DIM,
    )
    value_features = torch.randn(
        batch_size, steps, num_elements, encoding.BYTE_FEATURE_DIM,
    )
    has_value = torch.randint(
        0, 2, (batch_size, steps, num_elements), dtype=torch.bool,
    )
    states = torch.randint(
        0, 2, (batch_size, steps, num_elements, 8), dtype=torch.float32,
    )
    flags = torch.randint(
        0, 2, (batch_size, steps, num_elements, 4), dtype=torch.float32,
    )
    box = torch.rand(batch_size, steps, num_elements, 4)
    box_mask = torch.ones(batch_size, steps, num_elements, dtype=torch.bool)
    element_mask = torch.ones(batch_size, steps, num_elements, dtype=torch.bool)
    return {
        'role': role,
        'name_features': name_features,
        'value_features': value_features,
        'has_value': has_value,
        'states': states,
        'flags': flags,
        'box': box,
        'box_mask': box_mask,
        'element_mask': element_mask,
    }


def test_element_encoder_shape_params_and_masking():
    torch.manual_seed(0)
    encoder = encoding.ElementEncoder(num_roles=16)
    total_params = sum(p.numel() for p in encoder.parameters())
    assert total_params == 21376

    inputs = _synthetic_element_inputs(batch_size=2, steps=1, num_elements=4)
    inputs['element_mask'][1, 0, 2:] = False
    inputs['box_mask'][0, 0, 1] = False

    out = encoder(**inputs)
    assert out.shape == (2, 1, 4, encoding.MODEL_DIM)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.equal(out[1, 0, 2:], torch.zeros((2, encoding.MODEL_DIM)))

    altered = dict(inputs)
    altered['box'] = inputs['box'].clone()
    altered['box'][0, 0, 1] = torch.tensor([99.0, -50.0, 12.0, 4.0])
    out_altered = encoder(**altered)
    torch.testing.assert_close(out, out_altered)


def test_element_encoder_permutation_equivariance():
    torch.manual_seed(0)
    encoder = encoding.ElementEncoder(num_roles=16)
    inputs = _synthetic_element_inputs(batch_size=1, steps=1, num_elements=4)
    out = encoder(**inputs)

    perm = torch.tensor([2, 0, 3, 1])
    permuted_inputs = {
        key: val[:, :, perm] for key, val in inputs.items()
    }
    permuted_out = encoder(**permuted_inputs)
    torch.testing.assert_close(permuted_out, out[:, :, perm])


def test_element_encoder_gradients_to_shared_byte_encoder():
    torch.manual_seed(0)
    byte_enc = encoding.ByteEncoder()
    elem_enc = encoding.ElementEncoder(num_roles=16)

    name_bytes = torch.tensor([[[
        byte_encoding.encode_bytes('Doctor') + [0, 0],
        byte_encoding.encode_bytes('Email') + [0, 0, 0],
    ]]], dtype=torch.long)
    # Pad value_bytes to uniform length 9.
    value_bytes = torch.cat((
        torch.zeros((1, 1, 1, 9), dtype=torch.long),
        torch.tensor([[[
            byte_encoding.encode_bytes('sam@x.com'),
        ]]], dtype=torch.long),
    ), dim=2)

    name_features = byte_enc(name_bytes)
    value_features = byte_enc(value_bytes)
    inputs = _synthetic_element_inputs(batch_size=1, steps=1, num_elements=2)
    inputs['name_features'] = name_features
    inputs['value_features'] = value_features

    out = elem_enc(**inputs)
    weights = torch.randn_like(out)
    loss = (out * weights).sum()
    loss.backward()

    assert elem_enc.role_embedding.weight.grad[1:].abs().sum() > 0
    assert elem_enc.projection.weight.grad.abs().sum() > 0
    assert byte_enc.embedding.weight.grad[1:].abs().sum() > 0
    assert byte_enc.convolution.weight.grad.abs().sum() > 0


def test_action_encoder_params_kinds_target_and_permutation():
    torch.manual_seed(0)
    byte_enc = encoding.ByteEncoder()
    elem_enc = encoding.ElementEncoder(num_roles=16)
    action_enc = encoding.ActionEncoder()

    action_params = sum(p.numel() for p in action_enc.parameters())
    assert action_params == 27984
    total_all = (
        sum(p.numel() for p in byte_enc.parameters())
        + sum(p.numel() for p in elem_enc.parameters())
        + action_params
    )
    assert total_all == 56608

    elements = torch.randn(2, 1, 4, encoding.MODEL_DIM)
    action_kind = torch.tensor([[0], [2]], dtype=torch.long)
    action_target = torch.tensor([[1], [3]], dtype=torch.long)
    text_features = torch.randn(2, 1, encoding.BYTE_FEATURE_DIM)
    action_checked = torch.tensor([[0], [2]], dtype=torch.long)
    action_scroll = torch.tensor([[0.0], [0.5]], dtype=torch.float32)
    action_pointer = torch.tensor([[[0.2, 0.3]], [[0.4, 0.5]]])

    out = action_enc(
        elements, action_kind, action_target, text_features,
        action_checked, action_scroll, action_pointer,
    )
    assert out.shape == (2, 1, encoding.MODEL_DIM)
    assert torch.isfinite(out).all()

    # Different action kinds produce different representations.
    out_other_kind = action_enc(
        elements, torch.tensor([[1], [2]], dtype=torch.long),
        action_target, text_features, action_checked,
        action_scroll, action_pointer,
    )
    assert not torch.equal(out[0], out_other_kind[0])

    # Different target entity features change the action representation.
    altered_elements = elements.clone()
    altered_elements[0, 0, 1] += 5.0
    out_altered_target = action_enc(
        altered_elements, action_kind, action_target, text_features,
        action_checked, action_scroll, action_pointer,
    )
    assert not torch.equal(out[0], out_altered_target[0])

    # Target -1 must ignore entity 0 and last entity completely.
    no_target = torch.tensor([[-1], [-1]], dtype=torch.long)
    out_no_target_a = action_enc(
        elements, action_kind, no_target, text_features,
        torch.tensor([[1], [0]], dtype=torch.long),
        action_scroll, action_pointer,
    )
    elements_changed = elements.clone()
    elements_changed[:, :, 0] += 10.0
    elements_changed[:, :, -1] -= 10.0
    out_no_target_b = action_enc(
        elements_changed, action_kind, no_target, text_features,
        torch.tensor([[1], [0]], dtype=torch.long),
        action_scroll, action_pointer,
    )
    torch.testing.assert_close(out_no_target_a, out_no_target_b)

    # Permuting entities and remapping action_target leaves output invariant.
    perm = torch.tensor([2, 0, 3, 1])
    inv_perm = torch.argsort(perm)
    remapped_target = inv_perm[action_target]
    out_permuted = action_enc(
        elements[:, :, perm], action_kind, remapped_target,
        text_features, action_checked, action_scroll, action_pointer,
    )
    torch.testing.assert_close(out, out_permuted)


def test_action_encoder_gradients_and_cuda():
    torch.manual_seed(0)
    cpu_action_enc = encoding.ActionEncoder()
    elements = torch.randn(1, 1, 3, encoding.MODEL_DIM, requires_grad=True)
    text_features = torch.randn(
        1, 1, encoding.BYTE_FEATURE_DIM, requires_grad=True,
    )
    action_kind = torch.tensor([[1]], dtype=torch.long)
    action_target = torch.tensor([[2]], dtype=torch.long)
    action_checked = torch.tensor([[1]], dtype=torch.long)
    action_scroll = torch.tensor([[0.0]], dtype=torch.float32)
    action_pointer = torch.tensor([[[0.5, 0.5]]])

    cpu_out = cpu_action_enc(
        elements, action_kind, action_target, text_features,
        action_checked, action_scroll, action_pointer,
    )
    weights = torch.randn_like(cpu_out)
    loss = (cpu_out * weights).sum()
    loss.backward()

    assert cpu_action_enc.kind_embedding.weight.grad.abs().sum() > 0
    assert cpu_action_enc.projection.weight.grad.abs().sum() > 0
    assert elements.grad[0, 0, 2].abs().sum() > 0
    assert text_features.grad.abs().sum() > 0

    if torch.cuda.is_available():
        cuda_action_enc = encoding.ActionEncoder().to('cuda')
        cuda_action_enc.load_state_dict(cpu_action_enc.state_dict())
        cuda_out = cuda_action_enc(
            elements.detach().to('cuda'),
            action_kind.to('cuda'),
            action_target.to('cuda'),
            text_features.detach().to('cuda'),
            action_checked.to('cuda'),
            action_scroll.to('cuda'),
            action_pointer.to('cuda'),
        )
        assert cuda_out.device.type == 'cuda'
        assert torch.isfinite(cuda_out).all()
        torch.testing.assert_close(
            cuda_out.cpu(),
            cpu_out,
            rtol=1e-3,
            atol=1e-3,
        )
        (cuda_out * weights.to('cuda')).sum().backward()
        assert cuda_action_enc.projection.weight.grad.device.type == 'cuda'
