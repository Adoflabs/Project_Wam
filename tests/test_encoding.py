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
