"""Tests for frozen mmBERT-small SemanticEncoder perception."""

from __future__ import annotations

import pytest
import torch

from browser_world.wam2 import semantic

TEST_TEXTS = [
    'Destination Location',
    'City',
    'Bengaluru',
    'Select Bengaluru as the city and click Search.',
    'Bengaluru select karo aur Search par click karo.',
    'शहर में बेंगलुरु चुनें।',
]


@pytest.fixture(scope='module')
def cuda_encoder() -> semantic.SemanticEncoder:
    """Loads the frozen SemanticEncoder once on CUDA for module tests."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required for SemanticEncoder FP16 GPU test.')
    return semantic.SemanticEncoder(torch.device('cuda'))


def test_semantic_encoder_frozen_fp16_on_cuda(
    cuda_encoder: semantic.SemanticEncoder,
) -> None:
    assert not cuda_encoder.model.training
    params = list(cuda_encoder.model.parameters())
    assert params
    assert all(not p.requires_grad for p in params)
    assert all(p.dtype == torch.float16 for p in params)
    assert all(p.device.type == 'cuda' for p in params)


def test_semantic_encoder_real_multilingual_texts(
    cuda_encoder: semantic.SemanticEncoder,
) -> None:
    hidden, mask, offsets = cuda_encoder.encode(TEST_TEXTS, max_tokens=128)
    pooled = semantic.masked_mean(hidden, mask)

    assert hidden.ndim == 3
    assert hidden.shape[0] == len(TEST_TEXTS)
    assert hidden.shape[2] == semantic.TEXT_DIM == 384
    assert hidden.dtype == torch.float32
    assert torch.isfinite(hidden).all()

    assert mask.shape == hidden.shape[:2]
    assert mask.dtype == torch.bool
    assert offsets.shape == (*hidden.shape[:2], 2)
    assert offsets.device.type == 'cpu'

    for row_idx, text in enumerate(TEST_TEXTS):
        valid_positions = torch.nonzero(mask[row_idx].cpu(), as_tuple=False)
        assert valid_positions.numel() > 0
        for pos in valid_positions.flatten().tolist():
            start, end = offsets[row_idx, pos].tolist()
            assert 0 <= start < end <= len(text)
            assert text[start:end]

    assert list(pooled.shape) == [6, 384]
    assert pooled.dtype == torch.float32
    assert torch.isfinite(pooled).all()


def test_semantic_encoder_empty_content_and_budget_rejection(
    cuda_encoder: semantic.SemanticEncoder,
) -> None:
    hidden_empty, mask_empty, _ = cuda_encoder.encode([''], max_tokens=16)
    pooled_empty = semantic.masked_mean(hidden_empty, mask_empty)
    assert list(pooled_empty.shape) == [1, 384]
    assert int(mask_empty.sum().item()) == 0
    assert torch.equal(pooled_empty, torch.zeros_like(pooled_empty))

    zero_mask = torch.zeros_like(mask_empty)
    assert torch.equal(
        semantic.masked_mean(hidden_empty, zero_mask),
        torch.zeros_like(pooled_empty),
    )

    with pytest.raises(ValueError, match='token budget'):
        cuda_encoder.encode(
            ['Select Bengaluru as the city and click Search.'],
            max_tokens=4,
        )
