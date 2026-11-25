"""Frozen contextual language encoder with exact original-text offsets."""

from collections.abc import Sequence

import torch
import transformers


MODEL_ID = 'jhu-clsp/mmBERT-small'
REVISION = 'ffaae3d46e9b171c3a001dfd4252b71d62a18c78'
TEXT_DIM = 384


class SemanticEncoder:
    """Read-only encoder; no generation or trainable optimizer parameters."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_ID, revision=REVISION, use_fast=True, trust_remote_code=False,
        )
        if not self.tokenizer.is_fast:
            raise RuntimeError('Exact copy requires a fast tokenizer.')
        self.model = transformers.AutoModel.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=False,
            use_safetensors=False, attn_implementation='sdpa',
        )
        if self.model.config.hidden_size != TEXT_DIM:
            raise RuntimeError('Unexpected semantic hidden size.')
        dtype = torch.float16 if device.type == 'cuda' else torch.float32
        self.model.to(device=device, dtype=dtype).requires_grad_(False).eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str], max_tokens: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Returns FP32 [B,L,384], valid-text mask [B,L], CPU offsets [B,L,2]."""
        if not texts:
            raise ValueError('Expected nonempty text batch.')
        batch = self.tokenizer(list(texts), padding=True, truncation=False,
                               return_offsets_mapping=True,
                               return_special_tokens_mask=True, return_tensors='pt')
        offsets = batch.pop('offset_mapping')
        special = batch.pop('special_tokens_mask').bool()
        if batch['input_ids'].shape[1] > max_tokens:
            raise ValueError('Text exceeds configured token budget; do not truncate.')
        mask = batch['attention_mask'].bool() & ~special & (offsets[..., 1] > offsets[..., 0])
        inputs = {name: value.to(self.device) for name, value in batch.items()}
        hidden = self.model(**inputs).last_hidden_state.float()
        return hidden, mask.to(self.device), offsets


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Returns [B,384]; empty content is represented as a zero vector."""
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
