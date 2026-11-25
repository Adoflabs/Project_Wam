"""Learned feature encoders for observable world state."""

import torch
from torch import nn
from torch.nn import functional

from browser_world.dataset import byte_encoding

_BYTE_EMBEDDING_DIM = 16
BYTE_FEATURE_DIM = 64
_BYTE_KERNEL_SIZE = 3


class ByteEncoder(nn.Module):
    """Encodes variable-length byte sequences into feature vectors."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=byte_encoding.BYTE_VOCAB_SIZE,
            embedding_dim=_BYTE_EMBEDDING_DIM,
            padding_idx=byte_encoding.BYTE_PAD_ID,
        )
        self.convolution = nn.Conv1d(
            in_channels=_BYTE_EMBEDDING_DIM,
            out_channels=BYTE_FEATURE_DIM,
            kernel_size=_BYTE_KERNEL_SIZE,
            padding=_BYTE_KERNEL_SIZE // 2,
        )

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Encodes byte IDs while excluding padded positions.

        Args:
          byte_ids: Integer tensor with shape [..., byte_length]. The last
            dimension must be nonempty. Zero represents padding.

        Returns:
          Feature tensor with shape [..., BYTE_FEATURE_DIM]. Entirely padded
          sequences produce zero vectors.

        Raises:
          ValueError: If the input has no byte dimension or the byte dimension
            is empty.
          TypeError: If the input dtype is not torch.long.
        """
        if byte_ids.ndim == 0 or byte_ids.shape[-1] == 0:
            raise ValueError('Expected a nonempty byte dimension.')
        if byte_ids.dtype != torch.long:
            raise TypeError('Byte IDs must use torch.long.')

        leading_shape = byte_ids.shape[:-1]
        byte_length = byte_ids.shape[-1]
        flat_ids = byte_ids.reshape(-1, byte_length)
        byte_mask = flat_ids != byte_encoding.BYTE_PAD_ID

        embeddings = self.embedding(flat_ids)
        embeddings = embeddings.masked_fill(
            ~byte_mask.unsqueeze(-1),
            0.0,
        )
        features = self.convolution(embeddings.transpose(1, 2))
        features = functional.gelu(features)
        features = features.masked_fill(
            ~byte_mask.unsqueeze(1),
            0.0,
        )
        valid_counts = byte_mask.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1)
        pooled = features.sum(dim=-1) / valid_counts
        return pooled.reshape(*leading_shape, BYTE_FEATURE_DIM)
