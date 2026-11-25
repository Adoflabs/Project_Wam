"""Learned feature encoders for observable world state."""

import torch
from torch import nn
from torch.nn import functional

from browser_world.dataset import byte_encoding
from browser_world.dataset import schema

_BYTE_EMBEDDING_DIM = 16
BYTE_FEATURE_DIM = 64
_BYTE_KERNEL_SIZE = 3
MODEL_DIM = 128
_ROLE_EMBEDDING_DIM = 16
_ACTION_EMBEDDING_DIM = 16
_FLAGS_DIM = 4
_BOX_DIM = 4
_CHECK_STATE_COUNT = 3
_POINTER_DIM = 2


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


class ElementEncoder(nn.Module):
    """Projects observable entity properties into model features."""

    def __init__(self, num_roles: int) -> None:
        super().__init__()
        self.role_embedding = nn.Embedding(
            num_embeddings=num_roles,
            embedding_dim=_ROLE_EMBEDDING_DIM,
            padding_idx=schema.PAD,
        )
        input_dim = (
            _ROLE_EMBEDDING_DIM
            + 2 * BYTE_FEATURE_DIM
            + 1  # Value presence.
            + len(schema.STATE_NAMES)
            + _FLAGS_DIM
            + _BOX_DIM
            + 1  # Geometry presence.
        )
        self.projection = nn.Linear(input_dim, MODEL_DIM)
        self.normalization = nn.LayerNorm(MODEL_DIM)

    def forward(
        self,
        role: torch.Tensor,
        name_features: torch.Tensor,
        value_features: torch.Tensor,
        has_value: torch.Tensor,
        states: torch.Tensor,
        flags: torch.Tensor,
        box: torch.Tensor,
        box_mask: torch.Tensor,
        element_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes observable entities and zeroes padded positions.

        All inputs have leading dimensions [B, K, N].
        Name and value features have width BYTE_FEATURE_DIM.
        Returns [B, K, N, MODEL_DIM].
        """
        role_features = self.role_embedding(role)
        geometry = box.masked_fill(~box_mask.unsqueeze(-1), 0.0)
        combined = torch.cat(
            (
                role_features,
                name_features,
                value_features,
                has_value.unsqueeze(-1).to(role_features.dtype),
                states,
                flags,
                geometry,
                box_mask.unsqueeze(-1).to(role_features.dtype),
            ),
            dim=-1,
        )
        features = self.projection(combined)
        features = functional.gelu(features)
        features = self.normalization(features)
        return features.masked_fill(~element_mask.unsqueeze(-1), 0.0)


class ActionEncoder(nn.Module):
    """Encodes an intervention grounded in current entity features."""

    def __init__(self) -> None:
        super().__init__()
        self.kind_embedding = nn.Embedding(
            num_embeddings=len(schema.ACTION_KINDS),
            embedding_dim=_ACTION_EMBEDDING_DIM,
        )
        input_dim = (
            _ACTION_EMBEDDING_DIM
            + MODEL_DIM
            + BYTE_FEATURE_DIM
            + _CHECK_STATE_COUNT
            + 1  # Requested vertical scroll.
            + _POINTER_DIM
            + 1  # Target presence.
        )
        self.projection = nn.Linear(input_dim, MODEL_DIM)
        self.normalization = nn.LayerNorm(MODEL_DIM)

    def forward(
        self,
        elements: torch.Tensor,
        action_kind: torch.Tensor,
        action_target: torch.Tensor,
        text_features: torch.Tensor,
        action_checked: torch.Tensor,
        action_scroll: torch.Tensor,
        action_pointer: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes proposed actions.

        Args:
          elements: Current features [B, K, N, MODEL_DIM], N >= 1.
          action_kind: Action category IDs [B, K].
          action_target: Valid entity indices or -1, shape [B, K].
            Nonnegative indices must point to real entities.
          text_features: Encoded action text [B, K, BYTE_FEATURE_DIM].
          action_checked: Check-state category IDs [B, K].
          action_scroll: Requested vertical displacement [B, K].
          action_pointer: Pre-action pointer position [B, K, 2].

        Returns:
          Action representation [B, K, MODEL_DIM].
        """
        target_present = action_target >= 0
        gather_index = (
            action_target.clamp_min(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(*action_target.shape, 1, MODEL_DIM)
        )
        target_features = elements.gather(
            dim=-2,
            index=gather_index,
        ).squeeze(-2)
        target_features = target_features.masked_fill(
            ~target_present.unsqueeze(-1),
            0.0,
        )
        kind_features = self.kind_embedding(action_kind)
        checked_features = functional.one_hot(
            action_checked,
            num_classes=_CHECK_STATE_COUNT,
        ).to(kind_features.dtype)
        combined = torch.cat(
            (
                kind_features,
                target_features,
                text_features,
                checked_features,
                action_scroll.unsqueeze(-1),
                action_pointer,
                target_present.unsqueeze(-1).to(kind_features.dtype),
            ),
            dim=-1,
        )
        features = self.projection(combined)
        features = functional.gelu(features)
        return self.normalization(features)
