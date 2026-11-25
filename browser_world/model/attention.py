"""Relation-aware interactions between observable entities."""

import math

import torch
from torch import nn
from torch.nn import functional

from browser_world.model import encoding

_NUM_HEADS = 4
_HEAD_DIM = encoding.MODEL_DIM // _NUM_HEADS
_FEEDFORWARD_DIM = 256
_RELATION_DIM = 9


def relation_features(
    parent: torch.Tensor,
    box: torch.Tensor,
    box_mask: torch.Tensor,
    element_mask: torch.Tensor,
) -> torch.Tensor:
    """Constructs observed relationships between ordered entity pairs.

    Args:
      parent: Parent indices [B, K, N], with -1 for no parent.
      box: Normalized boxes [B, K, N, 4] in x, y, w, h order.
      box_mask: Valid geometry mask [B, K, N].
      element_mask: Real entity mask [B, K, N].

    Returns:
      Relations [B, K, N, N, 9].
      Axis -3 identifies the query entity i.
      Axis -2 identifies the key entity j.
      Features are self, key-is-parent, key-is-child, sibling,
      signed dx, signed dy, absolute dx, absolute dy, and
      geometry availability. Pairs involving padded entities are zero.
    """
    indices = torch.arange(
        parent.shape[-1],
        device=parent.device,
    )
    query_index = indices.unsqueeze(-1)
    key_index = indices.unsqueeze(0)
    query_parent = parent.unsqueeze(-1)
    key_parent = parent.unsqueeze(-2)

    same = query_index == key_index
    valid_pair = (
        element_mask.unsqueeze(-1) & element_mask.unsqueeze(-2)
    )
    siblings = (
        (query_parent == key_parent) & (query_parent >= 0) & ~same
    )
    geometry_valid = box_mask & element_mask
    centers = box[..., :2] + 0.5 * box[..., 2:]
    centers = centers.masked_fill(
        ~geometry_valid.unsqueeze(-1),
        0.0,
    )
    geometry_pair = (
        geometry_valid.unsqueeze(-1) & geometry_valid.unsqueeze(-2)
    )
    offset = centers.unsqueeze(-3) - centers.unsqueeze(-2)
    offset = offset.masked_fill(
        ~geometry_pair.unsqueeze(-1),
        0.0,
    )
    structural = torch.stack(
        (
            same.expand_as(valid_pair),
            query_parent == key_index,
            key_parent == query_index,
            siblings,
        ),
        dim=-1,
    ).to(box.dtype)
    relations = torch.cat(
        (
            structural,
            offset,
            offset.abs(),
            geometry_pair.unsqueeze(-1).to(box.dtype),
        ),
        dim=-1,
    )
    return relations.masked_fill(
        ~valid_pair.unsqueeze(-1),
        0.0,
    )


class RelationalAttentionBlock(nn.Module):
    """Updates entity representations using observed relationships."""

    def __init__(self) -> None:
        super().__init__()
        model_dim = encoding.MODEL_DIM
        self.attention_norm = nn.LayerNorm(model_dim)
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.relation_bias = nn.Linear(
            _RELATION_DIM,
            _NUM_HEADS,
            bias=False,
        )
        self.output_projection = nn.Linear(model_dim, model_dim)
        self.feedforward_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, _FEEDFORWARD_DIM),
            nn.GELU(),
            nn.Linear(_FEEDFORWARD_DIM, model_dim),
        )

    def forward(
        self,
        elements: torch.Tensor,
        relations: torch.Tensor,
        element_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Contextualizes entities without including padded elements.

        Args:
          elements: Entity features [B, K, N, MODEL_DIM].
          relations: Observed pair features [B, K, N, N, 9].
          element_mask: Real entity mask [B, K, N].

        Returns:
          Contextual entity features [B, K, N, MODEL_DIM]. Padded entities
          produce exact zero vectors. The input contract requires N >= 1 and
          valid observation-relative parent references.
        """
        elements = elements.masked_fill(
            ~element_mask.unsqueeze(-1),
            0.0,
        )
        batch_size, steps, count, _ = elements.shape
        qkv = self.qkv(self.attention_norm(elements))
        qkv = qkv.reshape(
            batch_size,
            steps,
            count,
            3,
            _NUM_HEADS,
            _HEAD_DIM,
        )
        query, key, value = qkv.permute(
            3, 0, 1, 4, 2, 5
        ).unbind(0)

        scores = query @ key.transpose(-2, -1) / math.sqrt(_HEAD_DIM)
        bias = self.relation_bias(relations).permute(0, 1, 4, 2, 3)
        scores = scores + bias

        key_mask = element_mask[:, :, None, None, :]
        scores = scores.masked_fill(
            ~key_mask,
            torch.finfo(scores.dtype).min,
        )
        weights = functional.softmax(scores, dim=-1).masked_fill(
            ~key_mask,
            0.0,
        )
        attended = (
            (weights @ value)
            .transpose(2, 3)
            .reshape(batch_size, steps, count, encoding.MODEL_DIM)
        )
        elements = elements + self.output_projection(attended)
        elements = elements.masked_fill(
            ~element_mask.unsqueeze(-1),
            0.0,
        )
        elements = elements + self.feedforward(
            self.feedforward_norm(elements)
        )
        return elements.masked_fill(
            ~element_mask.unsqueeze(-1),
            0.0,
        )
