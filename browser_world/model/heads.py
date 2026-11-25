"""Observable consequence predictions from future entity features."""

import torch
from torch import nn
from torch.nn import functional

from browser_world.dataset import schema
from browser_world.model import encoding

_PAGE_DIM = 4
_BOX_DIM = 4
BOX_DELTA_SCALE = 0.01


class PredictionHeads(nn.Module):
    """Predicts entity-level and page-level action consequences."""

    def __init__(self, num_roles: int) -> None:
        super().__init__()
        if num_roles < 2:
            raise ValueError('Expected padding and non-padding roles.')
        model_dim = encoding.MODEL_DIM
        self.changed = nn.Linear(model_dim, 1)
        self.removed = nn.Linear(model_dim, 1)
        self.next_states = nn.Linear(model_dim, len(schema.STATE_NAMES))
        self.value_source = nn.Linear(model_dim, len(schema.VALUE_SOURCES))
        self.box_delta = nn.Linear(model_dim, _BOX_DIM)
        self.global_projection = nn.Linear(
            2 * model_dim + _PAGE_DIM,
            model_dim,
        )
        self.outcomes = nn.Linear(model_dim, len(schema.OUTCOMES))
        self.new_roles = nn.Linear(model_dim, num_roles - 1)

    def forward(
        self,
        predicted: torch.Tensor,
        actions: torch.Tensor,
        page: torch.Tensor,
        element_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decodes predicted entity features into consequences.

        Args:
          predicted: Future entity features [B, K, N, MODEL_DIM].
          actions: Proposed action features [B, K, MODEL_DIM].
          page: Current page features [B, K, 4].
          element_mask: Current real-entity mask [B, K, N].

        Returns:
          Entity-level predictions for current slots and global
          predictions for each observation-action pair.

          Binary and categorical outputs are logits.

          New-role count position zero represents role ID one.
          Padding role ID zero has no predicted count.
        """
        predicted = predicted.masked_fill(
            ~element_mask.unsqueeze(-1),
            0.0,
        )
        entity_count = element_mask.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1)
        pooled = predicted.sum(dim=-2) / entity_count
        global_features = functional.gelu(
            self.global_projection(
                torch.cat((pooled, actions, page), dim=-1)
            )
        )
        role_counts = functional.softplus(self.new_roles(global_features))
        return {
            'changed_logits': self.changed(predicted).squeeze(-1),
            'removed_logits': self.removed(predicted).squeeze(-1),
            'next_state_logits': self.next_states(predicted),
            'value_logits': self.value_source(predicted),
            'box_delta': BOX_DELTA_SCALE * self.box_delta(predicted),
            'outcome_logits': self.outcomes(global_features),
            'new_role_counts': role_counts,
            'new_count': role_counts.sum(dim=-1),
        }
