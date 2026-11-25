"""Action-conditioned prediction of future entity features."""

import torch
from torch import nn
from torch.nn import functional

from browser_world.model import attention
from browser_world.model import encoding

_PAGE_DIM = 4
_DYNAMICS_DEPTH = 2


class ActionConditionedDynamics(nn.Module):
    """Predicts residual latent changes caused by proposed actions."""

    def __init__(self) -> None:
        super().__init__()
        model_dim = encoding.MODEL_DIM
        self.conditioning = nn.Linear(
            2 * model_dim + _PAGE_DIM + 1,
            model_dim,
        )
        self.blocks = nn.ModuleList([
            attention.RelationalAttentionBlock()
            for _ in range(_DYNAMICS_DEPTH)
        ])
        self.delta_norm = nn.LayerNorm(model_dim)
        self.delta_projection = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        elements: torch.Tensor,
        actions: torch.Tensor,
        page: torch.Tensor,
        action_target: torch.Tensor,
        relations: torch.Tensor,
        element_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts future features for current entity slots.

        Args:
          elements: Contextual entity features [B, K, N, MODEL_DIM].
          actions: Proposed action features [B, K, MODEL_DIM].
          page: Current normalized page features [B, K, 4].
          action_target: Entity indices or -1, shape [B, K].
          relations: Current pairwise relations [B, K, N, N, 9].
          element_mask: Real entity mask [B, K, N].

        Returns:
          Predicted features [B, K, N, MODEL_DIM].
          Padded slots produce exact zero vectors. Removal and
          creation are not applied by this module.

        Inputs must follow the existing dataset contract. This
        computation does not access actual future observations.
        """
        count = elements.shape[-2]
        indices = torch.arange(count, device=elements.device)
        target_mask = (action_target.unsqueeze(-1) == indices) & element_mask
        action_context = actions.unsqueeze(-2).expand_as(elements)
        page_context = page.unsqueeze(-2).expand(
            *elements.shape[:-1],
            page.shape[-1],
        )
        combined = torch.cat(
            (
                elements,
                action_context,
                page_context,
                target_mask.unsqueeze(-1).to(elements.dtype),
            ),
            dim=-1,
        )
        hidden = functional.gelu(self.conditioning(combined))
        hidden = hidden.masked_fill(~element_mask.unsqueeze(-1), 0.0)
        for block in self.blocks:
            hidden = block(hidden, relations, element_mask)
        delta = self.delta_projection(self.delta_norm(hidden))
        return (elements + delta).masked_fill(
            ~element_mask.unsqueeze(-1),
            0.0,
        )
