"""Composes the one-step browser world prediction model."""

import torch
from torch import nn

from browser_world.model import attention
from browser_world.model import dynamics
from browser_world.model import encoding
from browser_world.model import heads

_OBSERVATION_DEPTH = 2

INPUT_KEYS = (
    'role',
    'name_bytes',
    'value_bytes',
    'has_value',
    'states',
    'flags',
    'box',
    'box_mask',
    'parent',
    'page',
    'action_kind',
    'action_target',
    'action_text_bytes',
    'action_checked',
    'action_scroll',
    'action_pointer',
    'element_mask',
)


class WorldModel(nn.Module):
    """Predicts observable consequences before an action executes."""

    def __init__(self, num_roles: int) -> None:
        super().__init__()
        self.byte_encoder = encoding.ByteEncoder()
        self.element_encoder = encoding.ElementEncoder(num_roles)
        self.observation_blocks = nn.ModuleList([
            attention.RelationalAttentionBlock()
            for _ in range(_OBSERVATION_DEPTH)
        ])
        self.action_encoder = encoding.ActionEncoder()
        self.dynamics = dynamics.ActionConditionedDynamics()
        self.heads = heads.PredictionHeads(num_roles)

    def forward(
        self,
        *,
        role: torch.Tensor,
        name_bytes: torch.Tensor,
        value_bytes: torch.Tensor,
        has_value: torch.Tensor,
        states: torch.Tensor,
        flags: torch.Tensor,
        box: torch.Tensor,
        box_mask: torch.Tensor,
        parent: torch.Tensor,
        page: torch.Tensor,
        action_kind: torch.Tensor,
        action_target: torch.Tensor,
        action_text_bytes: torch.Tensor,
        action_checked: torch.Tensor,
        action_scroll: torch.Tensor,
        action_pointer: torch.Tensor,
        element_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predicts consequences from pre-action information only.

        Args:
          role: Role IDs, torch.int64 [B, K, N].
          name_bytes: Element name UTF-8 byte IDs, torch.int64 [B, K, N, L_name].
          value_bytes: Element value UTF-8 byte IDs, torch.int64 [B, K, N, L_val].
          has_value: Value presence flags, torch.bool [B, K, N].
          states: Element boolean states, torch.float32 [B, K, N, 8].
          flags: Visibility/viewport/obstruction/pinned, torch.float32 [B, K, N, 4].
          box: Viewport-normalized boxes, torch.float32 [B, K, N, 4].
          box_mask: Geometry presence flags, torch.bool [B, K, N].
          parent: Parent element indices or -1, torch.int64 [B, K, N].
          page: Normalized page scroll and size ratios, torch.float32 [B, K, 4].
          action_kind: Action category IDs, torch.int64 [B, K].
          action_target: Target entity index or -1, torch.int64 [B, K].
          action_text_bytes: Typed text UTF-8 byte IDs, torch.int64 [B, K, L_act].
          action_checked: Check-state category IDs, torch.int64 [B, K].
          action_scroll: Normalized vertical scroll, torch.float32 [B, K].
          action_pointer: Pre-action pointer position, torch.float32 [B, K, 2].
          element_mask: Real entity mask, torch.bool [B, K, N].

        Returns:
          Entity-level and global consequence predictions.
          The model does not reconstruct a complete future
          observation or perform temporal rollouts.
        """
        name_features = self.byte_encoder(name_bytes)
        value_features = self.byte_encoder(value_bytes)
        text_features = self.byte_encoder(action_text_bytes)

        elements = self.element_encoder(
            role=role,
            name_features=name_features,
            value_features=value_features,
            has_value=has_value,
            states=states,
            flags=flags,
            box=box,
            box_mask=box_mask,
            element_mask=element_mask,
        )
        relations = attention.relation_features(
            parent=parent,
            box=box,
            box_mask=box_mask,
            element_mask=element_mask,
        )
        for block in self.observation_blocks:
            elements = block(elements, relations, element_mask)

        actions = self.action_encoder(
            elements=elements,
            action_kind=action_kind,
            action_target=action_target,
            text_features=text_features,
            action_checked=action_checked,
            action_scroll=action_scroll,
            action_pointer=action_pointer,
        )
        predicted = self.dynamics(
            elements=elements,
            actions=actions,
            page=page,
            action_target=action_target,
            relations=relations,
            element_mask=element_mask,
        )
        return self.heads(
            predicted=predicted,
            actions=actions,
            page=page,
            element_mask=element_mask,
        )
