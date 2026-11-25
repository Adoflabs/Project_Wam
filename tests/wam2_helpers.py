"""Shared deterministic test doubles and observation constructors for WAM2."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from browser_world.interaction import observer
from browser_world.wam2 import semantic


class FakeSemanticEncoder:
    """Deterministic CPU encoder double for fast unit tests."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device('cpu')

    def encode(
        self, texts: Sequence[str], max_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del max_tokens
        batch = len(texts)
        hidden = torch.zeros(
            (batch, 1, semantic.TEXT_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.zeros((batch, 1), dtype=torch.bool, device=self.device)
        offsets = torch.zeros((batch, 1, 2), dtype=torch.long)
        for idx, text in enumerate(texts):
            if text:
                mask[idx, 0] = True
                offsets[idx, 0, 1] = len(text)
                encoded = text.encode('utf-8')[:semantic.TEXT_DIM]
                for ch_i, ch in enumerate(encoded):
                    hidden[idx, 0, ch_i] = (ch + 1) / 256.0
        return hidden, mask, offsets


def make_element(
    node_id: int,
    role: str,
    name: str,
    *,
    value: str | None = None,
    states: dict[str, bool] | None = None,
    box: observer.Box | None = observer.Box(10.0, 20.0, 100.0, 30.0),
    parent_id: int | None = None,
    match_key: str | None = None,
    tag: str = 'div',
    visible: bool = True,
    in_viewport: bool = True,
    obstructed: bool = False,
    pinned: bool = False,
) -> observer.Element:
    return observer.Element(
        node_id=node_id,
        role=role,
        name=name,
        value=value,
        states=dict(states or {}),
        parent_id=parent_id,
        match_key=match_key,
        tag=tag,
        box=box,
        visible=visible,
        in_viewport=in_viewport,
        hit_point=box.center if box is not None else None,
        hit_testable=True,
        obstructed=obstructed,
        clickable=role == 'button',
        pinned=pinned,
        paint_order=0,
    )


def make_observation(
    elements: Sequence[observer.Element],
    *,
    scroll_x: float = 0.0,
    scroll_y: float = 0.0,
) -> observer.Observation:
    motion = observer.Motion(
        1000.0, 500.0, scroll_x, scroll_y, 1000.0, 1000.0, 0, 0
    )
    return observer.Observation(
        sequence=1,
        page_epoch=0,
        page_version=1,
        state_digest='d1',
        consistent=True,
        url='http://127.0.0.1/',
        title='Test',
        elements=tuple(elements),
        motion=motion,
        temporal=observer.Temporal(0, (), False, False, False, 0, True),
    )
