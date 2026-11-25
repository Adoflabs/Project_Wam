"""Exact content representation, source masks, targets, and resolver."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import enum
import pathlib
from typing import Any

import torch

from browser_world.interaction import observer
from browser_world.wam2 import model

KEEP = 0
EMPTY = 1
ABSENT = 2
ACTION_TEXT = 3
UNKNOWN = 4
FIXED_SOURCE_COUNT = 5

CONTENT_HEAD_KEYS = frozenset({
    'content_query.weight',
    'content_query.bias',
    'content_key.weight',
    'content_key.bias',
    'content_fixed.weight',
    'content_fixed.bias',
})


class ContentStatus(enum.IntEnum):
    """Availability of exact textual content."""

    UNKNOWN = 0
    ABSENT = 1
    KNOWN = 2


@dataclasses.dataclass(frozen=True)
class ExactText:
    """Exact content and its availability."""

    status: ContentStatus
    text: str | None

    def __post_init__(self) -> None:
        if self.status == ContentStatus.KNOWN:
            if not isinstance(self.text, str):
                raise ValueError('KNOWN content requires a string text.')
        elif self.text is not None:
            raise ValueError(
                f'Content status {self.status.name} requires text=None.'
            )


@dataclasses.dataclass(frozen=True)
class EntityContent:
    """Exact content associated with one persistent slot."""

    name: ExactText
    value: ExactText


def source_count(slot_count: int = model.SLOT_COUNT) -> int:
    """Returns 5 + 2*N content-source classes for N slots."""
    return FIXED_SOURCE_COUNT + 2 * slot_count


def empty_slot_content(slot_count: int) -> tuple[EntityContent, ...]:
    """Returns all-UNKNOWN EntityContent tuple of length slot_count."""
    unk = ExactText(ContentStatus.UNKNOWN, None)
    return tuple(EntityContent(name=unk, value=unk) for _ in range(slot_count))


def extract_observation_content(
    obs: observer.Observation,
    elem_to_slot: Sequence[int],
    slot_count: int = model.SLOT_COUNT,
) -> tuple[EntityContent, ...]:
    """Populates aligned EntityContent from actual Observer elements."""
    slots = list(empty_slot_content(slot_count))
    for elem, slot in zip(obs.elements, elem_to_slot):
        name = ExactText(ContentStatus.KNOWN, elem.name)
        value = (
            ExactText(ContentStatus.ABSENT, None)
            if elem.value is None
            else ExactText(ContentStatus.KNOWN, elem.value)
        )
        slots[slot] = EntityContent(name=name, value=value)
    return tuple(slots)


def legal_source_mask(
    prev_content: Sequence[EntityContent],
    prev_present: torch.Tensor,
    action_text: str | None,
    device: torch.device,
) -> torch.Tensor:
    """Builds [1, N, 2, 5 + 2*N] bool mask of legal pre-action sources."""
    slot_count = len(prev_content)
    total_sources = source_count(slot_count)
    mask = torch.zeros(
        (1, slot_count, 2, total_sources), dtype=torch.bool, device=device
    )
    prev_on = [bool(prev_present[0, i].item() > 0.5) for i in range(slot_count)]

    # Fixed sources: EMPTY, ABSENT, UNKNOWN always legal; ACTION_TEXT if set.
    mask[0, :, :, EMPTY] = True
    mask[0, :, :, ABSENT] = True
    mask[0, :, :, UNKNOWN] = True
    if action_text is not None:
        mask[0, :, :, ACTION_TEXT] = True

    for i in range(slot_count):
        if prev_on[i]:
            if prev_content[i].name.status != ContentStatus.UNKNOWN:
                mask[0, i, 0, KEEP] = True
            if prev_content[i].value.status != ContentStatus.UNKNOWN:
                mask[0, i, 1, KEEP] = True

    for j in range(slot_count):
        if not prev_on[j]:
            continue
        if prev_content[j].name.status == ContentStatus.KNOWN:
            mask[0, :, :, FIXED_SOURCE_COUNT + 2 * j] = True
        if prev_content[j].value.status == ContentStatus.KNOWN:
            mask[0, :, :, FIXED_SOURCE_COUNT + 2 * j + 1] = True

    return mask


def _matching_sources(
    target: ExactText,
    prev_field: ExactText,
    slot_was_present: bool,
    prev_content: Sequence[EntityContent],
    prev_on: Sequence[bool],
    action_text: str | None,
) -> list[int]:
    """Returns all legal source indices producing `target`."""
    if target.status == ContentStatus.UNKNOWN:
        return [UNKNOWN]

    matches: list[int] = []
    if slot_was_present and prev_field == target:
        matches.append(KEEP)

    if target.status == ContentStatus.ABSENT:
        matches.append(ABSENT)
        return matches

    assert target.text is not None
    if target.text == '':
        matches.append(EMPTY)
    if action_text is not None and action_text == target.text:
        matches.append(ACTION_TEXT)

    for j, content in enumerate(prev_content):
        if not prev_on[j]:
            continue
        if (
            content.name.status == ContentStatus.KNOWN
            and content.name.text == target.text
        ):
            matches.append(FIXED_SOURCE_COUNT + 2 * j)
        if (
            content.value.status == ContentStatus.KNOWN
            and content.value.text == target.text
        ):
            matches.append(FIXED_SOURCE_COUNT + 2 * j + 1)

    return matches if matches else [UNKNOWN]


def build_content_targets(
    prev_content: Sequence[EntityContent],
    prev_present: torch.Tensor,
    after_content: Sequence[EntityContent],
    after_present: torch.Tensor,
    existence_known: torch.Tensor,
    action_text: str | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (target_mask [1,N,2,5+2*N], field_eligible [1,N,2])."""
    slot_count = len(prev_content)
    total_sources = source_count(slot_count)
    target_mask = torch.zeros(
        (1, slot_count, 2, total_sources), dtype=torch.bool, device=device
    )
    field_eligible = torch.zeros(
        (1, slot_count, 2), dtype=torch.bool, device=device
    )
    prev_on = [bool(prev_present[0, i].item() > 0.5) for i in range(slot_count)]

    for i in range(slot_count):
        if not (
            after_present[0, i].item() > 0.5
            and bool(existence_known[0, i].item())
        ):
            continue
        field_eligible[0, i, :] = True
        for f_idx, (prev_f, aft_f) in enumerate((
            (prev_content[i].name, after_content[i].name),
            (prev_content[i].value, after_content[i].value),
        )):
            for src_idx in _matching_sources(
                aft_f, prev_f, prev_on[i], prev_content, prev_on, action_text
            ):
                target_mask[0, i, f_idx, src_idx] = True

    return target_mask, field_eligible


def _resolve_one_field(
    src_idx: int,
    prev_field: ExactText,
    slot_was_present: bool,
    prev_content: Sequence[EntityContent],
    prev_on: Sequence[bool],
    action_text: str | None,
) -> ExactText:
    if src_idx == KEEP:
        if slot_was_present and prev_field.status != ContentStatus.UNKNOWN:
            return prev_field
        return ExactText(ContentStatus.UNKNOWN, None)
    if src_idx == EMPTY:
        return ExactText(ContentStatus.KNOWN, '')
    if src_idx == ABSENT:
        return ExactText(ContentStatus.ABSENT, None)
    if src_idx == ACTION_TEXT:
        if action_text is not None:
            return ExactText(ContentStatus.KNOWN, action_text)
        return ExactText(ContentStatus.UNKNOWN, None)
    if src_idx == UNKNOWN:
        return ExactText(ContentStatus.UNKNOWN, None)

    rel = src_idx - FIXED_SOURCE_COUNT
    if rel < 0:
        return ExactText(ContentStatus.UNKNOWN, None)
    j, is_val = divmod(rel, 2)
    if j >= len(prev_content) or not prev_on[j]:
        return ExactText(ContentStatus.UNKNOWN, None)
    ref = prev_content[j].value if is_val else prev_content[j].name
    if ref.status == ContentStatus.KNOWN:
        return ref
    return ExactText(ContentStatus.UNKNOWN, None)


def resolve_content(
    prev_content: Sequence[EntityContent],
    prev_present: torch.Tensor,
    action_text: str | None,
    selected_sources: torch.Tensor,
    pred_present: torch.Tensor,
) -> tuple[EntityContent, ...]:
    """Deterministically resolves future exact content without leakage."""
    slot_count = len(prev_content)
    prev_on = [bool(prev_present[0, i].item() > 0.5) for i in range(slot_count)]
    resolved: list[EntityContent] = []
    unk = ExactText(ContentStatus.UNKNOWN, None)

    for i in range(slot_count):
        if pred_present[0, i].item() < 0.5:
            resolved.append(EntityContent(name=unk, value=unk))
            continue
        name_src = int(selected_sources[0, i, 0].item())
        val_src = int(selected_sources[0, i, 1].item())
        resolved.append(
            EntityContent(
                name=_resolve_one_field(
                    name_src, prev_content[i].name, prev_on[i],
                    prev_content, prev_on, action_text,
                ),
                value=_resolve_one_field(
                    val_src, prev_content[i].value, prev_on[i],
                    prev_content, prev_on, action_text,
                ),
            )
        )
    return tuple(resolved)


def load_from_dynamics_checkpoint(
    ckpt_path: pathlib.Path,
    device: torch.device,
) -> tuple[model.WorldActionModel, dict[str, Any]]:
    """Loads pre-content dynamics checkpoint and initializes content head."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    world = model.WorldActionModel(
        role_count=len(ckpt['role_vocabulary']),
        slot_count=int(ckpt['slot_count']),
    ).to(device)
    incompat = world.load_state_dict(ckpt['model_state_dict'], strict=False)
    if incompat.unexpected_keys:
        raise RuntimeError(
            f'Unexpected checkpoint keys: {incompat.unexpected_keys}'
        )
    if set(incompat.missing_keys) != CONTENT_HEAD_KEYS:
        raise RuntimeError(
            f'Expected missing keys {sorted(CONTENT_HEAD_KEYS)}, '
            f'got {sorted(incompat.missing_keys)}'
        )
    return world, ckpt
