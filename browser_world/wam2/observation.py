"""Persistent observation, alignment, and action encoding."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import pathlib
from typing import Protocol

import torch

from browser_world.dataset import curate
from browser_world.dataset import schema
from browser_world.interaction import executor
from browser_world.interaction import observer
from browser_world.wam2 import content
from browser_world.wam2 import model
from browser_world.wam2 import semantic

FLAG_NAMES = ('visible', 'in_viewport', 'obstructed', 'pinned')
AFFORDANCE_NAMES = ('click', 'type', 'select', 'check')
RESERVED_NEW_SLOTS = 16
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_VOCAB_PATH = _REPO_ROOT / 'data' / 'curated' / 'bytes_v1' / 'vocab.json'


class TextEncoder(Protocol):
    """Protocol for frozen or deterministic text encoders."""

    device: torch.device

    def encode(
        self, texts: Sequence[str], max_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns hidden [B,L,384], valid mask [B,L], and CPU offsets."""


@dataclasses.dataclass(frozen=True)
class ObservationInputs:
    """Model inputs and alignment metadata for one observation."""

    names: torch.Tensor
    values: torch.Tensor
    roles: torch.Tensor
    states: torch.Tensor
    flags: torch.Tensor
    boxes: torch.Tensor
    affordances: torch.Tensor
    parents: torch.Tensor
    present: torch.Tensor

    box_known: torch.Tensor
    matched: torch.Tensor
    existence_known: torch.Tensor

    element_to_slot: tuple[int, ...]
    slot_to_element: tuple[int | None, ...]
    parent_known: torch.Tensor


@dataclasses.dataclass(frozen=True)
class AlignedObservation:
    """An actual observation and its persistent slot assignment."""

    observation: observer.Observation
    inputs: ObservationInputs
    allocated_slots: torch.Tensor
    exact_content: tuple[content.EntityContent, ...] = ()


@dataclasses.dataclass(frozen=True)
class ActionInputs:
    """Numerical representation of one proposed action."""

    kind: torch.Tensor
    target: torch.Tensor
    text: torch.Tensor
    checked: torch.Tensor
    scroll: torch.Tensor
    pointer: torch.Tensor


def load_role_vocab(
    vocab_path: pathlib.Path = DEFAULT_VOCAB_PATH,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Loads the training-derived role vocabulary in exact saved order."""
    data = json.loads(vocab_path.read_text(encoding='utf-8'))
    roles = tuple(data['roles'])
    return {role: idx for idx, role in enumerate(roles)}, roles


def assign_slots(
    current: observer.Observation,
    previous: AlignedObservation | None,
    slot_count: int = model.SLOT_COUNT,
) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
    """Assigns current elements to persistent slots.

    Returns the element-to-slot mapping and CPU Boolean matched
    and existence-known masks of shape [1, N].
    """
    num_elements = len(current.elements)
    matched = torch.zeros((1, slot_count), dtype=torch.bool)
    existence_known = torch.ones((1, slot_count), dtype=torch.bool)

    if previous is None:
        max_initial = slot_count - RESERVED_NEW_SLOTS
        if max_initial < 1:
            raise ValueError(
                f'slot_count={slot_count} cannot reserve '
                f'{RESERVED_NEW_SLOTS} slots.'
            )
        if num_elements > max_initial:
            raise ValueError(
                f'Initial observation has {num_elements} elements, exceeding '
                f'initial limit {max_initial} (reserving {RESERVED_NEW_SLOTS}).'
            )
        return tuple(range(num_elements)), matched, existence_known

    alignment = curate.align(previous.observation, current)
    prev_alloc = previous.allocated_slots[0]
    used_in_window = set(
        torch.nonzero(prev_alloc, as_tuple=False).flatten().tolist()
    )
    elem_to_slot = [-1] * num_elements

    for cur_idx, prev_idx in enumerate(alignment.after_to_before):
        if prev_idx >= 0:
            slot = previous.inputs.element_to_slot[prev_idx]
            elem_to_slot[cur_idx] = slot
            matched[0, slot] = True
            used_in_window.add(slot)

    next_free = 0
    for cur_idx, slot in enumerate(elem_to_slot):
        if slot >= 0:
            continue
        while next_free < slot_count and next_free in used_in_window:
            next_free += 1
        if next_free >= slot_count:
            raise ValueError(
                f'Slot capacity {slot_count} exhausted when allocating '
                f'element {cur_idx}.'
            )
        elem_to_slot[cur_idx] = next_free
        used_in_window.add(next_free)

    for prev_idx in alignment.ambiguous_before:
        slot = previous.inputs.element_to_slot[prev_idx]
        existence_known[0, slot] = False
    for cur_idx in alignment.ambiguous_after:
        slot = elem_to_slot[cur_idx]
        existence_known[0, slot] = False

    return tuple(elem_to_slot), matched, existence_known


def encode_texts(
    texts: tuple[str, ...],
    encoder: TextEncoder,
    batch_size: int = 32,
) -> torch.Tensor:
    """Returns pooled FP32 semantic features [len(texts), 384]."""
    if not texts:
        return torch.zeros(
            (0, semantic.TEXT_DIM), dtype=torch.float32, device=encoder.device
        )
    if batch_size < 1:
        raise ValueError(f'batch_size must be >= 1, got {batch_size}.')

    unique_texts = list(dict.fromkeys(texts))
    text_to_idx = {text: idx for idx, text in enumerate(unique_texts)}
    pooled_chunks: list[torch.Tensor] = []
    for start in range(0, len(unique_texts), batch_size):
        batch = unique_texts[start:start + batch_size]
        hidden, mask, _ = encoder.encode(batch, max_tokens=128)
        pooled_chunks.append(semantic.masked_mean(hidden, mask))

    unique_features = torch.cat(pooled_chunks, dim=0)
    gather_idx = torch.tensor(
        [text_to_idx[t] for t in texts],
        dtype=torch.long,
        device=unique_features.device,
    )
    return unique_features.index_select(0, gather_idx)


def encode_observation(
    current: observer.Observation,
    encoder: TextEncoder,
    role_to_index: dict[str, int],
    slot_count: int = model.SLOT_COUNT,
    previous: AlignedObservation | None = None,
) -> AlignedObservation:
    """Encodes an actual observation into aligned model input tensors."""
    elem_to_slot, matched, existence_known = assign_slots(
        current, previous, slot_count=slot_count
    )
    slot_to_elem: list[int | None] = [None] * slot_count
    for elem_idx, slot in enumerate(elem_to_slot):
        slot_to_elem[slot] = elem_idx

    device = encoder.device
    role_count = len(role_to_index)
    state_count = len(schema.STATE_NAMES)
    flag_count = len(FLAG_NAMES)
    aff_count = len(AFFORDANCE_NAMES)

    names = torch.zeros(
        (1, slot_count, semantic.TEXT_DIM), dtype=torch.float32, device=device
    )
    values = torch.zeros(
        (1, slot_count, semantic.TEXT_DIM), dtype=torch.float32, device=device
    )
    roles = torch.zeros(
        (1, slot_count, role_count), dtype=torch.float32, device=device
    )
    states = torch.zeros(
        (1, slot_count, state_count), dtype=torch.float32, device=device
    )
    flags = torch.zeros(
        (1, slot_count, flag_count), dtype=torch.float32, device=device
    )
    boxes = torch.zeros(
        (1, slot_count, 4), dtype=torch.float32, device=device
    )
    affordances = torch.zeros(
        (1, slot_count, aff_count), dtype=torch.float32, device=device
    )
    parents = torch.zeros(
        (1, slot_count, slot_count + 1), dtype=torch.float32, device=device
    )
    present = torch.zeros((1, slot_count), dtype=torch.float32, device=device)
    box_known = torch.zeros((1, slot_count), dtype=torch.bool, device=device)
    parent_known = torch.zeros((1, slot_count), dtype=torch.bool, device=device)

    allocated = (
        previous.allocated_slots.clone()
        if previous is not None
        else torch.zeros((1, slot_count), dtype=torch.bool)
    )

    if current.elements:
        raw_names = tuple(elem.name or '' for elem in current.elements)
        raw_values = tuple(elem.value or '' for elem in current.elements)
        name_embs = encode_texts(raw_names, encoder)
        value_embs = encode_texts(raw_values, encoder)

        slot_indices = torch.tensor(
            elem_to_slot, dtype=torch.long, device=device
        )
        names[0, slot_indices] = name_embs
        values[0, slot_indices] = value_embs
        present[0, slot_indices] = 1.0

        motion = current.motion
        vw = max(float(motion.viewport_width), 1.0)
        vh = max(float(motion.viewport_height), 1.0)
        node_id_to_slot = {
            elem.node_id: elem_to_slot[idx]
            for idx, elem in enumerate(current.elements)
        }
        unk_role = role_to_index.get('<unk>')

        for elem_idx, elem in enumerate(current.elements):
            slot = elem_to_slot[elem_idx]
            allocated[0, slot] = True

            r_idx = role_to_index.get(elem.role, unk_role)
            if r_idx is None:
                raise KeyError(
                    f'Unknown role {elem.role!r} and no <unk> in role_to_index.'
                )
            roles[0, slot, r_idx] = 1.0

            for s_idx, s_name in enumerate(schema.STATE_NAMES):
                states[0, slot, s_idx] = 1.0 if elem.states.get(s_name) else 0.0

            flags[0, slot, 0] = 1.0 if elem.visible else 0.0
            flags[0, slot, 1] = 1.0 if elem.in_viewport else 0.0
            flags[0, slot, 2] = 1.0 if elem.obstructed else 0.0
            flags[0, slot, 3] = 1.0 if elem.pinned else 0.0

            if elem.box is not None:
                bx = (float(elem.box.x) - float(motion.scroll_x)) / vw
                by = (float(elem.box.y) - float(motion.scroll_y)) / vh
                boxes[0, slot, 0] = bx
                boxes[0, slot, 1] = by
                boxes[0, slot, 2] = float(elem.box.width) / vw
                boxes[0, slot, 3] = float(elem.box.height) / vh
                box_known[0, slot] = True

            if elem.parent_id is None:
                parents[0, slot, 0] = 1.0
                parent_known[0, slot] = True
            elif elem.parent_id in node_id_to_slot:
                p_slot = node_id_to_slot[elem.parent_id]
                parents[0, slot, p_slot + 1] = 1.0
                parent_known[0, slot] = True
            else:
                parents[0, slot, 0] = 1.0
                parent_known[0, slot] = False

        aff_index = {name: idx for idx, name in enumerate(AFFORDANCE_NAMES)}
        for aff in executor.available_actions(current):
            if aff.target is None or aff.kind not in aff_index:
                continue
            target_slot = node_id_to_slot.get(aff.target)
            if target_slot is not None:
                affordances[0, target_slot, aff_index[aff.kind]] = 1.0

    inputs = ObservationInputs(
        names=names,
        values=values,
        roles=roles,
        states=states,
        flags=flags,
        boxes=boxes,
        affordances=affordances,
        parents=parents,
        present=present,
        box_known=box_known,
        matched=matched.to(device),
        existence_known=existence_known.to(device),
        element_to_slot=elem_to_slot,
        slot_to_element=tuple(slot_to_elem),
        parent_known=parent_known,
    )
    exact_content = content.extract_observation_content(
        current, elem_to_slot, slot_count=slot_count
    )
    return AlignedObservation(
        observation=current,
        inputs=inputs,
        allocated_slots=allocated,
        exact_content=exact_content,
    )


def observe(
    world: model.WorldActionModel,
    aligned: AlignedObservation,
    previous: model.Belief | None = None,
) -> model.Belief:
    """Creates a neural belief from aligned actual evidence."""
    inputs = aligned.inputs
    return world.observe(
        names=inputs.names,
        values=inputs.values,
        roles=inputs.roles,
        states=inputs.states,
        flags=inputs.flags,
        boxes=inputs.boxes,
        affordances=inputs.affordances,
        parents=inputs.parents,
        present=inputs.present,
        previous=previous,
        matched=inputs.matched if previous is not None else None,
    )


def encode_action(
    action: executor.Action,
    aligned: AlignedObservation,
    encoder: TextEncoder,
    pointer: tuple[float, float],
) -> ActionInputs:
    """Encodes a proposed intervention using pre-action evidence only."""
    if action.kind not in schema.ACTION_KINDS:
        raise ValueError(f'Unknown action kind {action.kind!r}.')

    device = encoder.device
    kind_idx = schema.ACTION_KINDS.index(action.kind)
    kind = torch.tensor([kind_idx], dtype=torch.long, device=device)

    if action.kind == executor.SCROLL:
        if action.target is not None:
            raise ValueError('Scroll action must have target=None.')
        target_slot = -1
    else:
        if action.target is None:
            raise ValueError(f'Action kind {action.kind!r} requires a target.')
        node_to_slot = {
            elem.node_id: aligned.inputs.element_to_slot[idx]
            for idx, elem in enumerate(aligned.observation.elements)
        }
        if action.target not in node_to_slot:
            raise ValueError(
                f'Action target node_id={action.target} not found in '
                f'pre-action observation.'
            )
        target_slot = node_to_slot[action.target]

    target = torch.tensor([target_slot], dtype=torch.long, device=device)

    if action.text:
        text = encode_texts((action.text,), encoder)
    else:
        text = torch.zeros(
            (1, semantic.TEXT_DIM), dtype=torch.float32, device=device
        )

    checked = torch.zeros((1, 3), dtype=torch.float32, device=device)
    if action.checked is None:
        checked[0, 0] = 1.0
    elif action.checked is False:
        checked[0, 1] = 1.0
    else:
        checked[0, 2] = 1.0

    motion = aligned.observation.motion
    vw = max(float(motion.viewport_width), 1.0)
    vh = max(float(motion.viewport_height), 1.0)
    scroll_val = float(action.scroll_y or 0.0) / vh
    scroll = torch.tensor([[scroll_val]], dtype=torch.float32, device=device)
    ptr_tensor = torch.tensor(
        [[float(pointer[0]) / vw, float(pointer[1]) / vh]],
        dtype=torch.float32,
        device=device,
    )
    return ActionInputs(
        kind=kind,
        target=target,
        text=text,
        checked=checked,
        scroll=scroll,
        pointer=ptr_tensor,
    )
