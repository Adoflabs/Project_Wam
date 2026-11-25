"""PyTorch data pipeline over curated examples.

A sample is a window of K consecutive steps from one run (K=1 is a single
transition). Windows never cross runs, so they never join branches or
resets, and no missing history is invented: a run of n steps gives
n - K + 1 windows, and runs shorter than K give none.

`collate` pads the variable number of elements (and of new elements) per
step and returns masks, as tensors shaped [batch, K, ...].

Typical usage example:

  data = training_data.Windows.load(pathlib.Path('data/curated/c01'),
                                    'train', k=3)
  loader = torch.utils.data.DataLoader(data, batch_size=8,
                                       collate_fn=training_data.collate)
"""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import json
import pathlib
from typing import Any

import torch
from torch.utils import data as torch_data

from browser_world.dataset import byte_encoding
from browser_world.dataset import schema

Example = Mapping[str, Any]


def load_examples(curated_dir: pathlib.Path, split: str) -> list[Example]:
    """Returns the curated examples of one split, in file order."""
    path = curated_dir / f'{split}.jsonl'
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def windows(examples: Sequence[Example], k: int) -> list[list[Example]]:
    """Returns every window of `k` consecutive steps within a run.

    Steps must be contiguous; a gap (from an excluded record) splits the
    run, since joining across it would invent a transition.
    """
    runs: dict[str, list[Example]] = collections.defaultdict(list)
    for example in examples:
        runs[example['meta']['run_id']].append(example)
    found = []
    for run in runs.values():
        run.sort(key=lambda e: e['meta']['step'])
        for start in range(len(run) - k + 1):
            window = run[start:start + k]
            steps = [e['meta']['step'] for e in window]
            if steps == list(range(steps[0], steps[0] + k)):
                found.append(window)
    return found


def _validate_byte_ids(ids: object) -> None:
    """Rejects invalid serialized byte IDs before tensor conversion."""
    if not isinstance(ids, list):
        raise ValueError('Byte IDs must be stored as a list.')
    for byte_id in ids:
        if type(byte_id) is not int:  # pylint: disable=unidiomatic-typecheck
            raise ValueError(
                f'Byte ID must be an integer: {byte_id!r}'
            )
        if not 1 <= byte_id < byte_encoding.BYTE_VOCAB_SIZE:
            raise ValueError(
                f'Byte ID outside the valid range: {byte_id}'
            )


def input_tensors(
    state: Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Converts encoded pre-action state and action dicts to CPU tensors."""
    elements = state['elements']

    def column(rows: Sequence[Mapping[str, Any]], key: str,
               dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([row[key] for row in rows], dtype=dtype)

    def byte_matrix(key: str) -> torch.Tensor:
        if not elements:
            return torch.zeros((0, 0), dtype=torch.long)
        max_len = max(len(row[key]) for row in elements)
        padded = [
            row[key] + [byte_encoding.BYTE_PAD_ID] * (max_len - len(row[key]))
            for row in elements
        ]
        return torch.tensor(padded, dtype=torch.long)

    step: dict[str, torch.Tensor] = {
        'role': column(elements, 'role', torch.long),
        'name_words': column(elements, 'name_words', torch.long),
        'name_chars': column(elements, 'name_chars', torch.long),
        'value_chars': column(elements, 'value_chars', torch.long),
        'has_value': column(elements, 'has_value', torch.bool),
        'states': column(elements, 'states', torch.float),
        'flags': column(elements, 'flags', torch.float),
        'box': column(elements, 'box', torch.float),
        'box_mask': column(elements, 'box_mask', torch.bool),
        'parent': column(elements, 'parent', torch.long),
        'page': torch.tensor(state['page'], dtype=torch.float),
        'action_kind': torch.tensor(action['kind']),
        'action_target': torch.tensor(action['target']),
        'action_text_chars': torch.tensor(action['text_chars']),
        'action_checked': torch.tensor(action['checked']),
        'action_scroll': torch.tensor(action['scroll'], dtype=torch.float),
        'action_pointer': torch.tensor(action['pointer'], dtype=torch.float),
    }
    has_action_bytes = 'text_bytes' in action
    any_elem_bytes = any(
        'name_bytes' in row or 'value_bytes' in row for row in elements)
    if has_action_bytes:
        if any('name_bytes' not in row or 'value_bytes' not in row
               for row in elements):
            raise ValueError(
                'Malformed byte-enabled record: missing element byte fields.')
    elif any_elem_bytes:
        raise ValueError(
            'Malformed byte-enabled record: missing action text_bytes.')

    if has_action_bytes:
        for element in elements:
            _validate_byte_ids(element['name_bytes'])
            _validate_byte_ids(element['value_bytes'])
        _validate_byte_ids(action['text_bytes'])
        step['name_bytes'] = byte_matrix('name_bytes')
        step['value_bytes'] = byte_matrix('value_bytes')
        step['action_text_bytes'] = torch.tensor(
            action['text_bytes'], dtype=torch.long)
    return step


def inference_batch(
    state: Mapping[str, Any],
    action: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Builds a B=1, K=1 tensor batch for live pre-action inference."""
    step = input_tensors(state, action)
    num_elements = step['role'].shape[0]
    if num_elements == 0:
        raise ValueError('Cannot run inference on an empty observation.')
    if 'name_bytes' in step:
        step['name_bytes'] = _pad_2d(
            step['name_bytes'],
            num_elements,
            max(1, step['name_bytes'].shape[1]),
            byte_encoding.BYTE_PAD_ID,
        )
        step['value_bytes'] = _pad_2d(
            step['value_bytes'],
            num_elements,
            max(1, step['value_bytes'].shape[1]),
            byte_encoding.BYTE_PAD_ID,
        )
        step['action_text_bytes'] = _pad(
            step['action_text_bytes'],
            max(1, step['action_text_bytes'].shape[0]),
            byte_encoding.BYTE_PAD_ID,
        )
    step['element_mask'] = torch.ones((num_elements,), dtype=torch.bool)
    return {
        key: value.unsqueeze(0).unsqueeze(0)
        for key, value in step.items()
    }


def _step_tensors(example: Example) -> dict[str, torch.Tensor]:
    """Converts one curated example to unpadded tensors."""
    step = input_tensors(
        example['inputs']['state'],
        example['inputs']['action'],
    )
    targets = example['targets']
    per = targets['elements']
    new = targets['new']

    def column(rows: Sequence[Mapping[str, Any]], key: str,
               dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor([row[key] for row in rows], dtype=dtype)

    def new_column(key: str, width: int | None = None) -> torch.Tensor:
        if new:
            return torch.tensor([row[key] for row in new], dtype=torch.long)
        return torch.zeros((0,) if width is None else (0, width),
                           dtype=torch.long)

    step.update({
        't_known': column(per, 'known', torch.bool),
        't_changed': column(per, 'changed', torch.float),
        't_removed': column(per, 'removed', torch.float),
        't_matched': column(per, 'matched', torch.bool),
        't_next_states': column(per, 'next_states', torch.float),
        't_value_source': column(per, 'value_source', torch.long),
        't_value_mask': column(per, 'value_mask', torch.bool),
        't_box_delta': column(per, 'box_delta', torch.float),
        't_box_mask': column(per, 'box_mask', torch.bool),
        't_outcomes': torch.tensor(targets['outcomes'], dtype=torch.float),
        't_new_role': new_column('role'),
        't_new_words': new_column('name_words', schema.MAX_NAME_WORDS),
        't_new_parent': new_column('parent_before'),
    })
    return step


class Windows(torch_data.Dataset):
    """K-step windows of curated examples, as lists of tensor dicts."""

    def __init__(self, items: Sequence[Sequence[Example]]):
        """Wraps precomputed windows; see `load`."""
        self._items = [[_step_tensors(e) for e in window] for window in items]
        self.meta = [[e['meta'] for e in window] for window in items]

    @classmethod
    def load(cls, curated_dir: pathlib.Path, split: str,
             k: int) -> 'Windows':
        """Loads every K-step window of a split."""
        return cls(windows(load_examples(curated_dir, split), k))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> list[dict[str, torch.Tensor]]:
        return self._items[index]


# Element-indexed keys: padded along the element axis.
_ELEMENT_KEYS = ('role', 'name_words', 'name_chars', 'value_chars',
                 'has_value', 'states', 'flags', 'box', 'box_mask', 'parent',
                 't_known', 't_changed', 't_removed', 't_matched',
                 't_next_states', 't_value_source', 't_value_mask',
                 't_box_delta', 't_box_mask')
_NEW_KEYS = ('t_new_role', 't_new_words', 't_new_parent')
# Padding for index-valued keys, so padding never points at element 0.
_PAD_VALUES = {'parent': -1, 't_new_parent': -1}


def collate(batch: Sequence[Sequence[Mapping[str, torch.Tensor]]]
            ) -> dict[str, torch.Tensor]:
    """Stacks windows into [batch, K, ...] tensors with padding masks.

    Adds `element_mask` [B, K, N] and `t_new_mask` [B, K, M], true for real
    entries. All targets must also be combined with their own masks
    (`t_known`, `t_matched`, `t_value_mask`, `t_box_mask`).
    """
    k = len(batch[0])
    byte_flags = {'name_bytes' in step for window in batch for step in window}
    if len(byte_flags) > 1:
        raise ValueError(
            'Cannot mix legacy-only and byte-enabled records in one batch.')
    has_bytes = next(iter(byte_flags))
    n = max(step['role'].shape[0] for window in batch for step in window)
    m = max(step['t_new_role'].shape[0] for window in batch
            for step in window)
    if has_bytes:
        l_name = max(1, max(step['name_bytes'].shape[1]
                            for window in batch for step in window))
        l_value = max(1, max(step['value_bytes'].shape[1]
                             for window in batch for step in window))
        l_action = max(1, max(step['action_text_bytes'].shape[0]
                              for window in batch for step in window))
    out: dict[str, list[torch.Tensor]] = collections.defaultdict(list)
    for window in batch:
        if len(window) != k:
            raise ValueError('All windows in a batch must have the same K.')
        for step in window:
            for key, value in step.items():
                if key in _ELEMENT_KEYS:
                    value = _pad(value, n, _PAD_VALUES.get(key, 0))
                elif key in _NEW_KEYS:
                    value = _pad(value, m, _PAD_VALUES.get(key, 0))
                elif key == 'name_bytes':
                    value = _pad_2d(
                        value, n, l_name, byte_encoding.BYTE_PAD_ID)
                elif key == 'value_bytes':
                    value = _pad_2d(
                        value, n, l_value, byte_encoding.BYTE_PAD_ID)
                elif key == 'action_text_bytes':
                    value = _pad(value, l_action, byte_encoding.BYTE_PAD_ID)
                out[key].append(value)
            out['element_mask'].append(
                torch.arange(n) < step['role'].shape[0])
            out['t_new_mask'].append(
                torch.arange(m) < step['t_new_role'].shape[0])
            out['t_new_count'].append(
                torch.tensor(step['t_new_role'].shape[0], dtype=torch.float))
    batch_size = len(batch)
    return {key: torch.stack(values).reshape(batch_size, k,
                                             *values[0].shape)
            for key, values in out.items()}


def _pad(
    value: torch.Tensor,
    length: int,
    fill: int,
) -> torch.Tensor:
    """Pads axis zero without reducing the original tensor."""
    missing = length - value.shape[0]
    if missing < 0:
        raise ValueError(
            'Padding length cannot be smaller than the input length.'
        )
    if missing == 0:
        return value
    padding = value.new_full(
        (missing, *value.shape[1:]),
        fill,
    )
    return torch.cat((value, padding), dim=0)


def _pad_2d(
    value: torch.Tensor,
    rows: int,
    cols: int,
    fill: int,
) -> torch.Tensor:
    """Pads both axes of a two-dimensional tensor."""
    if value.ndim != 2:
        raise ValueError('Expected a two-dimensional tensor.')
    input_rows, input_cols = value.shape
    if rows < input_rows or cols < input_cols:
        raise ValueError(
            'Padding cannot reduce an existing tensor dimension.'
        )
    output = value.new_full((rows, cols), fill)
    output[:input_rows, :input_cols] = value
    return output
