"""C0.1: validate raw experience and build curated training examples.

Steps:
  1. `validate()` checks the raw dataset strictly: schema, manifest
     correspondence, unique runs and steps, state boundaries, split
     membership, action targets, executor results, branch starting states
     and completion metadata. Fatal problems stop curation; records that
     are valid but unsuitable (not settled, not consistent) are excluded
     with a stated reason.
  2. `align()` matches before-state elements to after-state elements one to
     one, in tiers, and marks what it cannot decide as ambiguous.
  3. `curate()` turns each transition into model inputs and supervised
     targets with masks, fitting vocabularies on the training split only,
     and writes per-split JSONL plus a manifest.

Inputs hold only what an agent could know before acting: the observed
elements (roles, names, values, states, relationships, geometry), the
page's scroll and size, the agent's own pointer, and the action. Targets
come from evidence (the aligned after-state, the page's alerts and
headings, backend truth), never from booking rules written here.

Usage:

  python -m browser_world.dataset.curate --raw data/raw/e04/main \\
      --out data/curated/c01
"""

from __future__ import annotations

import argparse
import collections
from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import hashlib
import json
import pathlib
import re
import subprocess
from typing import Any

from browser_world.interaction import observer
from browser_world.dataset import byte_encoding
from browser_world.dataset import schema
from browser_world.collection import recorder
from browser_world.environment import scenarios

SUPPORTED_SCHEMA = 3
SPLITS = tuple(scenarios.SPLITS)
BOX_TOLERANCE_PX = 1.0
schema.PAD, schema.UNK = 0, 1
byte_encoding.BYTE_PAD_ID = 0
byte_encoding.BYTE_VOCAB_SIZE = 257
INPUT_ENCODINGS = ('legacy', 'utf8_bytes_v1')
MANIFEST_STATUSES = frozenset({'complete', 'failed', 'timed_out', 'refused',
                               'abandoned'})
_WORD_RE = re.compile(r'[a-z0-9]+')


def _validate_input_encoding(input_encoding: str) -> None:
    if input_encoding not in INPUT_ENCODINGS:
        raise ValueError(
            f'Unsupported input_encoding {input_encoding!r}; '
            f'expected one of {INPUT_ENCODINGS}')


# --- Loading -----------------------------------------------------------

def load(raw_dir: pathlib.Path
         ) -> tuple[dict[str, Any] | None, dict[str, list[dict[str, Any]]]]:
    """Returns the manifest (None if missing) and the records per split."""
    manifest_path = raw_dir / 'manifest.json'
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.exists() else None)
    records = {}
    for split in SPLITS:
        path = raw_dir / split / recorder.TRANSITIONS_FILE
        records[split] = recorder.read_jsonl(path) if path.exists() else []
    return manifest, records


def _observation(data: Mapping[str, Any]) -> observer.Observation:
    return observer.observation_from_dict(data)


# --- 1. Validation -----------------------------------------------------

@dataclasses.dataclass
class Validation:
    """Outcome of `validate`.

    Attributes:
      problems: Fatal defects; curation refuses to run while any exist.
      exclusions: Valid records not suited to training, keyed by
        'run_id#step', with the reason.
    """

    problems: list[str] = dataclasses.field(default_factory=list)
    exclusions: dict[str, str] = dataclasses.field(default_factory=dict)


def validate(raw_dir: pathlib.Path) -> Validation:
    """Checks a raw dataset strictly; see the module docstring."""
    report = Validation()
    manifest, records = load(raw_dir)
    if manifest is None:
        report.problems.append('manifest.json is missing')
    run_split: dict[str, str] = {}
    runs: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for split, split_records in records.items():
        for record in split_records:
            run_id = str(record.get('run_id'))
            where = f'{run_id}#{record.get("step")}'
            if record.get('schema_version') != SUPPORTED_SCHEMA:
                report.problems.append(
                    f'{where}: schema {record.get("schema_version")}')
                continue
            if run_split.setdefault(run_id, split) != split:
                report.problems.append(
                    f'{run_id}: in {run_split[run_id]} and {split}')
            runs[run_id].append(record)
            report.problems.extend(f'{where}: {p}'
                                   for p in _record_problems(record, split))
            reason = _exclusion(record)
            if reason:
                report.exclusions[where] = reason
    for run_id, run in runs.items():
        report.problems.extend(_run_problems(run_id, run))
    if manifest is not None:
        report.problems.extend(_manifest_problems(manifest, runs))
    report.problems.extend(_branch_problems(runs))
    return report


def _record_problems(record: Mapping[str, Any], split: str) -> list[str]:
    problems = []
    meta = record.get('scenario') or {}
    family = meta.get('family', '')
    try:
        family_split = scenarios.split_of(family)
    except ValueError:
        return [f'unknown family {family!r}']
    if not family_split == split == meta.get('split'):
        problems.append(f'family {family} belongs to {family_split}, file '
                        f'is {split}, metadata says {meta.get("split")}')
    if not str(meta.get('scenario_id', '')).startswith(f'{family}-'):
        problems.append(f'scenario id {meta.get("scenario_id")} does not '
                        f'match family {family}')
    action, target = record['action'], record['target']
    elements = record['before']['elements']
    if action['kind'] not in schema.ACTION_KINDS:
        problems.append(f'unsupported action kind {action["kind"]}')
    if action['kind'] == 'scroll':
        if action['target'] is not None or target is not None:
            problems.append('scroll with an element target')
    elif target is None or not 0 <= target['index'] < len(elements):
        problems.append('target index outside the before-state')
    else:
        element = elements[target['index']]
        if not (element['node_id'] == target['node_id'] == action['target']
                and element['role'] == target['role']
                and element['name'] == target['name']):
            problems.append('target does not match the before-state element')
    result = record['result']
    if result['action'] != action:
        problems.append('executor result is for a different action')
    if not result['executed'] or result['refusal'] is not None:
        problems.append('executor did not execute the recorded action')
    if result['observation_sequence'] != record['before']['sequence']:
        problems.append('executor acted on a different observation')
    return problems


def _exclusion(record: Mapping[str, Any]) -> str | None:
    if not record['settled']:
        return 'page did not settle before the after-observation'
    for moment in ('before', 'after'):
        if not record[moment].get('consistent'):
            return f'{moment}-observation not captured consistently'
    return None


def _run_problems(run_id: str,
                  run: list[dict[str, Any]]) -> list[str]:
    problems = []
    steps = [r['step'] for r in run]
    if len(set(steps)) != len(steps):
        problems.append(f'{run_id}: duplicate steps {steps}')
    run.sort(key=lambda r: r['step'])
    if [r['step'] for r in run] != list(range(len(run))):
        problems.append(f'{run_id}: steps are not 0..n-1: {steps}')
    for previous, current in zip(run, run[1:]):
        after = _observation(previous['after'])
        before = _observation(current['before'])
        found = recorder.differences(recorder.canonical(after),
                                     recorder.canonical(before))
        if (after.page_epoch, after.page_version, after.state_digest) != (
                before.page_epoch, before.page_version, before.state_digest):
            found.append('page epoch, version or live state changed')
        if previous['truth']['after'] != current['truth']['before']:
            found.append('backend truth changed')
        if found:
            problems.append(f'{run_id}#{current["step"]}: boundary '
                            f'{found[:2]}')
    return problems


def _manifest_problems(manifest: Mapping[str, Any],
                       runs: Mapping[str, list[dict[str, Any]]]
                       ) -> list[str]:
    problems = []
    if manifest.get('schema_version') != SUPPORTED_SCHEMA:
        problems.append(f'manifest schema {manifest.get("schema_version")}')
    listed = set()
    for episode in manifest.get('episodes', []):
        run_id = episode.get('run_id')
        if episode.get('status') not in MANIFEST_STATUSES:
            problems.append(f'{episode.get("episode_id")}: status '
                            f'{episode.get("status")}')
        # A branch run ends after its one alternative action by design, so
        # only non-branch runs must record how the page ended.
        if (episode.get('status') == 'complete' and not episode.get('ended')
                and not episode.get('branch')):
            problems.append(f'{episode.get("episode_id")}: complete but no '
                            'end recorded')
        if run_id is None:
            if episode.get('steps'):
                problems.append(f'{episode.get("episode_id")}: steps but no '
                                'run')
            continue
        listed.add(run_id)
        run = runs.get(run_id, [])
        if len(run) != episode.get('steps'):
            problems.append(f'{run_id}: manifest says {episode.get("steps")} '
                            f'steps, records have {len(run)}')
            continue
        meta = run[0]['scenario'] if run else {}
        for field in ('split', 'family', 'seed', 'policy'):
            if run and meta.get(field) != episode.get(field):
                problems.append(f'{run_id}: manifest {field}='
                                f'{episode.get(field)!r}, records '
                                f'{meta.get(field)!r}')
    for run_id in set(runs) - listed:
        problems.append(f'{run_id}: records not listed in the manifest')
    return problems


def _branch_problems(runs: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for run in runs.values():
        branch = run[0]['scenario'].get('branch') if run else None
        if branch:
            groups[branch['group']].append(run[0])
    problems = []
    for group, firsts in groups.items():
        if len({r['scenario']['split'] for r in firsts}) > 1:
            problems.append(f'{group}: branches cross splits')
        if len({r['scenario']['branch']['point_fingerprint']
                for r in firsts}) > 1:
            problems.append(f'{group}: branches disagree on the start')
        reference = recorder.canonical(_observation(firsts[0]['before']))
        for first in firsts[1:]:
            if recorder.differences(reference, recorder.canonical(
                    _observation(first['before']))):
                problems.append(f'{group}: starting observations differ')
            if first['truth']['before'] != firsts[0]['truth']['before']:
                problems.append(f'{group}: starting backend truth differs')
    return problems


# --- 2. Alignment ------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Alignment:
    """A one-to-one matching of before-state and after-state elements.

    Attributes:
      before_to_after: For each before element, its after index or -1.
      after_to_before: For each after element, its before index or -1.
      ambiguous_before: Unmatched before elements that could correspond
        to more than one after element (or vice versa); their targets are
        masked.
      ambiguous_after: The after elements involved in those ambiguities.
      tiers: How many pairs each tier matched.
    """

    before_to_after: tuple[int, ...]
    after_to_before: tuple[int, ...]
    ambiguous_before: frozenset[int]
    ambiguous_after: frozenset[int]
    tiers: Mapping[str, int]


def align(before: observer.Observation,
          after: observer.Observation) -> Alignment:
    """Matches elements in tiers: (1) a page key unique on both sides,
    (2) the same browser node within the same document, (3) the same role,
    name and matched parent, (4) the same role and name, unique among what
    is left (a moved element). Anything still shared by several candidates
    is ambiguous, never merged."""
    b_to_a = [-1] * len(before.elements)
    a_to_b = [-1] * len(after.elements)
    tiers: collections.Counter[str] = collections.Counter()

    def pair(i: int, j: int, tier: str) -> None:
        b_to_a[i], a_to_b[j] = j, i
        tiers[tier] += 1

    def free_b() -> Iterable[int]:
        return (i for i, j in enumerate(b_to_a) if j < 0)

    keys_b = collections.Counter(e.match_key for e in before.elements)
    keys_a = collections.Counter(e.match_key for e in after.elements)
    by_key = {e.match_key: j for j, e in enumerate(after.elements)}
    for i, element in enumerate(before.elements):
        page_key = element.match_key
        if page_key and keys_b[page_key] == 1 and keys_a[page_key] == 1:
            pair(i, by_key[page_key], 'page_key')
    if before.page_epoch == after.page_epoch:
        by_node = {e.node_id: j for j, e in enumerate(after.elements)}
        for i in list(free_b()):
            j = by_node.get(before.elements[i].node_id, -1)
            if j >= 0 > a_to_b[j] and (
                    after.elements[j].role == before.elements[i].role):
                pair(i, j, 'same_node')
    before_pos = {e.node_id: i for i, e in enumerate(before.elements)}
    after_pos = {e.node_id: j for j, e in enumerate(after.elements)}

    def structural(i: int) -> tuple[Any, ...]:
        element = before.elements[i]
        parent = _position(before_pos, element.parent_id)
        return (element.role, element.name,
                b_to_a[parent] if parent >= 0 else None)

    def structural_after(j: int) -> tuple[Any, ...]:
        element = after.elements[j]
        parent = _position(after_pos, element.parent_id)
        return (element.role, element.name, parent if parent >= 0 else None)

    progress = True
    while progress:
        progress = False
        groups_a: dict[tuple[Any, ...], list[int]] = (
            collections.defaultdict(list))
        for j, i in enumerate(a_to_b):
            if i < 0:
                groups_a[structural_after(j)].append(j)
        groups_b: dict[tuple[Any, ...], list[int]] = (
            collections.defaultdict(list))
        for i in free_b():
            groups_b[structural(i)].append(i)
        for group_key, candidates in groups_b.items():
            matches = groups_a.get(group_key, [])
            if len(candidates) == 1 and len(matches) == 1:
                pair(candidates[0], matches[0], 'structure')
                progress = True
    loose_a: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for j, i in enumerate(a_to_b):
        if i < 0:
            loose_a[(after.elements[j].role, after.elements[j].name)].append(j)
    loose_b: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for i in free_b():
        loose_b[(before.elements[i].role, before.elements[i].name)].append(i)
    ambiguous_b: set[int] = set()
    ambiguous_a: set[int] = set()
    for name_key, candidates in loose_b.items():
        others = loose_a.get(name_key, [])
        if len(candidates) == 1 and len(others) == 1:
            pair(candidates[0], others[0], 'moved')
        elif others:
            ambiguous_b.update(candidates)
            ambiguous_a.update(others)
    return Alignment(tuple(b_to_a), tuple(a_to_b), frozenset(ambiguous_b),
                     frozenset(ambiguous_a), dict(tiers))


# --- 3. Features and targets --------------------------------------------

@dataclasses.dataclass
class Vocab:
    """Token-to-id map; 0 is padding and 1 is unknown."""

    tokens: list[str] = dataclasses.field(
        default_factory=lambda: ['<pad>', '<unk>'])

    def __post_init__(self) -> None:
        self._index = {t: i for i, t in enumerate(self.tokens)}

    def add(self, token: str) -> None:
        """Adds a token if new (only while fitting on training data)."""
        if token not in self._index:
            self._index[token] = len(self.tokens)
            self.tokens.append(token)

    def encode(self, token: str) -> int:
        """Returns the token's id, or schema.UNK."""
        return self._index.get(token, schema.UNK)


def _position(positions: Mapping[int, int], node_id: int | None) -> int:
    """Returns the list position of a node id, or -1 for None/unknown."""
    return -1 if node_id is None else positions.get(node_id, -1)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _fixed(ids: Sequence[int], length: int) -> list[int]:
    return list(ids[:length]) + [schema.PAD] * max(0, length - len(ids))


@dataclasses.dataclass
class Vocabs:
    """Vocabularies fitted on the training split only."""

    roles: Vocab = dataclasses.field(default_factory=Vocab)
    words: Vocab = dataclasses.field(default_factory=Vocab)
    chars: Vocab = dataclasses.field(default_factory=Vocab)

    def fit(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Adds every role, word and character seen in the records."""
        for record in records:
            for moment in ('before', 'after'):
                for element in record[moment]['elements']:
                    self.roles.add(element['role'])
                    for word in _words(element['name']):
                        self.words.add(word)
                    for char in element['name'] + (element['value'] or ''):
                        self.chars.add(char)
            for char in record['action'].get('text') or '':
                self.chars.add(char)

    def text_chars(self, text: str | None) -> list[int]:
        """Returns fixed-length character ids for a string."""
        return _fixed([self.chars.encode(c) for c in (text or '')],
                      schema.MAX_CHARS)

    def name_words(self, text: str) -> list[int]:
        """Returns fixed-length word ids for a name."""
        return _fixed([self.words.encode(w) for w in _words(text)],
                      schema.MAX_NAME_WORDS)


def _viewport_box(box: observer.Box | None,
                  motion: observer.Motion) -> list[float] | None:
    if box is None:
        return None
    return [(box.x - motion.scroll_x) / motion.viewport_width,
            (box.y - motion.scroll_y) / motion.viewport_height,
            box.width / motion.viewport_width,
            box.height / motion.viewport_height]


def encode_state(observation: observer.Observation,
                 vocabs: Vocabs,
                 *,
                 input_encoding: str = 'legacy') -> dict[str, Any]:
    """Returns the allowed input features of one observation.

    No browser ids, page keys, tags, counters or temporal counts: only
    what the page shows, plus the element order and parent positions.
    """
    _validate_input_encoding(input_encoding)
    position = {e.node_id: i for i, e in enumerate(observation.elements)}
    motion = observation.motion
    elements = []
    for element in observation.elements:
        box = _viewport_box(element.box, motion)
        entry: dict[str, Any] = {
            'role': vocabs.roles.encode(element.role),
            'name_words': vocabs.name_words(element.name),
            'name_chars': vocabs.text_chars(element.name),
            'value_chars': vocabs.text_chars(element.value),
            'has_value': element.value is not None,
            'states': [int(bool(element.states.get(s))) for s in schema.STATE_NAMES],
            'flags': [int(element.visible), int(element.in_viewport),
                      int(element.obstructed), int(element.pinned)],
            'box': box or [0.0, 0.0, 0.0, 0.0],
            'box_mask': box is not None,
            'parent': _position(position, element.parent_id),
        }
        if input_encoding == 'utf8_bytes_v1':
            entry['name_bytes'] = byte_encoding.encode_bytes(element.name)
            entry['value_bytes'] = byte_encoding.encode_bytes(element.value)
        elements.append(entry)
    page = [motion.scroll_x / max(motion.content_width, 1.0),
            motion.scroll_y / max(motion.content_height, 1.0),
            motion.content_width / motion.viewport_width,
            motion.content_height / motion.viewport_height]
    return {'elements': elements, 'page': page}


def encode_action(record: Mapping[str, Any], vocabs: Vocabs,
                  motion: observer.Motion,
                  *,
                  input_encoding: str = 'legacy') -> dict[str, Any]:
    """Returns the action's input features, with the agent's own pointer
    position at the start of the action (known before acting)."""
    _validate_input_encoding(input_encoding)
    action, target = record['action'], record['target']
    start = record['result']['pointer_path'][0] if record['result'][
        'pointer_path'] else {'x': 0.0, 'y': 0.0}
    checked = action.get('checked')
    encoded: dict[str, Any] = {
        'kind': schema.ACTION_KINDS.index(action['kind']),
        'target': target['index'] if target else -1,
        'text_chars': vocabs.text_chars(action.get('text')),
        'checked': 0 if checked is None else 1 + int(checked),
        'scroll': (action.get('scroll_y') or 0.0) / motion.viewport_height,
        'pointer': [start['x'] / motion.viewport_width,
                    start['y'] / motion.viewport_height],
    }
    if input_encoding == 'utf8_bytes_v1':
        encoded['text_bytes'] = byte_encoding.encode_bytes(action.get('text'))
    return encoded


def _dropdown_index(observation: observer.Observation,
                    option_index: int) -> int:
    by_id = {e.node_id: i for i, e in enumerate(observation.elements)}
    index = _position(by_id, observation.elements[option_index].parent_id)
    while index >= 0 and observation.elements[index].role != 'combobox':
        index = _position(by_id, observation.elements[index].parent_id)
    return index


def _value_source(record: Mapping[str, Any], before: observer.Element,
                  after: observer.Element, index: int,
                  dropdown: int) -> int:
    action, target = record['action'], record['target']
    if after.value == before.value:
        return schema.VALUE_SOURCES.index('unchanged')
    if (action['kind'] == 'type' and target and target['index'] == index
            and after.value == action['text']):
        return schema.VALUE_SOURCES.index('typed')
    if (action['kind'] == 'select' and index == dropdown
            and after.value == target['name']):
        return schema.VALUE_SOURCES.index('option')
    if after.value == '':
        return schema.VALUE_SOURCES.index('cleared')
    return schema.VALUE_SOURCES.index('other')


def _element_changed(before: observer.Element, after: observer.Element) -> bool:
    if (before.value, before.name, dict(before.states), before.visible,
            before.in_viewport, before.obstructed) != (
                after.value, after.name, dict(after.states), after.visible,
                after.in_viewport, after.obstructed):
        return True
    if (before.box is None) != (after.box is None):
        return True
    if before.box is None or after.box is None:
        return False
    return any(abs(a - b) > BOX_TOLERANCE_PX for a, b in zip(
        dataclasses.astuple(before.box), dataclasses.astuple(after.box)))


def encode_targets(record: Mapping[str, Any], vocabs: Vocabs,
                   alignment: Alignment, before: observer.Observation,
                   after: observer.Observation) -> dict[str, Any]:
    """Returns supervised targets and masks, all derived from evidence."""
    dropdown = (_dropdown_index(before, record['target']['index'])
                if record['action']['kind'] == 'select' else -1)
    per_element = []
    for i, element in enumerate(before.elements):
        j = alignment.before_to_after[i]
        known = i not in alignment.ambiguous_before
        entry = {'known': known, 'removed': known and j < 0,
                 'changed': known and j < 0, 'matched': j >= 0,
                 'next_states': [0] * len(schema.STATE_NAMES),
                 'value_source': 0, 'value_mask': False,
                 'box_delta': [0.0] * 4, 'box_mask': False}
        if j >= 0:
            other = after.elements[j]
            entry['changed'] = _element_changed(element, other)
            entry['next_states'] = [int(bool(other.states.get(s)))
                                    for s in schema.STATE_NAMES]
            if element.value is not None or other.value is not None:
                entry['value_source'] = _value_source(record, element, other,
                                                      i, dropdown)
                entry['value_mask'] = True
            old = _viewport_box(element.box, before.motion)
            new = _viewport_box(other.box, after.motion)
            if old is not None and new is not None:
                entry['box_delta'] = [n - o for n, o in zip(new, old)]
                entry['box_mask'] = True
        per_element.append(entry)
    position = {e.node_id: j for j, e in enumerate(after.elements)}
    new_elements = []
    for j, element in enumerate(after.elements):
        if alignment.after_to_before[j] >= 0 or j in alignment.ambiguous_after:
            continue
        parent = _position(position, element.parent_id)
        new_elements.append({
            'role': vocabs.roles.encode(element.role),
            'name_words': vocabs.name_words(element.name),
            'parent_before': (alignment.after_to_before[parent]
                              if parent >= 0 else -1),
        })
    return {'elements': per_element, 'new': new_elements,
            'outcomes': _outcomes(record, alignment, before, after)}


def _outcomes(record: Mapping[str, Any], alignment: Alignment,
              before: observer.Observation,
              after: observer.Observation) -> list[int]:
    """Page-level outcomes from evidence: the page's own alerts and
    headings, scrolling, and backend truth deltas."""
    def is_alert(element: observer.Element) -> bool:
        return element.role == 'alert'

    created = any(is_alert(e) and alignment.after_to_before[j] < 0
                  and j not in alignment.ambiguous_after
                  for j, e in enumerate(after.elements))
    resolved = any(is_alert(e) and alignment.before_to_after[i] < 0
                   and i not in alignment.ambiguous_before
                   for i, e in enumerate(before.elements))
    persisting = any(is_alert(e) and alignment.before_to_after[i] >= 0
                     for i, e in enumerate(before.elements))
    headings = ({e.name for e in before.elements if e.role == 'heading'}
                != {e.name for e in after.elements if e.role == 'heading'})
    truth = record['truth']
    booked = len(truth['after']['bookings']) > len(
        truth['before']['bookings'])
    race = len(truth['after']['contests_resolved']) > len(
        truth['before']['contests_resolved']) and not booked
    scrolled = (before.motion.scroll_x, before.motion.scroll_y) != (
        after.motion.scroll_x, after.motion.scroll_y)
    unchanged = not recorder.differences(recorder.canonical(before),
                                         recorder.canonical(after))
    values = {'no_change': unchanged, 'alert_created': created,
              'alert_resolved': resolved, 'alert_persisting': persisting,
              'heading_changed': headings, 'booking_created': booked,
              'race_lost': race, 'scroll_changed': scrolled}
    return [int(values[name]) for name in schema.OUTCOMES]


def curate_record(record: Mapping[str, Any],
                  vocabs: Vocabs,
                  *,
                  input_encoding: str = 'legacy') -> dict[str, Any]:
    """Returns one curated example: metadata (never model input), inputs,
    targets and alignment statistics."""
    _validate_input_encoding(input_encoding)
    before = _observation(record['before'])
    after = _observation(record['after'])
    alignment = align(before, after)
    return {
        'meta': {'run_id': record['run_id'], 'step': record['step'],
                 'split': record['scenario']['split'],
                 'family': record['scenario']['family'],
                 'action_kind': record['action']['kind']},
        'inputs': {
            'state': encode_state(
                before, vocabs, input_encoding=input_encoding),
            'action': encode_action(
                record, vocabs, before.motion, input_encoding=input_encoding),
        },
        'targets': encode_targets(record, vocabs, alignment, before, after),
        'alignment': {'tiers': dict(alignment.tiers),
                      'ambiguous_before': len(alignment.ambiguous_before),
                      'ambiguous_after': len(alignment.ambiguous_after)},
    }


# --- Reports -----------------------------------------------------------

def label_coverage(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts what the targets contain, so gaps are visible before
    training. Alerts are split into appearing, persisting and resolving
    transitions rather than counted per state."""
    value_sources: collections.Counter[str] = collections.Counter()
    outcomes: collections.Counter[str] = collections.Counter()
    actions: collections.Counter[str] = collections.Counter()
    changed = removed = masked = new = elements = 0
    for example in examples:
        actions[example['meta']['action_kind']] += 1
        new += len(example['targets']['new'])
        for name, value in zip(schema.OUTCOMES, example['targets']['outcomes']):
            outcomes[name] += value
        for target in example['targets']['elements']:
            elements += 1
            if not target['known']:
                masked += 1
                continue
            changed += target['changed']
            removed += target['removed']
            if target['value_mask']:
                value_sources[schema.VALUE_SOURCES[target['value_source']]] += 1
    return {'examples': len(examples), 'actions': dict(actions),
            'elements': elements, 'changed_elements': changed,
            'removed_elements': removed, 'new_elements': new,
            'masked_ambiguous_elements': masked,
            'value_sources': dict(value_sources),
            'outcomes': dict(outcomes)}


def ambiguity(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarises how elements were aligned and what stayed ambiguous."""
    tiers: collections.Counter[str] = collections.Counter()
    ambiguous_examples = 0
    ambiguous_before = ambiguous_after = 0
    for example in examples:
        tiers.update(example['alignment']['tiers'])
        before = example['alignment']['ambiguous_before']
        after = example['alignment']['ambiguous_after']
        ambiguous_before += before
        ambiguous_after += after
        ambiguous_examples += bool(before or after)
    return {'pairs_by_tier': dict(tiers),
            'examples_with_ambiguity': ambiguous_examples,
            'ambiguous_before_elements': ambiguous_before,
            'ambiguous_after_elements': ambiguous_after}


# --- Curation ----------------------------------------------------------

class CurationError(Exception):
    """The raw dataset failed validation."""


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _working_tree_dirty() -> bool | None:
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], check=True,
                                capture_output=True, text=True)
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def curate(raw_dir: pathlib.Path, out_dir: pathlib.Path,
           *,
           input_encoding: str = 'legacy') -> dict[str, Any]:
    """Validates `raw_dir` and writes curated examples to `out_dir`.

    Returns:
      The curated manifest.

    Raises:
      CurationError: Validation found fatal problems.
      ValueError: `input_encoding` is unsupported or `raw_dir` and `out_dir`
        overlap.
      NotADirectoryError: `out_dir` exists and is not a directory.
      FileExistsError: `out_dir` already exists and is not empty.
    """
    _validate_input_encoding(input_encoding)
    raw_path = raw_dir.resolve()
    output_path = out_dir.resolve()
    if (
        output_path.is_relative_to(raw_path)
        or raw_path.is_relative_to(output_path)
    ):
        raise ValueError(
            'Raw and curated dataset directories must not overlap.'
        )
    if out_dir.exists():
        if not out_dir.is_dir():
            raise NotADirectoryError(
                f'Destination is not a directory: {out_dir}'
            )
        if any(out_dir.iterdir()):
            raise FileExistsError(
                f'Destination directory is not empty: {out_dir}'
            )
    report = validate(raw_dir)
    if report.problems:
        raise CurationError('\n'.join(report.problems[:20]))
    raw_manifest, records = load(raw_dir)
    vocabs = Vocabs()
    vocabs.fit(r for r in records['train']
               if f'{r["run_id"]}#{r["step"]}' not in report.exclusions)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    coverage = {}
    alignment = {}
    for split in SPLITS:
        kept = [r for r in records[split]
                if f'{r["run_id"]}#{r["step"]}' not in report.exclusions]
        kept.sort(key=lambda r: (r['run_id'], r['step']))
        examples = [curate_record(r, vocabs, input_encoding=input_encoding)
                    for r in kept]
        with (out_dir / f'{split}.jsonl').open(
                'w', encoding='utf-8', newline='\n') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        counts[split] = len(examples)
        coverage[split] = label_coverage(examples)
        alignment[split] = ambiguity(examples)
    vocab_path = out_dir / 'vocab.json'
    with vocab_path.open('w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps({
            'roles': vocabs.roles.tokens, 'words': vocabs.words.tokens,
            'chars': vocabs.chars.tokens}))
    source_files = {
        p.relative_to(raw_dir).as_posix(): _sha256(p)
        for p in sorted(raw_dir.rglob('*.json*'))
    }
    config: dict[str, Any] = {
        'state_names': list(schema.STATE_NAMES),
        'action_kinds': list(schema.ACTION_KINDS),
        'value_sources': list(schema.VALUE_SOURCES),
        'outcomes': list(schema.OUTCOMES),
        'max_name_words': schema.MAX_NAME_WORDS,
        'max_chars': schema.MAX_CHARS,
        'box_tolerance_px': BOX_TOLERANCE_PX,
        'vocab_fitted_on': 'train only',
    }
    manifest: dict[str, Any] = {
        'schema_version': SUPPORTED_SCHEMA,
        'source_files': source_files,
        'config': config,
        'vocab_sizes': {'roles': len(vocabs.roles.tokens),
                        'words': len(vocabs.words.tokens),
                        'chars': len(vocabs.chars.tokens)},
        'examples': counts,
        'exclusions': report.exclusions,
        'label_coverage': coverage,
        'alignment': alignment,
        'curated_files': {f'{split}.jsonl': _sha256(out_dir / f'{split}.jsonl')
                          for split in SPLITS},
    }
    if input_encoding == 'utf8_bytes_v1':
        config.update({
            'input_encoding': input_encoding,
            'byte_padding_id': byte_encoding.BYTE_PAD_ID,
            'byte_vocab_size': byte_encoding.BYTE_VOCAB_SIZE,
            'byte_truncation': False,
        })
        manifest.update({
            'input_encoding': input_encoding,
            'byte_padding_id': byte_encoding.BYTE_PAD_ID,
            'byte_vocab_size': byte_encoding.BYTE_VOCAB_SIZE,
            'curation_commit': _source_commit(),
            'curation_dirty': _working_tree_dirty(),
            'source_dataset': {
                'raw_dir': raw_dir.as_posix(),
                'schema_version': (raw_manifest.get('schema_version')
                                   if raw_manifest else None),
                'splits': (raw_manifest.get('splits')
                           if raw_manifest else None),
                'episodes': (len(raw_manifest.get('episodes', []))
                             if raw_manifest else 0),
                'manifest_sha256': source_files.get('manifest.json'),
            },
            'vocab_sha256': _sha256(vocab_path),
        })
    else:
        manifest['source_commit'] = _source_commit()
    with (out_dir / 'manifest.json').open(
            'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--raw', default='data/raw/e04/main')
    parser.add_argument('--out', default='data/curated/c01')
    parser.add_argument('--input-encoding', choices=INPUT_ENCODINGS,
                        default='legacy')
    args = parser.parse_args()
    manifest = curate(pathlib.Path(args.raw), pathlib.Path(args.out),
                      input_encoding=args.input_encoding)
    print(json.dumps({k: manifest[k] for k in
                      ('examples', 'exclusions', 'vocab_sizes')}, indent=1))


if __name__ == '__main__':
    main()
