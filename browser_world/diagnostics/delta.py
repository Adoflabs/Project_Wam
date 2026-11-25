"""Computes what changed between two observations.

The delta is the world model's prediction target: given the state before an
action and the action, predict this delta. Elements are matched across the
two observations by `match_key` (falling back to role and name), because
Chromium node ids change whenever an element is re-rendered.

Typical usage example:

  change = delta.diff(before, after)
  change['added'], change['removed'], change['changed']
"""

from __future__ import annotations

from typing import Any

from browser_world.interaction import observer

# Boxes that move less than this many CSS pixels count as unchanged.
_MOVE_TOLERANCE_PX = 1.0


def element_key(element: observer.Element) -> str:
    """Returns the identity used to match an element across observations."""
    if element.match_key:
        return element.match_key
    return f'{element.role}:{element.name}'


def _summary(element: observer.Element) -> dict[str, Any]:
    return {'key': element_key(element), 'role': element.role,
            'name': element.name}


def _box_changed(old: observer.Box | None, new: observer.Box | None) -> bool:
    if old is None or new is None:
        return old is not new
    return any(abs(a - b) > _MOVE_TOLERANCE_PX for a, b in (
        (old.x, new.x), (old.y, new.y),
        (old.width, new.width), (old.height, new.height)))


def _changed_fields(old: observer.Element,
                    new: observer.Element) -> dict[str, list[Any]]:
    """Returns {field: [old, new]} for every field that differs."""
    fields: dict[str, list[Any]] = {}
    for name in ('name', 'value', 'in_viewport', 'hit_testable'):
        if getattr(old, name) != getattr(new, name):
            fields[name] = [getattr(old, name), getattr(new, name)]
    for state in observer.STATE_NAMES:
        if old.states.get(state) != new.states.get(state):
            fields[state] = [old.states.get(state), new.states.get(state)]
    if _box_changed(old.box, new.box):
        fields['box'] = [_box_list(old.box), _box_list(new.box)]
    return fields


def _box_list(box: observer.Box | None) -> list[float] | None:
    if box is None:
        return None
    return [round(box.x), round(box.y), round(box.width), round(box.height)]


def _index(observation: observer.Observation) -> dict[str, observer.Element]:
    """Maps keys to elements, keeping the first of any duplicate keys."""
    index: dict[str, observer.Element] = {}
    for element in observation.elements:
        index.setdefault(element_key(element), element)
    return index


def diff(before: observer.Observation,
         after: observer.Observation) -> dict[str, Any]:
    """Returns the changes from `before` to `after`.

    Args:
      before: The earlier observation.
      after: The later observation.

    Returns:
      A JSON-ready dict with:
        url_changed: Whether the page URL changed.
        added: Summaries of elements only in `after`.
        removed: Summaries of elements only in `before`.
        changed: Summaries of matched elements plus a `fields` mapping of
          field name to [old, new].
    """
    old = _index(before)
    new = _index(after)
    changed = []
    for key, element in new.items():
        if key not in old:
            continue
        fields = _changed_fields(old[key], element)
        if fields:
            changed.append({**_summary(element), 'fields': fields})
    return {
        'url_changed': before.url != after.url,
        'added': [_summary(e) for k, e in new.items() if k not in old],
        'removed': [_summary(e) for k, e in old.items() if k not in new],
        'changed': changed,
    }
