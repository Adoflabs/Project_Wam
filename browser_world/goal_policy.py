"""A goal-directed decider for booking scenarios.

It knows the goal (what to book, which details to enter) and the page's
wording, which is privileged decider knowledge, never model input. At each
step it chooses from `available_actions` only: the first unfinished goal
field the page currently offers, otherwise Next, otherwise Confirm,
otherwise a scroll. So it follows any layout's order without being told it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses

from browser_world import booking
from browser_world import executor
from browser_world import observer
from browser_world import recorder
from browser_world import scenarios

DONE = 'booked'
CONFLICT = 'conflict'
VALIDATION_ERROR = 'validation_error'
_MAX_STEPS = 40
_TARGET_ROLES = {
    executor.SELECT: ('option',),
    executor.CHECK: ('checkbox', 'radio'),
    executor.TYPE: ('textbox', 'searchbox'),
}


@dataclasses.dataclass(frozen=True)
class _Intent:
    kind: str
    name: str
    text: str | None = None
    checked: bool | None = None


def intents(goal: scenarios.Goal,
            text: Mapping[str, str]) -> tuple[_Intent, ...]:
    """Returns the goal's field actions, in a fixed preference order."""
    return (
        _Intent(executor.SELECT, goal.service),
        _Intent(executor.SELECT, goal.doctor),
        _Intent(executor.SELECT, goal.date),
        _Intent(executor.CHECK, goal.slot, checked=True),
        _Intent(executor.TYPE, text['name'], goal.contact.name),
        _Intent(executor.TYPE, text['email'], goal.contact.email),
        _Intent(executor.TYPE, text['phone'], goal.contact.phone),
        _Intent(executor.CHECK, text['consent'], checked=True),
    )


def outcome_of(observation: observer.Observation,
               text: Mapping[str, str]) -> str | None:
    """Returns how the episode ended, or None while it is still going."""
    names = {(e.role, e.name) for e in observation.elements}
    if ('status', text['done']) in names:
        return DONE
    if ('alert', booking.MSG_SLOT_TAKEN) in names:
        return CONFLICT
    if ('alert', booking.MSG_EMAIL_INVALID) in names:
        return VALIDATION_ERROR
    return None


def _matching(offers: Sequence[executor.Affordance],
              intent: _Intent) -> executor.Affordance | None:
    for offer in offers:
        if (offer.kind == intent.kind and offer.name == intent.name
                and (intent.checked is None
                     or offer.checked == intent.checked)):
            return offer
    return None


def _pending_box(observation: observer.Observation,
                 todo: Sequence[_Intent]) -> observer.Box | None:
    """Returns where an unfinished goal field sits when it is on the page
    but not offered (off-screen). Options have no box, so their dropdown's
    box is used."""
    by_id = {e.node_id: e for e in observation.elements}
    for intent in todo:
        for element in observation.elements:
            if (element.name != intent.name
                    or element.role not in _TARGET_ROLES[intent.kind]):
                continue
            node: observer.Element | None = element
            while node is not None and node.box is None:
                node = by_id.get(node.parent_id) if node.parent_id else None
            if node is not None and node.box is not None:
                return node.box
    return None


def choose(observation: observer.Observation, todo: Sequence[_Intent],
           text: Mapping[str, str]
           ) -> tuple[executor.Action, _Intent | None] | None:
    """Returns the next action and the intent it fulfils (None for Next,
    Confirm or scrolling), or None if nothing useful is offered."""
    offers = executor.available_actions(observation)
    for intent in todo:
        offer = _matching(offers, intent)
        if offer is not None:
            return offer.action(intent.text), intent
    box = _pending_box(observation, todo)
    if box is not None:  # Reach unfinished fields before moving on.
        direction = ('scroll up' if box.y < observation.motion.scroll_y
                     else 'scroll down')
        offer = _matching(offers, _Intent(executor.SCROLL, direction))
        if offer is not None:
            return offer.action(), None
    for label in (text['next'], text['confirm']):
        offer = _matching(offers, _Intent(executor.CLICK, label))
        if offer is not None:
            return offer.action(), None
    for direction in ('scroll down', 'scroll up'):
        offer = _matching(offers, _Intent(executor.SCROLL, direction))
        if offer is not None:
            return offer.action(), None
    return None


def run_episode(watcher: observer.Observer, rec: recorder.Recorder,
                scenario: booking.Scenario, goal: scenarios.Goal,
                metadata: Mapping[str, object]
                ) -> tuple[str, list[recorder.Transition]]:
    """Pursues `goal` in one recorded run until the episode ends.

    Returns:
      A tuple (outcome, transitions), where outcome is DONE, CONFLICT or
      VALIDATION_ERROR.

    Raises:
      RuntimeError: The decider got stuck or the episode ran too long.
    """
    text = scenario.presentation.text
    todo = list(intents(goal, text))
    run_id = rec.new_run(f'episode-{scenario.scenario_id}-{goal.outcome}',
                         'start', scenario=metadata)
    transitions: list[recorder.Transition] = []
    for _ in range(_MAX_STEPS):
        watcher.wait_until_settled()
        observation = watcher.observe()
        ended = outcome_of(observation, text)
        if ended is not None:
            return ended, transitions
        choice = choose(observation, todo, text)
        if choice is None:
            raise RuntimeError('Nothing useful is offered.')
        action, intent = choice
        outcome = rec.record_step(run_id, observation, action)
        if not isinstance(outcome, recorder.Transition):
            raise RuntimeError(f'Refused: {outcome.refusal}: {outcome.detail}')
        transitions.append(outcome)
        if intent is not None:
            todo.remove(intent)
    raise RuntimeError('Episode did not end.')
