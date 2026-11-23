"""Performs a decided action on an observed element with real input events.

The executor never decides what to do. A decider (a test, a data collector,
later the world model) picks an action from `available_actions(observation)`
and supplies any text. The executor checks that the action is still safe to
perform against the current page, moves the mouse along a path to the
target, re-checks the target, and then clicks or types with Playwright's
mouse and keyboard. It never uses CSS selectors, page-specific knowledge or
backend state.

Safety contract for data collection (E0.3a):
  * Each observation may be used for one executed action. After acting, the
    page must be observed again before the next action.
  * The pre-press checks and the press are not atomic: the page can still
    change between them. The checks only shrink that window.

Typical usage example:

  watcher = observer.attach(page)
  hand = executor.Executor(page, watcher)
  observation = watcher.observe()
  choice = available_actions(observation)[0]   # The decider's choice.
  result = hand.execute(observation, executor.Action(choice.kind,
                                                     choice.target))
"""

from __future__ import annotations

import dataclasses
import enum
import time
from typing import Any

from playwright import sync_api

from browser_world import observer

CLICK = 'click'
TYPE = 'type'
_ROLES_FOR_ACTION = {
    CLICK: frozenset({'button'}),
    TYPE: frozenset({'searchbox', 'textbox'}),
}


class Refusal(enum.StrEnum):
    """Why the executor refused to perform an action."""

    OBSERVATION_USED = 'observation_used'
    BAD_ACTION = 'bad_action'
    UNKNOWN_TARGET = 'unknown_target'
    UNSUPPORTED_ROLE = 'unsupported_role'
    DISABLED = 'disabled'
    NOT_VISIBLE = 'not_visible'
    NOT_IN_VIEWPORT = 'not_in_viewport'
    OBSTRUCTED = 'obstructed'
    STALE_OBSERVATION = 'stale_observation'


@dataclasses.dataclass(frozen=True)
class Affordance:
    """Something that can be done to an observed element right now.

    It says nothing about whether doing it will succeed (for example,
    whether some text would pass validation); that is for the world model
    to learn.

    Attributes:
      kind: `CLICK` or `TYPE`.
      target: The element's `node_id` in the observation.
      role: The element's role, for the decider's convenience.
      name: The element's accessible name, for the decider's convenience.
    """

    kind: str
    target: int
    role: str
    name: str


@dataclasses.dataclass(frozen=True)
class Action:
    """A decided action.

    Attributes:
      kind: `CLICK` or `TYPE`.
      target: `node_id` of the target element in the observation acted on.
      text: Text that replaces the field's contents; required for `TYPE`.
    """

    kind: str
    target: int
    text: str | None = None


@dataclasses.dataclass(frozen=True)
class Speed:
    """How fast the executor moves the mouse and types.

    Attributes:
      move_steps: Mouse-move events sent along each path.
      move_ms: Total time spent travelling along a path.
      key_delay_ms: Pause between typed characters.
    """

    move_steps: int
    move_ms: float
    key_delay_ms: float


NORMAL_SPEED = Speed(move_steps=20, move_ms=250.0, key_delay_ms=35.0)
FAST_SPEED = Speed(move_steps=4, move_ms=0.0, key_delay_ms=0.0)


@dataclasses.dataclass(frozen=True)
class PathPoint:
    """One mouse position sent to the page, in viewport CSS pixels.

    Attributes:
      x: Viewport x coordinate.
      y: Viewport y coordinate.
      t_ms: Time since the action started.
    """

    x: float
    y: float
    t_ms: float


@dataclasses.dataclass(frozen=True)
class ActionResult:
    """What the executor did. It holds no post-action page state: only a
    new observation can say what the action caused.

    Attributes:
      action: The requested action.
      executed: Whether the press (and typing) happened.
      refusal: Why it did not happen, or None.
      detail: Human-readable explanation of a refusal.
      observation_sequence: Sequence number of the observation acted on.
      pointer_path: Mouse positions sent, starting at the previous position.
      approach_changed_page: Whether the DOM or scroll changed while the
        mouse travelled (for example, a hover effect).
      started_at_ms: Wall-clock start time, in milliseconds.
      duration_ms: Total time taken.
    """

    action: Action
    executed: bool
    refusal: Refusal | None
    detail: str
    observation_sequence: int
    pointer_path: tuple[PathPoint, ...]
    approach_changed_page: bool
    started_at_ms: float
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-ready dict."""
        return dataclasses.asdict(self)


def _element_problem(element: observer.Element,
                     kind: str) -> tuple[Refusal, str] | None:
    """Returns why `kind` cannot be done to `element`, judged from the
    observation alone, or None if it can."""
    if element.role not in _ROLES_FOR_ACTION[kind]:
        return Refusal.UNSUPPORTED_ROLE, f'Cannot {kind} a {element.role}.'
    if element.states.get('disabled'):
        return Refusal.DISABLED, 'The target is disabled.'
    if not element.visible:
        return Refusal.NOT_VISIBLE, 'The target is not rendered visibly.'
    if not element.in_viewport or element.hit_point is None:
        return Refusal.NOT_IN_VIEWPORT, 'The target is outside the viewport.'
    if element.obstructed:
        return Refusal.OBSTRUCTED, 'Another element covers the target.'
    return None


def available_actions(
        observation: observer.Observation) -> list[Affordance]:
    """Lists the actions that can be performed on observed elements now.

    Args:
      observation: The observation to act on.

    Returns:
      One affordance per element and supported action kind, in document
      order. Typing affordances carry no text; the decider supplies it.
    """
    affordances = []
    for element in observation.elements:
        for kind, roles in _ROLES_FOR_ACTION.items():
            if (element.role in roles
                    and _element_problem(element, kind) is None):
                affordances.append(Affordance(kind, element.node_id,
                                              element.role, element.name))
    return affordances


class Executor:
    """Performs actions on one page, keeping track of the mouse position.

    Attributes:
      pointer: Current mouse position in viewport coordinates. Playwright's
        mouse starts at the top-left corner.
    """

    def __init__(self, page: sync_api.Page, watcher: observer.Observer,
                 speed: Speed = NORMAL_SPEED):
        """Creates an executor for `page`.

        Args:
          page: The page to act on.
          watcher: The observer attached to the same page.
          speed: Mouse and typing speed.
        """
        self._page = page
        self._watcher = watcher
        self._speed = speed
        self._used_observations: set[int] = set()
        self.pointer = (0.0, 0.0)

    def execute(self, observation: observer.Observation,
                action: Action) -> ActionResult:
        """Performs `action`, or refuses with a reason.

        Args:
          observation: The latest observation of the page; `action.target`
            must refer to one of its elements.
          action: What the decider chose.

        Returns:
          What was done, including the mouse path and timing.
        """
        finish = _ResultBuilder(action, observation.sequence,
                                time.perf_counter(), time.time() * 1000)
        try:
            element = self._check_before(observation, action)
        except _RefusedError as e:
            return finish.refused(e.refusal, e.detail)
        self._used_observations.add(observation.sequence)

        try:
            doc_x, doc_y = _press_point(element)
        except _RefusedError as e:
            return finish.refused(e.refusal, e.detail)
        path = self._move_to(doc_x - observation.motion.scroll_x,
                             doc_y - observation.motion.scroll_y,
                             finish.started)
        changed = not self._watcher.is_current(observation)
        try:
            self._check_at_press(element.node_id, doc_x, doc_y)
        except _RefusedError as e:
            return finish.refused(e.refusal, e.detail, path, changed)

        self._page.mouse.down()
        self._page.mouse.up()
        if action.kind == TYPE:
            self._replace_text(action.text or '')
        return finish.done(path, changed)

    def _check_before(self, observation: observer.Observation,
                      action: Action) -> observer.Element:
        """Checks the action against the observation and the live page.

        Returns:
          The target element.

        Raises:
          _RefusedError: The action must not be performed.
        """
        if observation.sequence in self._used_observations:
            raise _RefusedError(
                Refusal.OBSERVATION_USED,
                'This observation was already acted on; observe again.')
        if action.kind not in _ROLES_FOR_ACTION:
            raise _RefusedError(Refusal.BAD_ACTION,
                                f'Unknown action kind {action.kind!r}.')
        if action.kind == TYPE and action.text is None:
            raise _RefusedError(Refusal.BAD_ACTION, 'Typing needs text.')
        matches = [e for e in observation.elements
                   if e.node_id == action.target]
        if len(matches) != 1:
            raise _RefusedError(
                Refusal.UNKNOWN_TARGET,
                f'{len(matches)} observed elements have node_id '
                f'{action.target}.')
        problem = _element_problem(matches[0], action.kind)
        if problem is not None:
            raise _RefusedError(*problem)
        if not self._watcher.is_current(observation):
            raise _RefusedError(
                Refusal.STALE_OBSERVATION,
                'The page changed since the observation; observe again.')
        return matches[0]

    def _check_at_press(self, node_id: int, doc_x: float,
                        doc_y: float) -> None:
        """Re-checks the live target with the mouse already over it.

        Raises:
          _RefusedError: The target is gone, covered or disabled.
        """
        reaches = self._watcher.hits_node(doc_x, doc_y, node_id)
        if reaches is None:
            raise _RefusedError(Refusal.STALE_OBSERVATION,
                                'The target no longer exists.')
        if not reaches:
            raise _RefusedError(
                Refusal.OBSTRUCTED,
                'Something now covers the target at the press point.')
        if not self._watcher.is_enabled(node_id):
            raise _RefusedError(Refusal.DISABLED,
                                'The target became disabled.')

    def _move_to(self, x: float, y: float,
                 started: float) -> tuple[PathPoint, ...]:
        """Moves the mouse in a straight line to viewport point (x, y),
        sending one mouse-move event per step, and records the path."""
        start_x, start_y = self.pointer
        steps = self._speed.move_steps
        pause_ms = self._speed.move_ms / steps
        points = [PathPoint(start_x, start_y, _elapsed_ms(started))]
        for step in range(1, steps + 1):
            fraction = step / steps
            point_x = start_x + (x - start_x) * fraction
            point_y = start_y + (y - start_y) * fraction
            self._page.mouse.move(point_x, point_y)
            points.append(PathPoint(point_x, point_y, _elapsed_ms(started)))
            if pause_ms:
                self._page.wait_for_timeout(pause_ms)
        self.pointer = (x, y)
        return tuple(points)

    def _replace_text(self, text: str) -> None:
        """Replaces the focused field's contents, one key at a time."""
        keyboard = self._page.keyboard
        keyboard.press('ControlOrMeta+A')
        keyboard.press('Backspace')
        keyboard.type(text, delay=self._speed.key_delay_ms)


class _RefusedError(Exception):
    """An action must not be performed.

    Attributes:
      refusal: The reason code.
      detail: Human-readable explanation.
    """

    def __init__(self, refusal: Refusal, detail: str):
        super().__init__(f'{refusal}: {detail}')
        self.refusal = refusal
        self.detail = detail


def _press_point(element: observer.Element) -> tuple[float, float]:
    """Returns the document point to press: the point the observer
    hit-tested, at the centre of the element's visible part.

    Raises:
      _RefusedError: The observer did not hit-test the element (cannot
        happen after the viewport check, but the point is optional).
    """
    if element.hit_point is None:
        raise _RefusedError(Refusal.NOT_IN_VIEWPORT,
                            'The target was not hit-tested.')
    return element.hit_point


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


@dataclasses.dataclass(frozen=True)
class _ResultBuilder:
    """Builds `ActionResult`s that share one action's identity and timing."""

    action: Action
    sequence: int
    started: float
    started_at_ms: float

    def refused(self, refusal: Refusal, detail: str,
                path: tuple[PathPoint, ...] = (),
                changed: bool = False) -> ActionResult:
        """Returns a result for an action that was not performed."""
        return self._build(refusal, detail, path, changed)

    def done(self, path: tuple[PathPoint, ...],
             changed: bool) -> ActionResult:
        """Returns a result for a performed action."""
        return self._build(None, '', path, changed)

    def _build(self, refusal: Refusal | None, detail: str,
               path: tuple[PathPoint, ...], changed: bool) -> ActionResult:
        return ActionResult(
            action=self.action,
            executed=refusal is None,
            refusal=refusal,
            detail=detail,
            observation_sequence=self.sequence,
            pointer_path=path,
            approach_changed_page=changed,
            started_at_ms=self.started_at_ms,
            duration_ms=_elapsed_ms(self.started),
        )
