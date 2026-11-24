"""Performs a decided action on the page with real input events.

The executor never decides what to do. A decider (a test, a data collector,
later the world model) picks an action from `available_actions(observation)`
and supplies any text. The executor checks that the action is still safe to
perform against the current page, moves the mouse along a path to the
target, re-checks the target, and then clicks or types with Playwright's
mouse and keyboard. It never uses CSS selectors, page-specific knowledge or
backend state.

Action kinds:
  * click: a button.
  * type: replace a text field's contents.
  * select: an option of a native dropdown; the target is the option
    element, and the executor opens the dropdown and picks it by keyboard.
  * check: set a checkbox or radio button to an explicit state.
  * scroll: scroll the page with the mouse wheel; no element target.

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
import math
import time
from typing import Any

from playwright import sync_api

from browser_world import observer

CLICK = 'click'
TYPE = 'type'
SELECT = 'select'
CHECK = 'check'
SCROLL = 'scroll'
# Roles each element-targeted action accepts. SELECT targets an option,
# whose dropdown is the control actually pressed.
_ROLES_FOR_ACTION = {
    CLICK: frozenset({'button'}),
    TYPE: frozenset({'searchbox', 'textbox'}),
    SELECT: frozenset({'option'}),
    CHECK: frozenset({'checkbox', 'radio'}),
}
ACTION_KINDS = frozenset(_ROLES_FOR_ACTION) | {SCROLL}
_DROPDOWN_ROLE = 'combobox'
# Default distance offered by scroll affordances, as a share of the
# viewport height.
_SCROLL_FRACTION = 0.8


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
    ALREADY_IN_STATE = 'already_in_state'


@dataclasses.dataclass(frozen=True)
class Affordance:
    """Something that can be done to an observed element right now.

    It says nothing about whether doing it will succeed (for example,
    whether some text would pass validation); that is for the world model
    to learn.

    Attributes:
      kind: One of `ACTION_KINDS`.
      target: The element's `node_id` in the observation; None for SCROLL.
      role: The element's role ('page' for SCROLL), for the decider.
      name: The element's accessible name, for the decider.
      checked: For CHECK, the state the action would set.
      scroll_y: For SCROLL, a suggested distance in CSS pixels.
    """

    kind: str
    target: int | None
    role: str
    name: str
    checked: bool | None = None
    scroll_y: float | None = None

    def action(self, text: str | None = None) -> 'Action':
        """Returns this affordance as an action; TYPE needs `text`."""
        return Action(self.kind, self.target, text=text, checked=self.checked,
                      scroll_y=self.scroll_y)


@dataclasses.dataclass(frozen=True)
class Action:
    """A decided action.

    Attributes:
      kind: One of `ACTION_KINDS`.
      target: `node_id` of the target element in the observation acted on
        (for SELECT, the option to pick); must be None for SCROLL.
      text: Text that replaces the field's contents; required for TYPE.
      checked: The state to set; required for CHECK.
      scroll_y: Pixels to scroll the page, positive is down; required for
        SCROLL.
    """

    kind: str
    target: int | None = None
    text: str | None = None
    checked: bool | None = None
    scroll_y: float | None = None


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


def _element_problem(element: observer.Element, kind: str,
                     role_checked: bool = False
                     ) -> tuple[Refusal, str] | None:
    """Returns why `kind` cannot be done to the pressed `element`, judged
    from the observation alone, or None if it can. `role_checked` skips
    the role test for controls reached through another element."""
    if not role_checked and element.role not in _ROLES_FOR_ACTION[kind]:
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


def _dropdown_of(observation: observer.Observation,
                 option: observer.Element) -> observer.Element | None:
    """Returns the native dropdown an option belongs to, or None. Options
    inside an <optgroup> have the group as their parent, so the search
    walks up the parent chain."""
    by_id = {e.node_id: e for e in observation.elements}
    parent = by_id.get(option.parent_id) if option.parent_id else None
    while parent is not None:
        if parent.role == _DROPDOWN_ROLE:
            return parent
        parent = by_id.get(parent.parent_id) if parent.parent_id else None
    return None


def _options_of(observation: observer.Observation,
                dropdown: observer.Element) -> list[observer.Element]:
    """Returns a dropdown's options in document order."""
    return [e for e in observation.elements
            if e.role == 'option' and _dropdown_of(observation, e) is dropdown]


def _control_problem(observation: observer.Observation,
                     element: observer.Element,
                     kind: str) -> tuple[Refusal, str] | None:
    """Returns why `kind` cannot be done to `element`, judged from the
    observation alone, or None if it can. For SELECT the option's
    dropdown is the control that must be usable."""
    if element.role not in _ROLES_FOR_ACTION[kind]:
        return Refusal.UNSUPPORTED_ROLE, f'Cannot {kind} a {element.role}.'
    if kind != SELECT:
        return _element_problem(element, kind)
    dropdown = _dropdown_of(observation, element)
    if dropdown is None:
        return (Refusal.UNSUPPORTED_ROLE,
                'The option is not inside a native dropdown.')
    if element.states.get('disabled'):
        return Refusal.DISABLED, 'The option is disabled.'
    return _element_problem(dropdown, CLICK, role_checked=True)


def _scroll_affordances(
        observation: observer.Observation) -> list[Affordance]:
    """Offers scrolling the page down and up when there is room."""
    motion = observation.motion
    step = round(motion.viewport_height * _SCROLL_FRACTION)
    affordances = []
    if motion.scroll_y + motion.viewport_height < motion.content_height:
        affordances.append(Affordance(SCROLL, None, 'page', 'scroll down',
                                      scroll_y=float(step)))
    if motion.scroll_y > 0:
        affordances.append(Affordance(SCROLL, None, 'page', 'scroll up',
                                      scroll_y=float(-step)))
    return affordances


def available_actions(
        observation: observer.Observation) -> list[Affordance]:
    """Lists the actions that can be performed now.

    Args:
      observation: The observation to act on.

    Returns:
      Affordances for observed elements in document order, then page
      scrolling. Typing affordances carry no text; the decider supplies it.
      CHECK affordances set the opposite of the current state (radio
      buttons can only be set).
    """
    affordances = []
    for element in observation.elements:
        for kind, roles in _ROLES_FOR_ACTION.items():
            if element.role not in roles:
                continue
            if _control_problem(observation, element, kind) is not None:
                continue
            checked = None
            if kind == CHECK:
                checked = not element.states.get('checked')
                if element.role == 'radio' and not checked:
                    continue
            affordances.append(Affordance(kind, element.node_id,
                                          element.role, element.name,
                                          checked=checked))
    return affordances + _scroll_affordances(observation)


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
          observation: The latest observation of the page; an element
            target must refer to one of its elements.
          action: What the decider chose.

        Returns:
          What was done, including the mouse path and timing.
        """
        finish = _ResultBuilder(action, observation.sequence,
                                time.perf_counter(), time.time() * 1000)
        try:
            plan = self._plan(observation, action)
        except _RefusedError as e:
            return finish.refused(e.refusal, e.detail)
        self._used_observations.add(observation.sequence)
        if plan.control is None:
            return self._scroll(action.scroll_y or 0.0, finish)

        try:
            doc_x, doc_y = _press_point(plan.control)
        except _RefusedError as e:
            return finish.refused(e.refusal, e.detail)
        path = self._move_to(doc_x - observation.motion.scroll_x,
                             doc_y - observation.motion.scroll_y,
                             finish.started)
        changed = not self._watcher.is_current(observation)
        try:
            self._check_at_press(plan.control.node_id, doc_x, doc_y)
        except _RefusedError as e:
            return finish.refused(e.refusal, e.detail, path, changed)

        self._page.mouse.down()
        self._page.mouse.up()
        if action.kind == TYPE:
            self._replace_text(action.text or '')
        elif action.kind == SELECT:
            self._pick_option(plan.option_steps)
        return finish.done(path, changed)

    def _plan(self, observation: observer.Observation,
              action: Action) -> '_Plan':
        """Checks the action against the observation and the live page.

        Returns:
          What to press, and for SELECT how far down the dropdown to go.

        Raises:
          _RefusedError: The action must not be performed.
        """
        if observation.sequence in self._used_observations:
            raise _RefusedError(
                Refusal.OBSERVATION_USED,
                'This observation was already acted on; observe again.')
        _check_fields(action)
        plan = _Plan(control=None, option_steps=0)
        if action.kind != SCROLL:
            plan = _element_plan(observation, action)
        if not self._watcher.is_current(observation):
            raise _RefusedError(
                Refusal.STALE_OBSERVATION,
                'The page changed since the observation; observe again.')
        return plan

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

    def _scroll(self, scroll_y: float,
                finish: '_ResultBuilder') -> ActionResult:
        """Scrolls with the mouse wheel at the current mouse position.

        The page scrolls unless the mouse is over a scrollable element,
        which then scrolls instead; the after-observation records which.
        """
        point = PathPoint(self.pointer[0], self.pointer[1],
                          _elapsed_ms(finish.started))
        self._page.mouse.wheel(0, scroll_y)
        return finish.done((point,), changed=False)

    def _pick_option(self, steps: int) -> None:
        """Picks an option in the dropdown the press just opened: Home goes
        to the first enabled option, each ArrowDown to the next enabled
        one (disabled options are skipped), and Enter commits once."""
        keyboard = self._page.keyboard
        keyboard.press('Home')
        for _ in range(steps):
            keyboard.press('ArrowDown')
        keyboard.press('Enter')

    def _replace_text(self, text: str) -> None:
        """Replaces the focused field's contents, one key at a time."""
        keyboard = self._page.keyboard
        keyboard.press('ControlOrMeta+A')
        keyboard.press('Backspace')
        keyboard.type(text, delay=self._speed.key_delay_ms)


@dataclasses.dataclass(frozen=True)
class _Plan:
    """What an accepted action will press.

    Attributes:
      control: The element to press, or None for SCROLL.
      option_steps: For SELECT, ArrowDown presses from the first enabled
        option to the chosen one.
    """

    control: observer.Element | None
    option_steps: int


def _check_fields(action: Action) -> None:
    """Rejects an action whose fields do not fit its kind.

    Raises:
      _RefusedError: With `BAD_ACTION`.
    """
    def bad(detail: str) -> _RefusedError:
        return _RefusedError(Refusal.BAD_ACTION, detail)

    if action.kind not in ACTION_KINDS:
        raise bad(f'Unknown action kind {action.kind!r}.')
    if action.kind == SCROLL:
        if action.target is not None:
            raise bad('Only page scrolling is supported; target must be None.')
        if not action.scroll_y or not math.isfinite(action.scroll_y):
            raise bad('Scrolling needs a non-zero, finite scroll_y.')
        return
    if action.target is None:
        raise bad(f'{action.kind} needs an element target.')
    if action.kind == TYPE and action.text is None:
        raise bad('Typing needs text.')
    if action.kind == CHECK and action.checked is None:
        raise bad('Checking needs the state to set.')


def _element_plan(observation: observer.Observation,
                  action: Action) -> _Plan:
    """Resolves an element-targeted action against the observation.

    Raises:
      _RefusedError: The target is missing, unsupported or unusable.
    """
    matches = [e for e in observation.elements if e.node_id == action.target]
    if len(matches) != 1:
        raise _RefusedError(
            Refusal.UNKNOWN_TARGET,
            f'{len(matches)} observed elements have node_id {action.target}.')
    target = matches[0]
    problem = _control_problem(observation, target, action.kind)
    if problem is not None:
        raise _RefusedError(*problem)
    if action.kind == CHECK:
        if target.role == 'radio' and not action.checked:
            raise _RefusedError(Refusal.BAD_ACTION,
                                'A radio button cannot be unchecked by '
                                'clicking it.')
        if bool(target.states.get('checked')) == action.checked:
            raise _RefusedError(Refusal.ALREADY_IN_STATE,
                                f'The target is already checked='
                                f'{action.checked}.')
    if action.kind != SELECT:
        return _Plan(control=target, option_steps=0)
    dropdown = _dropdown_of(observation, target)
    if dropdown is None:  # Ruled out by _control_problem; keeps types exact.
        raise _RefusedError(Refusal.UNSUPPORTED_ROLE, 'No dropdown.')
    enabled = [o for o in _options_of(observation, dropdown)
               if not o.states.get('disabled')]
    return _Plan(control=dropdown, option_steps=enabled.index(target))


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
