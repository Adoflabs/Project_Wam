"""Records real browser experiences: before, action, during, after, truth.

A `Recorder` takes the observation a decider acted on and the action it
chose, executes the action through the `Executor`, waits for the page to
settle, observes again, and appends one `Transition` to a raw JSONL file.
Refused actions never become transitions; they go to a separate diagnostic
log, because a refusal says nothing about how the browser responds.

Every transition belongs to a run: one continuous stretch of a browser
session with no reset, reload or branch in between. Steps are numbered
within the run, and each step's before-state is checked against the
previous step's after-state; a mismatch is recorded, never silently joined.
`sequence_problems()` says whether a raw record may be used for
continuous-episode training.

Raw records keep everything, including Chrome node ids and `match_key`, for
traceability and label alignment. `model_input()` is the only view a model
may see, and it holds nothing that was unknown before the action.

Typical usage example:

  recorder = recorder_lib.Recorder(page, watcher, hand, raw_dir)
  run_id = recorder.new_run('episode-1', origin='start')
  before = watcher.observe()
  outcome = recorder.record_step(run_id, before, chosen_action)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
import json
import pathlib
import time
from typing import Any
import urllib.parse
import uuid

from playwright import sync_api

from browser_world import executor
from browser_world import observer

SCHEMA_VERSION = 2
TRANSITIONS_FILE = 'transitions.jsonl'
REFUSALS_FILE = 'refusals.jsonl'
# Query parameters that name a session rather than application state. Only
# these are removed when comparing pages.
EPHEMERAL_QUERY_PARAMS = frozenset({'session'})

# Returns backend ground truth as JSON-ready data. Called before and after
# each action; its output is never model input.
TruthFn = Callable[[], Any]


@dataclasses.dataclass(frozen=True)
class Target:
    """The element an action was aimed at, as seen in the before-state.

    Attributes:
      node_id: Chrome's backend node id; meaningful only in that browser.
      index: Position in `before.elements`; how a model refers to it.
      role: The element's role.
      name: The element's accessible name.
      match_key: The page's stable key, for label alignment only.
    """

    node_id: int
    index: int
    role: str
    name: str
    match_key: str | None


@dataclasses.dataclass(frozen=True)
class Continuity:
    """How a step joins the previous step of its run.

    Attributes:
      previous_step: The run's previous recorded step, or None for the
        run's first step.
      continuous: Whether the before-state matches the previous step's
        after-state (always True for a run's first step).
      differences: How they differ, from `differences()`.
    """

    previous_step: int | None
    continuous: bool
    differences: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Transition:
    """One executed action and its observed consequence.

    Attributes:
      schema_version: Version of this record's layout.
      run_id: The continuous run this step belongs to.
      run_origin: How the run began, such as 'start', 'reset' or 'branch'.
      episode_id: The episode (task attempt) the run belongs to.
      step: Position of this step within the run.
      recorded_at_ms: Wall-clock time the step started.
      before: The observation the decider acted on.
      action: The decided action.
      target: The action's target in `before`.
      result: What the executor did, including the mouse path.
      events: Browser events from just before the mouse moved until the
        page settled or timed out, with their completeness.
      window_ms: Length of that event window.
      settled: Whether the page settled before `after` was observed.
      after: The observation once the page settled (or timed out).
      continuity: How this step joins the previous step of its run.
      truth: Backend ground truth `{'before': ..., 'after': ...}`, or None.
      artifacts: Paths of optional screenshot and HTML captures, relative
        to the raw directory. Never model input.
    """

    schema_version: int
    run_id: str
    run_origin: str
    episode_id: str
    step: int
    recorded_at_ms: float
    before: observer.Observation
    action: executor.Action
    target: Target
    result: executor.ActionResult
    events: observer.EventTrace
    window_ms: float
    settled: bool
    after: observer.Observation
    continuity: Continuity
    truth: Mapping[str, Any] | None
    artifacts: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-ready dict of the whole record."""
        return {
            'schema_version': self.schema_version,
            'run_id': self.run_id,
            'run_origin': self.run_origin,
            'episode_id': self.episode_id,
            'step': self.step,
            'recorded_at_ms': self.recorded_at_ms,
            'before': self.before.to_dict(),
            'action': dataclasses.asdict(self.action),
            'target': dataclasses.asdict(self.target),
            'result': self.result.to_dict(),
            'events': [dataclasses.asdict(e) for e in self.events.events],
            'events_status': self.events.status(),
            'window_ms': self.window_ms,
            'settled': self.settled,
            'after': self.after.to_dict(),
            'continuity': dataclasses.asdict(self.continuity),
            'truth': self.truth,
            'artifacts': dict(self.artifacts),
        }


@dataclasses.dataclass
class _Run:
    """Recorder-side state of one continuous run."""

    episode_id: str
    origin: str
    next_step: int = 0
    last_after: dict[str, Any] | None = None


class Recorder:
    """Executes decided actions and records what really happened."""

    def __init__(self, page: sync_api.Page, watcher: observer.Observer,
                 hand: executor.Executor, raw_dir: pathlib.Path,
                 truth_fn: TruthFn | None = None,
                 capture_baselines: bool = False):
        """Creates a recorder writing into `raw_dir`.

        Args:
          page: The page being acted on.
          watcher: The observer attached to `page`.
          hand: The executor for `page`.
          raw_dir: Directory for the raw JSONL files and artifacts.
          truth_fn: Optional source of backend ground truth.
          capture_baselines: Whether to also save a screenshot and the HTML
            before and after each action, for future comparisons of input
            types on identical experiences. Off by default.
        """
        self._page = page
        self._watcher = watcher
        self._hand = hand
        self._raw_dir = raw_dir
        self._truth_fn = truth_fn
        self._capture_baselines = capture_baselines
        self._runs: dict[str, _Run] = {}

    @property
    def transitions_path(self) -> pathlib.Path:
        """The raw JSONL file of executed transitions."""
        return self._raw_dir / TRANSITIONS_FILE

    @property
    def refusals_path(self) -> pathlib.Path:
        """The diagnostic JSONL file of refused actions."""
        return self._raw_dir / REFUSALS_FILE

    def new_run(self, episode_id: str, origin: str) -> str:
        """Starts a continuous run and returns its id.

        Call it at the start of every episode and again after anything that
        breaks continuity: a backend reset, a reload, or replaying a branch.

        Args:
          episode_id: The episode the run belongs to.
          origin: How the run begins, such as 'start', 'reset' or 'branch'.

        Returns:
          A new, unique run id.
        """
        run_id = f'{episode_id}/run-{uuid.uuid4().hex[:8]}'
        self._runs[run_id] = _Run(episode_id, origin)
        return run_id

    def record_step(self, run_id: str, before: observer.Observation,
                    action: executor.Action
                    ) -> Transition | executor.ActionResult:
        """Executes `action` on `before` and records the outcome as the
        run's next step.

        Args:
          run_id: A run from `new_run()`.
          before: The observation the decider chose the action from.
          action: The decided action.

        Returns:
          The recorded transition, or the executor's result if it refused.
          A refusal does not use up a step number.

        Raises:
          ValueError: `run_id` was not started by this recorder.
        """
        run = self._runs.get(run_id)
        if run is None:
            raise ValueError(f'Unknown run {run_id!r}; call new_run() first.')
        step = run.next_step
        continuity = _continuity(run, before)
        recorded_at_ms = time.time() * 1000
        truth_before = self._truth()
        artifacts = self._capture(run_id, step, 'before')
        self._watcher.start_events()
        window_start = time.monotonic()
        result = self._hand.execute(before, action)
        if not result.executed:
            self._watcher.take_events()  # Close the window; nothing ran.
            self._append(self.refusals_path, _refusal_record(
                run_id, run.episode_id, step, recorded_at_ms, before, result))
            return result
        settled = self._watcher.wait_until_settled()
        after = self._watcher.observe()
        events = self._watcher.take_events()
        window_ms = round((time.monotonic() - window_start) * 1000, 2)
        artifacts.update(self._capture(run_id, step, 'after'))
        truth = None
        if self._truth_fn is not None:
            truth = {'before': truth_before, 'after': self._truth()}
        run.next_step += 1
        run.last_after = canonical(after)
        transition = Transition(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            run_origin=run.origin,
            episode_id=run.episode_id,
            step=step,
            recorded_at_ms=recorded_at_ms,
            before=before,
            action=action,
            target=_target_of(before, action),
            result=result,
            events=events,
            window_ms=window_ms,
            settled=settled,
            after=after,
            continuity=continuity,
            truth=truth,
            artifacts=artifacts,
        )
        self._append(self.transitions_path, transition.to_dict())
        return transition

    def _truth(self) -> Any:
        return self._truth_fn() if self._truth_fn is not None else None

    def _capture(self, run_id: str, step: int,
                 moment: str) -> dict[str, str]:
        """Saves a screenshot and the HTML when baselines are enabled.

        Returns:
          Artifact paths relative to the raw directory, keyed like
          'screenshot_before'.
        """
        if not self._capture_baselines:
            return {}
        folder = pathlib.Path('artifacts') / run_id
        (self._raw_dir / folder).mkdir(parents=True, exist_ok=True)
        stem = f'step{step:04d}_{moment}'
        screenshot = folder / f'{stem}.png'
        html = folder / f'{stem}.html'
        # caret='initial' stops Playwright injecting a style element to hide
        # the caret; that injection mutates the DOM and would make the
        # observation stale.
        self._page.screenshot(path=self._raw_dir / screenshot,
                              caret='initial')
        (self._raw_dir / html).write_text(self._page.content(),
                                          encoding='utf-8')
        return {f'screenshot_{moment}': str(screenshot),
                f'html_{moment}': str(html)}

    def _append(self, path: pathlib.Path, record: Mapping[str, Any]) -> None:
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as file:
            file.write(json.dumps(record) + '\n')


def _target_of(before: observer.Observation,
               action: executor.Action) -> Target:
    """Describes the action's target as it appears in `before`.

    Raises:
      ValueError: The target is not in `before`; the executor would have
        refused such an action.
    """
    for index, element in enumerate(before.elements):
        if element.node_id == action.target:
            return Target(element.node_id, index, element.role,
                          element.name, element.match_key)
    raise ValueError(f'Target {action.target} is not in the observation.')


def _continuity(run: _Run, before: observer.Observation) -> Continuity:
    """Compares `before` with the run's previous after-state."""
    if run.last_after is None:
        return Continuity(previous_step=None, continuous=True, differences=())
    found = differences(run.last_after, canonical(before))
    return Continuity(previous_step=run.next_step - 1, continuous=not found,
                      differences=tuple(found))


def _refusal_record(run_id: str, episode_id: str, step: int,
                    recorded_at_ms: float, before: observer.Observation,
                    result: executor.ActionResult) -> dict[str, Any]:
    """Returns the diagnostic record for a refused action."""
    return {
        'schema_version': SCHEMA_VERSION,
        'run_id': run_id,
        'episode_id': episode_id,
        'attempted_step': step,
        'recorded_at_ms': recorded_at_ms,
        'observation_sequence': before.sequence,
        'result': result.to_dict(),
    }


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    """Returns the records of a JSONL file, in order."""
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def sequence_problems(record: Mapping[str, Any]) -> list[str]:
    """Returns why a raw transition may not be used for continuous-episode
    training; an empty list means it may.

    Records from schema version 1 have no run identity or continuity check,
    so they stay as diagnostic evidence only. Event completeness is reported
    in `events_status` but does not exclude a record, since the first model
    predicts the settled after-state.
    """
    if record.get('schema_version', 0) < 2:
        return ['legacy record without run identity or continuity check']
    problems = []
    if not record['continuity']['continuous']:
        problems.append('before-state does not match the previous '
                        "step's after-state")
    if not record['settled']:
        problems.append('page did not settle before the after-observation')
    return problems


def canonical(observation: observer.Observation) -> dict[str, Any]:
    """Returns the replay-comparable content of an observation.

    Drops everything that legitimately differs between identical runs:
    Chrome node ids, `match_key`, observation counters and digests,
    timestamps, the temporal plane and the URL's `EPHEMERAL_QUERY_PARAMS`.
    Other query parameters and the fragment are kept, since they can name
    application state.
    Elements stay in document order and parents become list positions, so
    two elements with the same label cannot be confused.
    """
    position = {e.node_id: i for i, e in enumerate(observation.elements)}
    elements = []
    for element in observation.elements:
        box = element.box
        parent = (position.get(element.parent_id)
                  if element.parent_id is not None else None)
        elements.append({
            'role': element.role,
            'name': element.name,
            'value': element.value,
            'states': dict(element.states),
            'parent': parent,
            'box': ([box.x, box.y, box.width, box.height]
                    if box is not None else None),
            'visible': element.visible,
            'in_viewport': element.in_viewport,
            'obstructed': element.obstructed,
            'pinned': element.pinned,
        })
    motion = observation.motion
    return {
        'url': stable_url(observation.url),
        'title': observation.title,
        'scroll': [motion.scroll_x, motion.scroll_y],
        'viewport': [motion.viewport_width, motion.viewport_height],
        'elements': elements,
    }


def stable_url(url: str) -> str:
    """Returns `url` without its `EPHEMERAL_QUERY_PARAMS`."""
    parts = urllib.parse.urlsplit(url)
    query = [(key, value) for key, value in
             urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
             if key not in EPHEMERAL_QUERY_PARAMS]
    return urllib.parse.urlunsplit(
        parts._replace(query=urllib.parse.urlencode(query)))


def differences(first: Mapping[str, Any], second: Mapping[str, Any],
                tolerance_px: float = 1.0) -> list[str]:
    """Lists how two `canonical()` observations differ.

    Args:
      first: One canonical observation.
      second: Another canonical observation.
      tolerance_px: Largest box or scroll difference, in CSS pixels, that
        still counts as equal (sub-pixel layout rounding).

    Returns:
      Human-readable differences; empty when the observations match.
    """
    found = []
    for key in ('url', 'title', 'viewport'):
        if first[key] != second[key]:
            found.append(f'{key}: {first[key]!r} != {second[key]!r}')
    if not _close(first['scroll'], second['scroll'], tolerance_px):
        found.append(f"scroll: {first['scroll']} != {second['scroll']}")
    if len(first['elements']) != len(second['elements']):
        found.append(f"element count: {len(first['elements'])} != "
                     f"{len(second['elements'])}")
        return found
    for i, (a, b) in enumerate(zip(first['elements'], second['elements'])):
        for field in a:
            if field == 'box':
                same = (a['box'] == b['box'] if a['box'] is None
                        or b['box'] is None
                        else _close(a['box'], b['box'], tolerance_px))
            else:
                same = a[field] == b[field]
            if not same:
                found.append(f"element {i} ({a['role']} {a['name']!r}) "
                             f"{field}: {a[field]!r} != {b[field]!r}")
    return found


def _close(first: list[float], second: list[float],
           tolerance: float) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(first, second))


def model_input(transition: Transition) -> dict[str, Any]:
    """Returns the only part of a transition a world model may see.

    It holds the before-state's elements without Chrome ids or
    `match_key` (parents become list positions), the page-level motion
    plane, and the action with its target as a list position. It excludes
    the result, events, after-state, backend truth, artifacts and
    identifiers. It also leaves out the before-state's temporal plane for
    now: its counts depend on when the previous observation was taken
    rather than on what the page shows.
    """
    before = canonical(transition.before)
    return {
        'state': {
            'url': before['url'],
            'title': before['title'],
            'scroll': before['scroll'],
            'viewport': before['viewport'],
            'elements': before['elements'],
        },
        'action': {
            'kind': transition.action.kind,
            'target_index': transition.target.index,
            'text': transition.action.text,
        },
    }
