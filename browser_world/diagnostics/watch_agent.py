"""Watch the agent book appointments in a real Chromium window, live.

Starts the booking server and a live viewer on localhost, opens a visible
Chromium window, and lets the goal-directed decider run episodes across
scenario families: successful bookings, race conflicts and validation
errors. Every step goes through the real observer, executor and recorder.

The agent's pointer and its next target are highlighted with Chrome's
debugging overlay, which is drawn outside the page: it changes nothing the
observer sees.

Run on your own computer, from the repository root:

  python -m browser_world.diagnostics.watch_agent
  # Watch the Chromium window; open http://127.0.0.1:7801 for the viewer.

  python -m browser_world.diagnostics.watch_agent --episodes 6 --families civic-wizard
  python -m browser_world.diagnostics.watch_agent --headless   # no window, for checks
"""

from __future__ import annotations

import argparse
import pathlib
import time
import uuid

from playwright import sync_api

from browser_world.environment import booking
from browser_world.interaction import executor
from browser_world.collection import goal_policy
from browser_world.diagnostics import live_demo
from browser_world.interaction import observer
from browser_world.collection import recorder
from browser_world.environment import scenarios
from browser_world.environment import server

# Slow enough to follow by eye.
WATCH_SPEED = executor.Speed(move_steps=30, move_ms=700.0, key_delay_ms=90.0)
_OUTCOME_CYCLE = (scenarios.SUCCESS, scenarios.CONFLICT,
                  scenarios.INVALID_EMAIL)
_POINTER_PX = 14
_POINTER_COLOR = {'r': 220, 'g': 40, 'b': 110, 'a': 0.85}
_TARGET_COLOR = {'r': 31, 'g': 111, 'b': 178, 'a': 0.35}
_MAX_STEPS = 40


class Overlay:
    """Draws the pointer and the next target with Chrome's debug overlay.

    The overlay is painted above the page by the browser itself; it adds
    nothing to the DOM, so observations and freshness checks are unchanged.
    """

    def __init__(self, page: sync_api.Page):
        """Opens a DevTools session on `page` for drawing."""
        self._cdp = page.context.new_cdp_session(page)
        self._cdp.send('DOM.enable')
        self._cdp.send('Overlay.enable')

    def pointer(self, x: float, y: float) -> None:
        """Marks the pointer at viewport point (x, y)."""
        half = _POINTER_PX // 2
        self._cdp.send('Overlay.highlightRect', {
            'x': max(0, round(x) - half), 'y': max(0, round(y) - half),
            'width': _POINTER_PX, 'height': _POINTER_PX,
            'color': _POINTER_COLOR})

    def target(self, node_id: int) -> None:
        """Highlights the element the agent is about to use."""
        self._cdp.send('Overlay.highlightNode', {
            'backendNodeId': node_id,
            'highlightConfig': {'contentColor': _TARGET_COLOR,
                                'showInfo': True}})


def describe(action: executor.Action, target: recorder.Target | None) -> str:
    """Returns a short human description of an action."""
    name = target.name if target is not None else 'page'
    if action.kind == executor.SCROLL:
        return f'scroll the page by {action.scroll_y:+.0f}px'
    if action.kind == executor.TYPE:
        return f'type "{action.text}" into "{name}"'
    if action.kind == executor.CHECK:
        return f'set "{name}" to {"on" if action.checked else "off"}'
    return f'{action.kind} "{name}"'


def _episode(page: sync_api.Page, watcher: observer.Observer,
             overlay: Overlay, log: live_demo.FrameLog, raw_dir: pathlib.Path,
             site: str, assignments: dict[str, booking.Scenario],
             family: str, seed: int, outcome: str, pause_s: float) -> str:
    """Runs and records one episode in `page`; returns how it ended."""
    scenario = scenarios.generate(family, seed)
    session_id = uuid.uuid4().hex
    assignments[session_id] = scenario
    page.goto(f'{site}/?session={session_id}')
    page.wait_for_selector('body[data-ready=true]')
    watcher.wait_until_settled()
    headers = {server.SESSION_HEADER: session_id}
    rec = recorder.Recorder(
        page, watcher,
        executor.Executor(page, watcher, WATCH_SPEED, on_move=overlay.pointer),
        raw_dir,
        truth_fn=lambda: page.request.get(f'{site}/api/truth/bookings',
                                          headers=headers).json())
    goal = scenarios.goal(scenario, seed, outcome)
    text = scenario.presentation.text
    wanted = goal_policy.intents(goal, text)
    run_id = rec.new_run(f'watch-{scenario.scenario_id}-{outcome}', 'start',
                         scenario={'scenario_id': scenario.scenario_id,
                                   'family': family, 'seed': seed,
                                   'split': scenarios.split_of(family)})
    print(f'\n=== {family} (seed {seed}, {scenarios.split_of(family)}) '
          f'goal: {outcome} - {goal.service} / {goal.doctor} / {goal.date} '
          f'{goal.slot}')
    for step in range(_MAX_STEPS):
        watcher.wait_until_settled()
        started = time.perf_counter()
        observation = watcher.observe()
        observe_ms = (time.perf_counter() - started) * 1000
        ended = goal_policy.outcome_of(observation, text)
        if ended is not None:
            log.add(observation, f'{family}: ended - {ended}', observe_ms)
            print(f'    ended: {ended}')
            return ended
        if step == 0:
            log.add(observation, f'{family}: page loaded', observe_ms)
        choice = goal_policy.choose(
            observation, goal_policy.pending(observation, wanted), text)
        if choice is None:
            raise RuntimeError('The decider found nothing useful to do.')
        action, _ = choice
        if action.target is not None:
            target = next(e for e in observation.elements
                          if e.node_id == action.target)
            overlay.target(_dropdown_or_self(observation, target))
            page.wait_for_timeout(pause_s * 1000)
        outcome_of_step = rec.record_step(run_id, observation, action)
        if not isinstance(outcome_of_step, recorder.Transition):
            raise RuntimeError(f'Refused: {outcome_of_step.refusal}')
        label = describe(action, outcome_of_step.target)
        print(f'    step {outcome_of_step.step:2d}: {label}')
        log.add(outcome_of_step.after, f'{family}: {label}', observe_ms)
    raise RuntimeError('The episode did not end.')


def _dropdown_or_self(observation: observer.Observation,
                      element: observer.Element) -> int:
    """Returns the node to highlight: an option's dropdown, else itself."""
    by_id = {e.node_id: e for e in observation.elements}
    node: observer.Element | None = element
    while node is not None and node.box is None:
        node = by_id.get(node.parent_id) if node.parent_id else None
    return node.node_id if node is not None else element.node_id


def main() -> None:
    """Starts the servers and Chromium, and runs the episodes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--episodes', type=int, default=6)
    parser.add_argument('--families', nargs='+',
                        default=['harbor-details_first', 'riverside-combined',
                                 'civic-wizard'])
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pause', type=float, default=0.8,
                        help='Seconds to show each target before acting.')
    parser.add_argument('--site-port', type=int, default=7800)
    parser.add_argument('--viewer-port', type=int, default=7801)
    parser.add_argument('--raw-dir', default='data/raw/watch')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    assignments: dict[str, booking.Scenario] = {}
    app = server.create_app(
        lambda sid: assignments.get(sid, booking.default_scenario()))
    site = live_demo.start_server(app, args.site_port)
    log = live_demo.FrameLog()
    viewer = live_demo.start_server(live_demo.create_viewer_app(log),
                                    args.viewer_port)
    site_url = f'http://127.0.0.1:{args.site_port}'
    print(f'Booking site: {site_url}   Live viewer: '
          f'http://127.0.0.1:{args.viewer_port}')
    results = []
    try:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=args.headless)
            page = browser.new_page(viewport={'width': 1280, 'height': 700})
            watcher = observer.attach(page)
            overlay = Overlay(page)
            for i in range(args.episodes):
                family = args.families[i % len(args.families)]
                outcome = _OUTCOME_CYCLE[i % len(_OUTCOME_CYCLE)]
                results.append((family, outcome, _episode(
                    page, watcher, overlay, log, pathlib.Path(args.raw_dir),
                    site_url, assignments, family, args.seed + i, outcome,
                    args.pause)))
            print('\nSummary:')
            for family, outcome, ended in results:
                print(f'  {family:<24} goal {outcome:<14} ended {ended}')
            print(f'Records: {args.raw_dir}/{recorder.TRANSITIONS_FILE}')
            if not args.headless:
                input('Press Enter to close the browser.')
            browser.close()
    except KeyboardInterrupt:
        pass
    finally:
        for running in (site, viewer):
            running.should_exit = True


if __name__ == '__main__':
    main()
