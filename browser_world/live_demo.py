"""Live four-plane viewer: watch the observer while you use the booking site.

Opens a real (headed) Chromium window on the booking site and serves a
viewer page. Click and type in the Chromium window; the viewer shows each new
observation, a wireframe drawn only from the observation, and what changed.

Usage:

  python -m browser_world.live_demo
  # Chromium opens on the booking site; open http://127.0.0.1:7801 to watch.

  python -m browser_world.live_demo --record frames.json
  # Headless: runs a scripted booking and saves every frame for playback.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import pathlib
import threading
import time
from typing import Any

from playwright import sync_api
from starlette import applications
from starlette import requests
from starlette import responses
from starlette import routing
import uvicorn

from browser_world import booking
from browser_world import delta
from browser_world import observer
from browser_world import server

_VIEWER_HTML = pathlib.Path(__file__).parent / 'static' / 'viewer.html'
_START_TIMEOUT_S = 10
_POLL_MS = 400


class FrameLog:
    """Thread-safe, append-only list of observation frames.

    The Playwright thread appends; the viewer server's thread reads.
    """

    def __init__(self):
        self._frames: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last: observer.Observation | None = None

    def add(self, observation: observer.Observation, label: str,
            observe_ms: float) -> None:
        """Appends a frame for `observation`, with its delta from the last.

        Args:
          observation: The new observation.
          label: What caused it, such as the action just taken.
          observe_ms: Time the observation took.
        """
        change = (delta.diff(self._last, observation)
                  if self._last is not None else None)
        frame = {
            'label': label,
            'observe_ms': round(observe_ms, 1),
            'observation': observation.to_dict(),
            'text': observation.to_text(),
            'delta': change,
        }
        with self._lock:
            frame['index'] = len(self._frames)
            self._frames.append(frame)
        self._last = observation

    def since(self, index: int) -> list[dict[str, Any]]:
        """Returns the frames from `index` onwards."""
        with self._lock:
            return self._frames[index:]

    def all(self) -> list[dict[str, Any]]:
        """Returns every frame."""
        return self.since(0)


def _signature(observation: observer.Observation) -> str:
    """Returns a fingerprint of what a person would see as a change."""
    return json.dumps([observation.url] + [
        [e.match_key, e.role, e.name, e.value, sorted(e.states.items()),
         e.in_viewport]
        for e in observation.elements
    ])


def start_server(app: Any, port: int) -> uvicorn.Server:
    """Runs an ASGI app in a daemon thread and waits for it to start.

    Raises:
      RuntimeError: The server did not start in time.
    """
    uv_server = uvicorn.Server(uvicorn.Config(
        app, host='127.0.0.1', port=port, log_level='warning'))
    threading.Thread(target=uv_server.run, daemon=True).start()
    deadline = time.monotonic() + _START_TIMEOUT_S
    while not uv_server.started:
        if time.monotonic() > deadline:
            raise RuntimeError(f'Server on port {port} did not start.')
        time.sleep(0.05)
    return uv_server


def create_viewer_app(log: FrameLog) -> applications.Starlette:
    """Builds the viewer: the page plus a frames endpoint it polls."""

    async def page(request: requests.Request) -> responses.Response:
        del request  # Unused.
        html = _VIEWER_HTML.read_text(encoding='utf-8')
        return responses.HTMLResponse('<!doctype html>\n' + html)

    async def frames(request: requests.Request) -> responses.Response:
        after = int(request.query_params.get('after', '0'))
        return responses.JSONResponse(log.since(after))

    return applications.Starlette(routes=[
        routing.Route('/', page),
        routing.Route('/api/frames', frames),
    ])


def _observe(watcher: observer.Observer, log: FrameLog, label: str) -> None:
    started = time.perf_counter()
    observation = watcher.observe()
    log.add(observation, label, (time.perf_counter() - started) * 1000)


def watch_live(page: sync_api.Page, watcher: observer.Observer,
               log: FrameLog) -> None:
    """Records a frame whenever what the page shows changes.

    Polls instead of relying on DOM mutations alone, because typing changes
    a field's value without mutating the DOM. Returns when the page closes.
    """
    last_signature = ''
    count = 0
    while not page.is_closed():
        try:
            watcher.wait_until_settled(quiet_ms=150, timeout_ms=2000)
            started = time.perf_counter()
            observation = watcher.observe()
            observe_ms = (time.perf_counter() - started) * 1000
            signature = _signature(observation)
            if signature != last_signature:
                label = 'Page loaded' if not count else f'Change {count}'
                log.add(observation, label, observe_ms)
                last_signature = signature
                count += 1
            page.wait_for_timeout(_POLL_MS)
        except sync_api.Error:
            if page.is_closed():
                return
            # A navigation can interrupt an observation; try again.


def _scripted_steps(page: sync_api.Page) -> list[
        tuple[str, Callable[[], object]]]:
    """Returns the (label, action) steps of the recorded demo booking."""
    steps: list[tuple[str, Callable[[], object]]] = [
        ('Select service: General check-up',
         lambda: page.select_option('#service', 'checkup')),
        ('Select doctor: Dr. Asha Patel',
         lambda: page.select_option('#doctor', 'dr-patel')),
        ('Select date: Tue 24 Nov',
         lambda: page.select_option('#date', '2026-11-24')),
        ('Click Next without choosing a time',
         lambda: page.click('[data-wam-key="next-appointment"]')),
        ('Choose time 10:00',
         lambda: page.check('input[name=slot][value="10:00"]')),
        ('Click Next', lambda: page.click('[data-wam-key="next-appointment"]')),
        ('Type name: Sam Rivera', lambda: page.fill('#name', 'Sam Rivera')),
        ('Type email: sam@x (invalid)', lambda: page.fill('#email', 'sam@x')),
        ('Type phone: 555-010-2030',
         lambda: page.fill('#phone', '555-010-2030')),
        ('Tick the privacy checkbox', lambda: page.check('#consent')),
        ('Click Next', lambda: page.click('[data-wam-key="next-details"]')),
        ('Fix email: sam@example.com',
         lambda: page.fill('#email', 'sam@example.com')),
        ('Click Next', lambda: page.click('[data-wam-key="next-details"]')),
        ('Click Confirm booking',
         lambda: page.click('[data-wam-key="confirm"]')),
    ]
    return steps


def record(page: sync_api.Page, watcher: observer.Observer,
           log: FrameLog) -> None:
    """Runs the scripted booking, recording a frame after every action.

    The date selection also records the intermediate loading state, before
    the time slots arrive.
    """
    _observe(watcher, log, 'Page loaded')
    for label, action in _scripted_steps(page):
        action()
        if label.startswith('Select date'):
            _observe(watcher, log, f'{label} (times still loading)')
            label = f'{label} (times loaded)'
        watcher.wait_until_settled()
        _observe(watcher, log, label)


def main() -> None:
    """Starts the booking site, the viewer and Chromium."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--site-port', type=int, default=7800)
    parser.add_argument('--viewer-port', type=int, default=7801)
    parser.add_argument('--record', metavar='PATH',
                        help='Run headless, record a scripted booking, and '
                        'write the frames to PATH as JSON.')
    args = parser.parse_args()

    log = FrameLog()
    site = start_server(server.create_app(booking.default_scenario()),
                         args.site_port)
    viewer = None
    if not args.record:
        viewer = start_server(create_viewer_app(log), args.viewer_port)
        print(f'Viewer: http://127.0.0.1:{args.viewer_port}  '
              '(close the Chromium window or press Ctrl+C to stop)')
    try:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=bool(args.record))
            page = browser.new_page(viewport={'width': 1280, 'height': 800})
            watcher = observer.attach(page)
            page.goto(f'http://127.0.0.1:{args.site_port}/?session=live')
            page.wait_for_selector('body[data-ready=true]')
            watcher.wait_until_settled()
            if args.record:
                record(page, watcher, log)
                pathlib.Path(args.record).write_text(
                    json.dumps(log.all()), encoding='utf-8')
                print(f'Wrote {len(log.all())} frames to {args.record}')
            else:
                watch_live(page, watcher, log)
            browser.close()
    except KeyboardInterrupt:
        pass
    finally:
        site.should_exit = True
        if viewer is not None:
            viewer.should_exit = True


if __name__ == '__main__':
    main()
