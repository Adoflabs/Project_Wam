"""Prints what the four-plane observer sees on a page.

With no URL, starts the booking environment in-process and observes its
first page. Useful for checking the observer by eye.

Usage:

  python -m browser_world.diagnostics.inspect_page
  python -m browser_world.diagnostics.inspect_page --url https://example.com --json
"""

from __future__ import annotations

import argparse
import json
import threading
import time

from playwright import sync_api
import uvicorn

from browser_world.environment import booking
from browser_world.interaction import observer
from browser_world.environment import server

_LOCAL_PORT = 7899
_START_TIMEOUT_S = 10


def _start_local_server() -> tuple[uvicorn.Server, str]:
    """Starts the booking server in a daemon thread.

    Returns:
      A tuple (server, page_url) for the running environment.

    Raises:
      RuntimeError: The server did not start in time.
    """
    uv_server = uvicorn.Server(uvicorn.Config(
        server.create_app(booking.default_scenario()),
        host='127.0.0.1', port=_LOCAL_PORT, log_level='warning'))
    threading.Thread(target=uv_server.run, daemon=True).start()
    deadline = time.monotonic() + _START_TIMEOUT_S
    while not uv_server.started:
        if time.monotonic() > deadline:
            raise RuntimeError('Booking server did not start.')
        time.sleep(0.05)
    return uv_server, f'http://127.0.0.1:{_LOCAL_PORT}/?session=inspect'


def main() -> None:
    """Observes one page and prints the observation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--url', help='Page to observe. Default: local env.')
    parser.add_argument('--json', action='store_true',
                        help='Print JSON instead of compact text.')
    parser.add_argument('--planes', default=','.join(observer.ALL_PLANES),
                        help='Comma-separated planes to show.')
    args = parser.parse_args()

    local_server = None
    url = args.url
    if url is None:
        local_server, url = _start_local_server()
    planes = [p for p in args.planes.split(',') if p]
    try:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={'width': 1280, 'height': 800})
            watcher = observer.attach(page)
            page.goto(url)
            watcher.wait_until_settled()
            started = time.perf_counter()
            observation = watcher.observe()
            elapsed_ms = (time.perf_counter() - started) * 1000
            if args.json:
                print(json.dumps(observation.to_dict(planes), indent=2))
            else:
                print(observation.to_text(planes))
                print(f'# {len(observation.elements)} elements, observed in '
                      f'{elapsed_ms:.0f} ms')
            browser.close()
    finally:
        if local_server is not None:
            local_server.should_exit = True


if __name__ == '__main__':
    main()
