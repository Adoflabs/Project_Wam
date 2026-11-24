"""Shared fixtures: a live booking server and observed Chromium pages."""

# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

from __future__ import annotations

from collections.abc import Callable, Iterator
import dataclasses
import socket
import threading
import time
import uuid

from playwright import sync_api
import pytest
import uvicorn

from browser_world import booking
from browser_world import observer
from browser_world import server

_SERVER_START_TIMEOUT_S = 10
TEST_URL = 'http://wam.test/'


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.fixture(scope='session')
def base_url() -> Iterator[str]:
    """Runs the booking server with the default scenario in a thread."""
    port = _free_port()
    config = uvicorn.Config(server.create_app(booking.default_scenario()),
                            host='127.0.0.1', port=port, log_level='warning')
    uv_server = uvicorn.Server(config)
    thread = threading.Thread(target=uv_server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + _SERVER_START_TIMEOUT_S
    while not uv_server.started:
        if time.monotonic() > deadline:
            raise RuntimeError('Booking server did not start.')
        time.sleep(0.05)
    yield f'http://127.0.0.1:{port}'
    uv_server.should_exit = True
    thread.join(timeout=_SERVER_START_TIMEOUT_S)


@pytest.fixture(scope='session')
def browser() -> Iterator[sync_api.Browser]:
    with sync_api.sync_playwright() as playwright:
        chromium = playwright.chromium.launch()
        yield chromium
        chromium.close()


@dataclasses.dataclass
class World:
    """One isolated environment instance driven through a real page.

    Attributes:
      page: The Playwright page showing the booking site.
      watcher: The four-plane observer attached to `page`.
      session_id: The environment session this page uses.
      base_url: The server's root URL.
    """

    page: sync_api.Page
    watcher: observer.Observer
    session_id: str
    base_url: str

    def act(self, action: Callable[[], object]) -> observer.Observation:
        """Runs `action`, waits for the page to settle, and observes."""
        action()
        if not self.watcher.wait_until_settled():
            raise TimeoutError('Page did not settle after the action.')
        return self.watcher.observe()

    def truth_bookings(self) -> list[dict[str, str]]:
        """Returns backend ground truth; never shown to the page."""
        response = self.page.request.get(
            f'{self.base_url}/api/truth/bookings',
            headers={server.SESSION_HEADER: self.session_id})
        return response.json()

    def reset(self) -> None:
        self.page.request.post(
            f'{self.base_url}/api/reset',
            headers={server.SESSION_HEADER: self.session_id})

    def choose_appointment(self, service: str, doctor: str, date_iso: str,
                           slot: str | None) -> observer.Observation:
        """Fills step 1. Leaves the slot unchosen when `slot` is None."""
        self.act(lambda: self.page.select_option('#service', service))
        self.act(lambda: self.page.select_option('#doctor', doctor))
        observation = self.act(
            lambda: self.page.select_option('#date', date_iso))
        if slot is not None:
            observation = self.act(
                lambda: self.page.check(f'input[name=slot][value="{slot}"]'))
        return observation

    def fill_details(self, name: str, email: str, phone: str,
                     consent: bool) -> observer.Observation:
        """Fills step 2 without submitting it."""
        self.act(lambda: self.page.fill('#name', name))
        self.act(lambda: self.page.fill('#email', email))
        observation = self.act(lambda: self.page.fill('#phone', phone))
        if consent:
            observation = self.act(lambda: self.page.check('#consent'))
        return observation

    def click(self, match_key: str) -> observer.Observation:
        return self.act(
            lambda: self.page.click(f'[data-wam-key="{match_key}"]'))


@pytest.fixture
def new_page(
        browser: sync_api.Browser) -> Iterator[Callable[..., sync_api.Page]]:
    """Returns a factory for pages, each in its own browser context.

    Contexts are closed when the test ends.
    """
    contexts = []

    def factory(viewport_height: int = 800) -> sync_api.Page:
        context = browser.new_context(
            viewport={'width': 1280, 'height': viewport_height})
        contexts.append(context)
        return context.new_page()

    yield factory
    for context in contexts:
        context.close()


@pytest.fixture
def make_world(new_page: Callable[..., sync_api.Page],
               base_url: str) -> Callable[..., World]:
    """Returns a factory for isolated booking worlds."""

    def factory(viewport_height: int = 800) -> World:
        page = new_page(viewport_height)
        watcher = observer.attach(page)
        session_id = uuid.uuid4().hex
        page.goto(f'{base_url}/?session={session_id}')
        page.wait_for_selector('body[data-ready=true]')
        watcher.wait_until_settled()
        watcher.observe()  # Clears the page-load temporal counts.
        return World(page, watcher, session_id, base_url)

    return factory


@pytest.fixture
def world(make_world: Callable[..., World]) -> World:
    return make_world()


@pytest.fixture
def open_page(
        new_page: Callable[..., sync_api.Page]
) -> Callable[..., tuple[sync_api.Page, observer.Observer]]:
    """Returns a factory that serves `html` at a fake URL and observes it."""

    def factory(html: str, query: str = '',
                viewport_height: int = 800
                ) -> tuple[sync_api.Page, observer.Observer]:
        page = new_page(viewport_height)
        page.route(f'{TEST_URL}**', lambda route: route.fulfill(
            body=html, content_type='text/html'))
        watcher = observer.attach(page)
        page.goto(f'{TEST_URL}{query}')
        watcher.wait_until_settled()
        return page, watcher

    return factory
