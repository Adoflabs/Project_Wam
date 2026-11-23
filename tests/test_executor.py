"""E0.3a: observation-grounded actions, executed in real Chromium.

Tests act like a decider: they pick targets from `available_actions` or by
role and name in the observation, never with CSS selectors. Outcomes are
checked with the observer. Page setup may use the existing helpers.
"""

# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

from collections.abc import Callable
import json

from playwright import sync_api
import pytest

from browser_world import booking
from browser_world import executor
from browser_world import observer

_TEST_URL = 'http://wam.test/'

# A target button with a full-width hover band between it and the mouse's
# starting corner, so every path to the button crosses the band. The
# band's behaviour is chosen per test with `?mode=`.
_HOVER_PAGE = """<!doctype html>
<body style="margin:0;font:16px sans-serif">
  <div id="band" style="position:absolute;top:100px;height:80px;
       left:0;right:0;background:#eee">band</div>
  <button id="go" style="position:absolute;top:400px;left:600px">Go</button>
  <p role="status" id="status" style="position:absolute;top:600px">idle</p>
  <script>
    const mode = new URLSearchParams(location.search).get('mode');
    const go = document.getElementById('go');
    go.onclick = () => { document.getElementById('status').textContent =
                             'clicked'; };
    document.getElementById('band').onmouseenter = () => {
      if (mode === 'tooltip') {
        const tip = document.createElement('p');
        tip.textContent = 'tooltip';
        tip.style.cssText = 'position:absolute;top:20px;left:900px';
        document.body.append(tip);
      } else if (mode === 'disable') {
        go.disabled = true;
      } else if (mode === 'menu') {
        const menu = document.createElement('div');
        menu.style.cssText = 'position:fixed;top:300px;left:500px;' +
            'width:300px;height:200px;background:#ccf;z-index:10';
        document.body.append(menu);
      }
    };
  </script>
</body>"""

_CONTROLS_PAGE = """<!doctype html>
<body style="margin:0;font:16px sans-serif">
  <button id="first" style="position:absolute;top:40px;left:40px">Save</button>
  <button id="second" style="position:absolute;top:40px;left:300px">Save</button>
  <button disabled style="position:absolute;top:120px;left:40px">Locked</button>
  <button style="position:absolute;top:120px;left:300px;opacity:0">Ghost</button>
  <button id="covered" style="position:absolute;top:200px;left:40px">Covered</button>
  <div style="position:absolute;top:190px;left:20px;width:200px;height:60px;
       background:#000;opacity:.5"></div>
  <div style="position:absolute;top:300px;left:40px;width:200px;height:80px;
       overflow:auto;border:1px solid">
    <div style="height:200px"></div><button>Clipped</button>
    <div style="height:200px"></div>
  </div>
  <p role="status" id="status" style="position:absolute;top:500px">idle</p>
  <script>
    document.getElementById('first').onclick = () => {
      document.getElementById('status').textContent = 'first'; };
    document.getElementById('second').onclick = () => {
      document.getElementById('status').textContent = 'second'; };
  </script>
</body>"""

_MOVES_PAGE = """<!doctype html>
<body style="margin:0;height:900px">
  <button style="position:absolute;top:500px;left:700px">Target</button>
  <p role="status" id="moves" style="position:absolute;top:20px">moves 0</p>
  <script>
    let count = 0;
    document.addEventListener('mousemove', () => {
      count += 1;
      document.getElementById('moves').textContent = `moves ${count}`;
    });
  </script>
</body>"""


@pytest.fixture
def open_page(
        new_page: Callable[..., sync_api.Page]
) -> Callable[..., tuple[sync_api.Page, observer.Observer]]:
    """Returns a factory that serves `html` at a fake URL and observes it."""

    def factory(html: str, query: str = '',
                viewport_height: int = 800
                ) -> tuple[sync_api.Page, observer.Observer]:
        page = new_page(viewport_height)
        page.route(f'{_TEST_URL}**', lambda route: route.fulfill(
            body=html, content_type='text/html'))
        watcher = observer.attach(page)
        page.goto(f'{_TEST_URL}{query}')
        watcher.wait_until_settled()
        return page, watcher

    return factory


def _find(observation: observer.Observation, role: str,
          name: str) -> observer.Element:
    """Picks the single element with this role and name, like a decider."""
    matches = [e for e in observation.elements
               if e.role == role and e.name == name]
    assert len(matches) == 1, f'{len(matches)} matches for {role} {name}'
    return matches[0]


def _status(watcher: observer.Observer) -> str:
    watcher.wait_until_settled()
    return _find_status(watcher.observe())


def _find_status(observation: observer.Observation) -> str:
    return next(e.name for e in observation.elements if e.role == 'status')


def _click(element: observer.Element) -> executor.Action:
    return executor.Action(executor.CLICK, element.node_id)


def test_click_next_from_affordances(world):
    hand = executor.Executor(world.page, world.watcher)
    before = world.watcher.observe()
    choices = executor.available_actions(before)
    assert [(c.kind, c.name) for c in choices] == [('click', 'Next')]

    result = hand.execute(before, executor.Action(choices[0].kind,
                                                  choices[0].target))
    assert result.executed, result
    world.watcher.wait_until_settled()
    after = world.watcher.observe()
    assert after.find('error-service').name == booking.MSG_SERVICE_REQUIRED

    start, end = result.pointer_path[0], result.pointer_path[-1]
    press_x, press_y = _find(before, 'button', 'Next').hit_point
    assert (start.x, start.y) == (0.0, 0.0)
    assert (end.x, end.y) == (press_x, press_y)  # Scroll is 0 here.
    assert len(result.pointer_path) == executor.NORMAL_SPEED.move_steps + 1
    assert hand.pointer == (press_x, press_y)


def test_type_email_then_replace_it(world):
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', '09:00')
    world.click('next-appointment')
    hand = executor.Executor(world.page, world.watcher)

    observation = world.watcher.observe()
    kinds = {(c.kind, c.name) for c in executor.available_actions(observation)}
    assert ('type', 'Email address') in kinds
    email = _find(observation, 'textbox', 'Email address')
    result = hand.execute(observation, executor.Action(
        executor.TYPE, email.node_id, 'sam@x'))
    assert result.executed, result
    world.watcher.wait_until_settled()
    observation = world.watcher.observe()
    assert _find(observation, 'textbox', 'Email address').value == 'sam@x'

    result = hand.execute(observation, executor.Action(
        executor.TYPE, email.node_id, 'sam@example.com'))
    assert result.executed, result
    world.watcher.wait_until_settled()
    observation = world.watcher.observe()
    assert (_find(observation, 'textbox', 'Email address').value
            == 'sam@example.com')


def test_reused_observation_is_refused(world):
    hand = executor.Executor(world.page, world.watcher)
    observation = world.watcher.observe()
    next_button = _find(observation, 'button', 'Next')
    assert hand.execute(observation, _click(next_button)).executed
    again = hand.execute(observation, _click(next_button))
    assert again.refusal is executor.Refusal.OBSERVATION_USED
    assert not again.executed


def test_unknown_target_and_bad_actions_are_refused(world):
    hand = executor.Executor(world.page, world.watcher)
    observation = world.watcher.observe()
    next_button = _find(observation, 'button', 'Next')
    cases = [
        (executor.Action(executor.CLICK, 999_999),
         executor.Refusal.UNKNOWN_TARGET),
        (executor.Action(executor.TYPE, next_button.node_id, 'hi'),
         executor.Refusal.UNSUPPORTED_ROLE),
        (executor.Action('hover', next_button.node_id),
         executor.Refusal.BAD_ACTION),
        (executor.Action(executor.TYPE, next_button.node_id),
         executor.Refusal.BAD_ACTION),
    ]
    for action, refusal in cases:
        assert hand.execute(observation, action).refusal is refusal


def test_stale_observation_after_rerender_is_refused(world):
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', slot=None)
    hand = executor.Executor(world.page, world.watcher)
    old = world.watcher.observe()
    world.act(lambda: world.page.select_option('#date', '2026-11-25'))
    result = hand.execute(old, _click(_find(old, 'button', 'Next')))
    assert result.refusal is executor.Refusal.STALE_OBSERVATION
    assert not result.pointer_path  # Refused before moving.


def test_offscreen_refused_then_clicks_after_scrolling(make_world):
    world = make_world(viewport_height=300)
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', slot=None)
    hand = executor.Executor(world.page, world.watcher)
    observation = world.watcher.observe()
    next_button = _find(observation, 'button', 'Next')
    result = hand.execute(observation, _click(next_button))
    assert result.refusal is executor.Refusal.NOT_IN_VIEWPORT

    world.page.mouse.wheel(0, 400)
    world.watcher.wait_until_settled()
    scrolled = world.watcher.observe()
    assert scrolled.motion.scroll_y > 0
    next_button = _find(scrolled, 'button', 'Next')
    result = hand.execute(scrolled, _click(next_button))
    assert result.executed, result
    end = result.pointer_path[-1]
    assert end.y == next_button.hit_point[1] - scrolled.motion.scroll_y
    world.watcher.wait_until_settled()
    after = world.watcher.observe()
    assert after.find('error-slot').name == booking.MSG_SLOT_REQUIRED


def test_partly_visible_field_is_pressed_in_its_visible_part(make_world):
    # Regression: the field's centre is below the viewport but its top is
    # visible. Found by running the executor on the real booking flow.
    world = make_world(viewport_height=300)
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', '09:00')
    world.click('next-appointment')
    hand = executor.Executor(world.page, world.watcher)
    observation = world.watcher.observe()
    email = _find(observation, 'textbox', 'Email address')
    viewport_bottom = (observation.motion.scroll_y
                       + observation.motion.viewport_height)
    assert email.box.center[1] > viewport_bottom  # Centre is off-screen.
    assert email.hit_point[1] < viewport_bottom
    result = hand.execute(observation, executor.Action(
        executor.TYPE, email.node_id, 'sam@example.com'))
    assert result.executed, result
    world.watcher.wait_until_settled()
    after = world.watcher.observe()
    assert (_find(after, 'textbox', 'Email address').value
            == 'sam@example.com')


def test_scrolling_after_observation_makes_it_stale(open_page):
    page, watcher = open_page(_MOVES_PAGE, viewport_height=700)
    hand = executor.Executor(page, watcher)
    observation = watcher.observe()
    page.mouse.wheel(0, 50)
    watcher.wait_until_settled()
    assert page.evaluate('window.scrollY') > 0
    result = hand.execute(observation, _click(_find(observation, 'button',
                                                    'Target')))
    assert result.refusal is executor.Refusal.STALE_OBSERVATION


def test_duplicate_names_click_the_chosen_element(open_page):
    page, watcher = open_page(_CONTROLS_PAGE)
    hand = executor.Executor(page, watcher)
    observation = watcher.observe()
    saves = [e for e in observation.elements
             if e.role == 'button' and e.name == 'Save']
    assert len(saves) == 2
    result = hand.execute(observation, _click(saves[1]))
    assert result.executed, result
    assert _status(watcher) == 'second'


def test_blocked_targets_are_refused_and_not_offered(open_page):
    page, watcher = open_page(_CONTROLS_PAGE)
    hand = executor.Executor(page, watcher)
    observation = watcher.observe()
    offered = {c.name for c in executor.available_actions(observation)}
    assert offered == {'Save'}
    expected = {
        'Locked': executor.Refusal.DISABLED,
        'Ghost': executor.Refusal.NOT_VISIBLE,
        'Covered': executor.Refusal.OBSTRUCTED,
        'Clipped': executor.Refusal.OBSTRUCTED,
    }
    for name, refusal in expected.items():
        result = hand.execute(observation, _click(_find(observation,
                                                        'button', name)))
        assert result.refusal is refusal, (name, result)
        assert not result.pointer_path
    assert _status(watcher) == 'idle'


def test_disabled_after_observation_is_refused(open_page):
    page, watcher = open_page(_HOVER_PAGE)
    hand = executor.Executor(page, watcher)
    observation = watcher.observe()
    page.evaluate("document.getElementById('go').disabled = true")
    result = hand.execute(observation, _click(_find(observation, 'button',
                                                    'Go')))
    assert result.refusal is executor.Refusal.STALE_OBSERVATION


def test_harmless_hover_change_still_clicks(open_page):
    page, watcher = open_page(_HOVER_PAGE, query='?mode=tooltip')
    hand = executor.Executor(page, watcher)
    observation = watcher.observe()
    result = hand.execute(observation, _click(_find(observation, 'button',
                                                    'Go')))
    assert result.executed, result
    assert result.approach_changed_page
    assert _status(watcher) == 'clicked'


def test_hover_that_disables_target_is_refused_at_press(open_page):
    page, watcher = open_page(_HOVER_PAGE, query='?mode=disable')
    hand = executor.Executor(page, watcher)
    observation = watcher.observe()
    result = hand.execute(observation, _click(_find(observation, 'button',
                                                    'Go')))
    assert result.refusal is executor.Refusal.DISABLED
    assert result.approach_changed_page
    assert result.pointer_path  # Refused after moving, before pressing.
    assert _status(watcher) == 'idle'


def test_hover_menu_covering_target_is_refused_at_press(open_page):
    page, watcher = open_page(_HOVER_PAGE, query='?mode=menu')
    hand = executor.Executor(page, watcher)
    observation = watcher.observe()
    result = hand.execute(observation, _click(_find(observation, 'button',
                                                    'Go')))
    assert result.refusal is executor.Refusal.OBSTRUCTED
    assert result.approach_changed_page
    assert _status(watcher) == 'idle'


def _move_count(status: str) -> int:
    return int(status.removeprefix('moves '))


def _run_click(open_page, speed: executor.Speed) -> tuple[
        executor.ActionResult, int]:
    """Clicks Target and returns the result and the mouse-move events the
    page received during the action. (Chromium also sends one move when
    the page loads, before the executor exists.)"""
    page, watcher = open_page(_MOVES_PAGE)
    hand = executor.Executor(page, watcher, speed)
    observation = watcher.observe()
    before = _move_count(_find_status(observation))
    result = hand.execute(observation, _click(_find(observation, 'button',
                                                    'Target')))
    return result, _move_count(_status(watcher)) - before


def test_paths_are_repeatable_and_send_real_mouse_moves(open_page):
    first, moves = _run_click(open_page, executor.NORMAL_SPEED)
    second, _ = _run_click(open_page, executor.NORMAL_SPEED)
    assert first.executed and second.executed
    assert ([(p.x, p.y) for p in first.pointer_path]
            == [(p.x, p.y) for p in second.pointer_path])
    assert moves == executor.NORMAL_SPEED.move_steps


def test_fast_mode_moves_in_fewer_steps_and_less_time(open_page):
    normal, normal_moves = _run_click(open_page, executor.NORMAL_SPEED)
    fast, fast_moves = _run_click(open_page, executor.FAST_SPEED)
    assert normal_moves == executor.NORMAL_SPEED.move_steps
    assert fast_moves == executor.FAST_SPEED.move_steps
    assert ((fast.pointer_path[-1].x, fast.pointer_path[-1].y)
            == (normal.pointer_path[-1].x, normal.pointer_path[-1].y))
    normal_travel = normal.pointer_path[-1].t_ms - normal.pointer_path[0].t_ms
    fast_travel = fast.pointer_path[-1].t_ms - fast.pointer_path[0].t_ms
    assert normal_travel >= executor.NORMAL_SPEED.move_ms * 0.9
    assert fast_travel < normal_travel


def test_result_is_json_ready(world):
    hand = executor.Executor(world.page, world.watcher)
    observation = world.watcher.observe()
    result = hand.execute(observation, _click(_find(observation, 'button',
                                                    'Next')))
    record = json.loads(json.dumps(result.to_dict()))
    assert record['executed'] is True
    assert record['refusal'] is None
    assert record['action']['kind'] == 'click'
    assert record['observation_sequence'] == observation.sequence
    assert len(record['pointer_path']) == len(result.pointer_path)
