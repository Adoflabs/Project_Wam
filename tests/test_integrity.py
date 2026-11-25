"""Regression tests for the four data-integrity defects found by the audit.

1. URL fingerprint dropped meaningful query parameters.
2. Independent runs could be joined into a fake continuous episode.
3. Freshness missed changes that do not mutate the DOM.
4. Event traces could lose events without saying so.
"""

# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

import pytest

from browser_world.interaction import executor
from browser_world.collection import recorder

_FRESH_PAGE = """<!doctype html>
<body style="margin:0;font:16px sans-serif">
  <input id="email" aria-label="Email">
  <input type="checkbox" id="agree"><label for="agree">I agree</label>
  <select id="svc" aria-label="Service">
    <option>Pick</option><option>Check-up</option></select>
  <button id="save">Save</button>
  <div id="box" style="height:60px;width:200px;overflow:auto">
    <div style="height:300px">tall</div></div>
  <p role="status" id="s">idle</p>
  <script>
    window.burst = (n) => new Promise((resolve) => {
      const ch = new MessageChannel(); let i = 0;
      ch.port1.onmessage = () => {
        document.getElementById('s').setAttribute('data-i', String(i));
        i += 1;
        if (i < n) ch.port2.postMessage(0); else resolve(i);
      };
      ch.port2.postMessage(0);
    });
  </script>
</body>"""

_PARAM_PAGE = """<!doctype html><title>Same page</title>
<h1>Doctor profile</h1><button>Book</button>"""


def _choose(observation, kind, name, text=None):
    matches = [a for a in executor.available_actions(observation)
               if a.kind == kind and a.name == name]
    assert len(matches) == 1, f'{len(matches)} affordances {kind} {name}'
    return executor.Action(kind, matches[0].target, text)


def _make_recorder(world, raw_dir):
    hand = executor.Executor(world.page, world.watcher, executor.FAST_SPEED)
    return recorder.Recorder(world.page, world.watcher, hand, raw_dir,
                             truth_fn=world.truth_bookings)


# 1. URL fingerprint.

def test_stable_url_removes_only_the_session_parameter():
    assert (recorder.stable_url(
        'http://h/book?session=abc&doctor=patel&date=2026-11-24#slot')
            == 'http://h/book?doctor=patel&date=2026-11-24#slot')
    assert recorder.stable_url('http://h/?session=abc') == 'http://h/'


@pytest.mark.parametrize('first, second', [
    ('?session=a&doctor=patel', '?session=a&doctor=singh'),
    ('?session=a&date=2026-11-24', '?session=a&date=2026-11-25'),
    ('?session=a&slot=09:00', '?session=a&slot=10:00'),
])
def test_meaningful_parameters_change_the_fingerprint(open_page, first,
                                                      second):
    _, one = open_page(_PARAM_PAGE, query=first)
    _, two = open_page(_PARAM_PAGE, query=second)
    found = recorder.differences(recorder.canonical(one.observe()),
                                 recorder.canonical(two.observe()))
    assert any(d.startswith('url:') for d in found), found


def test_session_alone_does_not_change_the_fingerprint(open_page):
    _, one = open_page(_PARAM_PAGE, query='?session=a&doctor=patel')
    _, two = open_page(_PARAM_PAGE, query='?session=b&doctor=patel')
    assert not recorder.differences(recorder.canonical(one.observe()),
                                    recorder.canonical(two.observe()))


# 2. Episode continuity.

def test_consecutive_steps_in_one_run_join(world, tmp_path):
    rec = _make_recorder(world, tmp_path)
    run_id = rec.new_run('ep-join', 'start')
    before = world.watcher.observe()
    first = rec.record_step(run_id, before, _choose(before, 'click', 'Next'))
    before = world.watcher.observe()
    second = rec.record_step(run_id, before, _choose(before, 'click', 'Next'))
    assert (first.step, second.step) == (0, 1)
    assert second.continuity == recorder.Continuity(0, True, ())
    saved = recorder.read_jsonl(rec.transitions_path)
    assert [recorder.sequence_problems(r) for r in saved] == [[], []]


def test_change_between_steps_is_recorded_as_discontinuous(world, tmp_path):
    rec = _make_recorder(world, tmp_path)
    run_id = rec.new_run('ep-break', 'start')
    before = world.watcher.observe()
    rec.record_step(run_id, before, _choose(before, 'click', 'Next'))
    world.act(lambda: world.page.select_option('#service', 'checkup'))
    before = world.watcher.observe()
    second = rec.record_step(run_id, before, _choose(before, 'click', 'Next'))
    assert second.step == 1
    assert not second.continuity.continuous
    assert second.continuity.differences  # Says what differs.
    saved = recorder.read_jsonl(rec.transitions_path)
    assert len(saved) == 2  # Kept as evidence, never silently joined.
    assert recorder.sequence_problems(saved[1]) == [
        "before-state does not match the previous step's after-state"]


def test_reset_starts_a_separate_run(world, tmp_path):
    rec = _make_recorder(world, tmp_path)
    first_run = rec.new_run('ep-reset', 'start')
    before = world.watcher.observe()
    rec.record_step(first_run, before, _choose(before, 'click', 'Next'))
    world.reset()
    world.page.reload()
    world.page.wait_for_selector('body[data-ready=true]')
    world.watcher.wait_until_settled()
    second_run = rec.new_run('ep-reset', 'reset')
    before = world.watcher.observe()
    t = rec.record_step(second_run, before, _choose(before, 'click', 'Next'))
    assert second_run != first_run
    assert (t.step, t.run_origin, t.continuity.previous_step) == (
        0, 'reset', None)


def test_refusal_does_not_use_a_step_and_unknown_runs_fail(world, tmp_path):
    rec = _make_recorder(world, tmp_path)
    run_id = rec.new_run('ep-refuse', 'start')
    before = world.watcher.observe()
    refused = rec.record_step(run_id, before,
                              executor.Action(executor.CLICK, 999_999))
    assert isinstance(refused, executor.ActionResult)
    before = world.watcher.observe()
    t = rec.record_step(run_id, before, _choose(before, 'click', 'Next'))
    assert t.step == 0
    with pytest.raises(ValueError):
        rec.record_step('no-such-run', before,
                        _choose(before, 'click', 'Next'))


def test_legacy_records_are_excluded_from_sequences():
    assert recorder.sequence_problems({'schema_version': 1}) == [
        'legacy record without run identity or continuity check']


# 3. Freshness.

@pytest.mark.parametrize('change', [
    "document.getElementById('email').value = 'x@y.z'",
    "document.getElementById('agree').checked = true",
    "document.getElementById('svc').selectedIndex = 1",
    "document.getElementById('save').focus()",
    "document.getElementById('box').scrollTop = 100",
])
def test_silent_state_changes_make_an_observation_stale(open_page, change):
    page, watcher = open_page(_FRESH_PAGE)
    observation = watcher.observe()
    assert watcher.is_current(observation)
    page.evaluate(change)
    assert not watcher.is_current(observation)


def test_unchanged_page_stays_current(open_page):
    _, watcher = open_page(_FRESH_PAGE)
    observation = watcher.observe()
    for _ in range(3):
        assert watcher.is_current(observation)


def test_typing_changes_the_page_version_and_is_traced(open_page):
    page, watcher = open_page(_FRESH_PAGE)
    page.focus('#email')
    watcher.wait_until_settled()
    observation = watcher.observe()
    watcher.start_events()
    page.keyboard.type('ab')
    trace = watcher.take_events()
    assert watcher.current_version()[1] > observation.page_version
    types = [e.detail['type'] for e in trace.events
             if e.kind == 'interaction']
    assert types.count('input') == 2


def test_executor_refuses_after_a_silent_change(open_page):
    page, watcher = open_page(_FRESH_PAGE)
    hand = executor.Executor(page, watcher, executor.FAST_SPEED)
    observation = watcher.observe()
    save = next(e for e in observation.elements if e.name == 'Save')
    page.evaluate("document.getElementById('email').value = 'changed'")
    result = hand.execute(observation, executor.Action(executor.CLICK,
                                                       save.node_id))
    assert result.refusal is executor.Refusal.STALE_OBSERVATION


# 4. Event completeness.

def test_quiet_window_is_complete(open_page):
    page, watcher = open_page(_FRESH_PAGE)
    watcher.start_events()
    page.evaluate("document.getElementById('s').textContent = 'busy'")
    watcher.wait_until_settled()
    trace = watcher.take_events()
    assert trace.status() == {'complete': True, 'dropped': 0,
                              'incomplete_reasons': []}


def test_event_cap_is_reported_with_the_dropped_count(open_page):
    page, watcher = open_page(_FRESH_PAGE)
    watcher.start_events()
    sent = page.evaluate('burst(6000)')
    watcher.wait_until_settled()
    trace = watcher.take_events()
    mutations = [e for e in trace.events if e.kind == 'mutation']
    assert not trace.complete
    assert trace.incomplete_reasons == ('cap_reached',)
    assert len(trace.events) == 5000
    assert len(mutations) + trace.dropped == sent


def test_navigation_marks_the_trace_incomplete(open_page):
    page, watcher = open_page(_FRESH_PAGE)
    watcher.start_events()
    page.evaluate("document.getElementById('s').textContent = 'gone'")
    page.reload()
    watcher.wait_until_settled()
    trace = watcher.take_events()
    assert not trace.complete
    assert 'navigation' in trace.incomplete_reasons
    assert trace.dropped is None  # The old document's losses are unknown.


def test_transition_record_carries_event_status(world, tmp_path):
    rec = _make_recorder(world, tmp_path)
    before = world.watcher.observe()
    rec.record_step(rec.new_run('ep-status', 'start'), before,
                    _choose(before, 'click', 'Next'))
    saved = recorder.read_jsonl(rec.transitions_path)[0]
    assert saved['events_status'] == {'complete': True, 'dropped': 0,
                                      'incomplete_reasons': []}
