"""E0.3b-1: recording real transitions from the booking site in Chromium.

Targets are chosen like a decider would, from the observation by role and
name; replays choose again the same way, never by node id or selector.
"""

# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

import json
import pathlib

import pytest

from browser_world import booking
from browser_world import executor
from browser_world import observer
from browser_world import recorder


def _choose(observation: observer.Observation, kind: str,
            name: str, text: str | None = None) -> executor.Action:
    """Picks the single matching affordance, like a decider."""
    matches = [a for a in executor.available_actions(observation)
               if a.kind == kind and a.name == name]
    assert len(matches) == 1, f'{len(matches)} affordances {kind} {name}'
    return executor.Action(kind, matches[0].target, text)


def _make_recorder(world, raw_dir: pathlib.Path,
                   capture_baselines: bool = False,
                   hand: executor.Executor | None = None
                   ) -> recorder.Recorder:
    hand = hand or executor.Executor(world.page, world.watcher,
                                     executor.FAST_SPEED)
    return recorder.Recorder(world.page, world.watcher, hand, raw_dir,
                             truth_fn=world.truth_bookings,
                             capture_baselines=capture_baselines)


@pytest.fixture
def next_transition(world, tmp_path) -> recorder.Transition:
    """One recorded 'click Next with nothing chosen' transition."""
    rec = _make_recorder(world, tmp_path)
    before = world.watcher.observe()
    outcome = rec.record_step(rec.new_run('ep-next', 'start'), before,
                              _choose(before, executor.CLICK, 'Next'))
    assert isinstance(outcome, recorder.Transition), outcome
    return outcome


def test_click_next_records_validation_transition(next_transition, tmp_path):
    t = next_transition
    assert t.schema_version == recorder.SCHEMA_VERSION
    assert (t.episode_id, t.step, t.run_origin) == ('ep-next', 0, 'start')
    assert t.run_id.startswith('ep-next/run-')
    assert t.continuity == recorder.Continuity(None, True, ())
    assert t.events.complete and t.events.dropped == 0
    assert (t.target.role, t.target.name) == ('button', 'Next')
    assert t.before.elements[t.target.index].node_id == t.action.target
    assert t.result.executed and t.settled
    assert t.before.find('error-service') is None
    assert t.after.find('error-service').name == booking.MSG_SERVICE_REQUIRED
    assert t.after.find('service').states['invalid']
    assert t.truth == {'before': [], 'after': []}  # No booking was made.
    assert len(recorder.read_jsonl(tmp_path / recorder.TRANSITIONS_FILE)) == 1
    assert not (tmp_path / recorder.REFUSALS_FILE).exists()


def test_typing_records_a_value_change(world, tmp_path):
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', '09:00')
    world.click('next-appointment')
    rec = _make_recorder(world, tmp_path)
    before = world.watcher.observe()
    action = _choose(before, executor.TYPE, 'Email address',
                     'sam@example.com')
    t = rec.record_step(rec.new_run('ep-type', 'start'), before, action)
    assert isinstance(t, recorder.Transition), t
    assert t.before.elements[t.target.index].value == ''
    assert t.after.find('email').value == 'sam@example.com'
    assert t.action.text == 'sam@example.com'


def test_events_are_timestamped_within_the_window(next_transition):
    t = next_transition
    times = [e.t_ms for e in t.events.events]
    assert times == sorted(times)
    kinds = {e.kind for e in t.events.events}
    assert 'mutation' in kinds  # The error message appeared.
    requests = [e for e in t.events.events if e.kind == 'request']
    validate = [e for e in requests
                if '/api/validate/appointment' in e.detail['url']]
    assert len(validate) == 1
    assert validate[0].detail['initiator'] == 'fetch'
    assert validate[0].detail['end_ms'] >= validate[0].t_ms
    # Converting the page clock costs at most a few ms of error.
    for event in t.events.events:
        assert -5 <= event.t_ms <= t.window_ms + 5, event
    assert t.result.pointer_path  # The mouse path is part of the record.


def test_refusal_is_logged_separately(world, tmp_path):
    rec = _make_recorder(world, tmp_path)
    before = world.watcher.observe()
    outcome = rec.record_step(rec.new_run('ep-refuse', 'start'), before,
                              executor.Action(executor.CLICK, 999_999))
    assert isinstance(outcome, executor.ActionResult)
    assert outcome.refusal is executor.Refusal.UNKNOWN_TARGET
    assert not (tmp_path / recorder.TRANSITIONS_FILE).exists()
    refusals = recorder.read_jsonl(tmp_path / recorder.REFUSALS_FILE)
    assert len(refusals) == 1
    assert refusals[0]['result']['refusal'] == 'unknown_target'


def test_reset_reload_and_replay_reproduce_the_transition(world, tmp_path):
    hand = executor.Executor(world.page, world.watcher, executor.FAST_SPEED)
    rec = _make_recorder(world, tmp_path, hand=hand)
    runs = []
    for attempt in range(2):
        if attempt:
            world.reset()
            world.page.reload()
            world.page.wait_for_selector('body[data-ready=true]')
            world.watcher.wait_until_settled()
        before = world.watcher.observe()
        run_id = rec.new_run('ep-replay', 'reset' if attempt else 'start')
        t = rec.record_step(run_id, before,
                            _choose(before, executor.CLICK, 'Next'))
        assert isinstance(t, recorder.Transition), t
        runs.append(t)
    assert len(runs) == 2
    first, second = runs  # pylint: disable=unbalanced-tuple-unpacking  # Built above.
    assert first.run_id != second.run_id  # A reset starts a separate run.
    assert (first.step, second.step) == (0, 0)
    assert first.before.page_epoch != second.before.page_epoch  # Reloaded.
    assert first.action.target != second.action.target  # New node ids.
    assert not recorder.differences(recorder.canonical(first.before),
                                    recorder.canonical(second.before))
    assert not recorder.differences(recorder.canonical(first.after),
                                    recorder.canonical(second.after))
    # The fingerprint is sensitive: before and after do differ.
    assert recorder.differences(recorder.canonical(first.before),
                                recorder.canonical(first.after))


def test_saved_record_reads_back_without_loss(next_transition, tmp_path):
    saved = recorder.read_jsonl(tmp_path / recorder.TRANSITIONS_FILE)[0]
    assert saved == json.loads(json.dumps(next_transition.to_dict()))
    assert observer.observation_from_dict(saved['before']) == (
        next_transition.before)
    assert observer.observation_from_dict(saved['after']) == (
        next_transition.after)


def test_baseline_artifacts_are_off_by_default_and_linked_when_on(
        next_transition, make_world, tmp_path):
    assert next_transition.artifacts == {}
    assert not (tmp_path / 'artifacts').exists()

    world = make_world()
    raw_dir = tmp_path / 'with-baselines'
    rec = _make_recorder(world, raw_dir, capture_baselines=True)
    before = world.watcher.observe()
    t = rec.record_step(rec.new_run('ep-art', 'start'), before,
                        _choose(before, executor.CLICK, 'Next'))
    # Capturing must not change the page, or the action would be refused.
    assert isinstance(t, recorder.Transition), t
    assert set(t.artifacts) == {'screenshot_before', 'html_before',
                                'screenshot_after', 'html_after'}
    png = (raw_dir / t.artifacts['screenshot_after']).read_bytes()
    assert png.startswith(b'\x89PNG')
    html = (raw_dir / t.artifacts['html_after']).read_text(encoding='utf-8')
    assert booking.MSG_SERVICE_REQUIRED in html
    assert 'screenshot' not in json.dumps(recorder.model_input(t))


def _all_keys(value) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for item in value.values():
            keys |= _all_keys(item)
        return keys
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()


def test_model_input_cannot_leak_privileged_data(next_transition):
    t = next_transition
    projected = recorder.model_input(t)
    assert set(projected) == {'state', 'action'}
    keys = _all_keys(projected)
    for forbidden in ('node_id', 'match_key', 'target', 'truth', 'after',
                      'events', 'result', 'artifacts', 'temporal',
                      'sequence', 'page_epoch', 'page_version',
                      'state_digest', 'episode_id', 'run_id', 'continuity'):
        assert forbidden not in keys, forbidden
    text = json.dumps(projected)
    match_keys = {e.match_key for e in t.before.elements if e.match_key}
    for key in match_keys:
        assert f'"{key}"' not in text, key
    assert '?session=' not in text
    assert booking.MSG_SERVICE_REQUIRED not in text  # Future outcome.
    assert projected['action'] == {'kind': 'click',
                                   'target_index': t.target.index,
                                   'text': None, 'checked': None,
                                   'scroll_y': None}
