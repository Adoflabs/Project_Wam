"""E0.3a-2: select, check and scroll in real Chromium, and their recording.

Targets come from `available_actions`, like a decider would choose them.
Outcomes are checked with the observer and, where relevant, the backend.
"""

# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

import pytest

from browser_world.interaction import executor
from browser_world.interaction import observer
from browser_world.collection import recorder

_GROUPED_PAGE = """<!doctype html>
<body style="margin:0;font:16px sans-serif">
  <label for="s">Plan</label>
  <select id="s"><option>Pick</option>
    <optgroup label="G"><option>Alpha</option><option disabled>Beta</option>
    </optgroup><option>Gamma</option></select>
  <div role="listbox" aria-label="Custom">
    <div role="option" tabindex="-1">Custom option</div></div>
</body>"""

_STUBBORN_PAGE = """<!doctype html>
<body><input id="x" aria-label="Changing">
<script>
  let n = 0;
  setInterval(() => { document.getElementById('x').value = String(n++); }, 1);
</script></body>"""


def _offer(observation, kind, name, checked=None):
    matches = [a for a in executor.available_actions(observation)
               if a.kind == kind and a.name == name
               and (checked is None or a.checked == checked)]
    assert len(matches) == 1, f'{len(matches)} affordances {kind} {name}'
    return matches[0]


def _do(world, hand, kind, name, text=None, checked=None):
    observation = world.watcher.observe()
    result = hand.execute(observation,
                          _offer(observation, kind, name, checked).action(text))
    assert result.executed, result
    world.watcher.wait_until_settled()
    return world.watcher.observe()


def _option_names(observation, dropdown_key):
    dropdown = observation.find(dropdown_key)
    return [e.name for e in observation.elements
            if e.role == 'option' and e.parent_id == dropdown.node_id]


def test_select_service_enables_dependent_doctors(world):
    hand = executor.Executor(world.page, world.watcher, executor.FAST_SPEED)
    before = world.watcher.observe()
    assert before.find('doctor').states['disabled']
    world.watcher.start_events()
    after = _do(world, hand, executor.SELECT, 'General check-up')
    trace = world.watcher.take_events()
    assert after.find('service').value == 'General check-up'
    assert not after.find('doctor').states['disabled']
    assert _option_names(after, 'doctor') == [
        'Select a doctor', 'Dr. Asha Patel', 'Dr. Daniel Kim']
    changes = [e for e in trace.events
               if e.kind == 'interaction' and e.detail['type'] == 'change']
    doctor_requests = [e for e in trace.events if e.kind == 'request'
                       and '/api/doctors' in e.detail['url']]
    assert len(changes) == 1 and len(doctor_requests) == 1  # One commit.


def test_doctor_date_and_time_chain(world):
    hand = executor.Executor(world.page, world.watcher, executor.FAST_SPEED)
    _do(world, hand, executor.SELECT, 'General check-up')
    after_doctor = _do(world, hand, executor.SELECT, 'Dr. Asha Patel')
    assert not after_doctor.find('date').states['disabled']
    after_date = _do(world, hand, executor.SELECT, 'Tue 24 Nov')
    slots = {e.name: e.states['disabled'] for e in after_date.elements
             if e.role == 'radio'}
    assert slots['09:30'] is True and slots['10:00'] is False
    after_time = _do(world, hand, executor.CHECK, '10:00', checked=True)
    assert after_time.find('slot-10:00').states['checked']


def test_select_skips_disabled_options_and_follows_optgroups(open_page):
    page, watcher = open_page(_GROUPED_PAGE)
    hand = executor.Executor(page, watcher, executor.FAST_SPEED)
    observation = watcher.observe()
    names = [a.name for a in executor.available_actions(observation)
             if a.kind == executor.SELECT]
    assert names == ['Pick', 'Alpha', 'Gamma']  # Beta is disabled.
    result = hand.execute(observation,
                          _offer(observation, 'select', 'Gamma').action())
    assert result.executed, result
    watcher.wait_until_settled()
    after = watcher.observe()
    assert next(e for e in after.elements
                if e.role == 'combobox').value == 'Gamma'


def test_select_refusals(open_page):
    page, watcher = open_page(_GROUPED_PAGE)
    hand = executor.Executor(page, watcher, executor.FAST_SPEED)
    observation = watcher.observe()
    beta = next(e for e in observation.elements if e.name == 'Beta')
    custom = next(e for e in observation.elements
                  if e.name == 'Custom option')
    assert hand.execute(observation, executor.Action(
        executor.SELECT, beta.node_id)).refusal is executor.Refusal.DISABLED
    assert hand.execute(observation, executor.Action(
        executor.SELECT, custom.node_id)).refusal is (
            executor.Refusal.UNSUPPORTED_ROLE)


def test_check_rules(world):
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', '09:00')
    world.click('next-appointment')
    hand = executor.Executor(world.page, world.watcher, executor.FAST_SPEED)
    after = _do(world, hand, executor.CHECK, 'I accept the privacy notice',
                checked=True)
    assert after.find('consent').states['checked']
    consent = after.find('consent')
    assert hand.execute(after, executor.Action(
        executor.CHECK, consent.node_id, checked=True)).refusal is (
            executor.Refusal.ALREADY_IN_STATE)
    after = _do(world, hand, executor.CHECK, 'I accept the privacy notice',
                checked=False)
    assert not after.find('consent').states['checked']
    with pytest.raises(AssertionError):  # Unchecking a radio is not offered.
        _offer(after, executor.CHECK, '09:00', checked=False)


def test_radio_cannot_be_unchecked(world):
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', '09:00')
    hand = executor.Executor(world.page, world.watcher, executor.FAST_SPEED)
    observation = world.watcher.observe()
    radio = observation.find('slot-09:00')
    result = hand.execute(observation, executor.Action(
        executor.CHECK, radio.node_id, checked=False))
    assert result.refusal is executor.Refusal.BAD_ACTION


def test_scroll_moves_the_page(make_world):
    world = make_world(viewport_height=300)
    hand = executor.Executor(world.page, world.watcher, executor.FAST_SPEED)
    before = world.watcher.observe()
    offer = _offer(before, executor.SCROLL, 'scroll down')
    assert offer.target is None and offer.scroll_y == 240.0
    result = hand.execute(before, offer.action())
    assert result.executed, result
    world.watcher.wait_until_settled()
    after = world.watcher.observe()
    assert after.motion.scroll_y > before.motion.scroll_y
    assert _offer(after, executor.SCROLL, 'scroll up').scroll_y == -240.0


@pytest.mark.parametrize('action', [
    executor.Action(executor.SCROLL, 123, scroll_y=100.0),
    executor.Action(executor.SCROLL, scroll_y=0.0),
    executor.Action(executor.SCROLL, scroll_y=float('nan')),
    executor.Action(executor.CLICK),
    executor.Action(executor.CHECK, 1),
])
def test_malformed_actions_are_refused_before_acting(world, action):
    hand = executor.Executor(world.page, world.watcher, executor.FAST_SPEED)
    result = hand.execute(world.watcher.observe(), action)
    assert result.refusal is executor.Refusal.BAD_ACTION
    assert not result.pointer_path


def test_scroll_is_recorded_without_an_element_target(make_world, tmp_path):
    world = make_world(viewport_height=300)
    hand = executor.Executor(world.page, world.watcher, executor.FAST_SPEED)
    rec = recorder.Recorder(world.page, world.watcher, hand, tmp_path)
    before = world.watcher.observe()
    t = rec.record_step(rec.new_run('ep-scroll', 'start'), before,
                        _offer(before, executor.SCROLL, 'scroll down').action())
    assert isinstance(t, recorder.Transition), t
    assert t.target is None
    assert t.after.motion.scroll_y > 0
    assert recorder.model_input(t)['action'] == {
        'kind': 'scroll', 'target_index': None, 'text': None,
        'checked': None, 'scroll_y': 240.0}
    saved = recorder.read_jsonl(rec.transitions_path)[0]
    assert saved['target'] is None and saved['events_status']['complete']


def test_failed_write_invalidates_the_run(world, tmp_path, monkeypatch):
    hand = executor.Executor(world.page, world.watcher, executor.FAST_SPEED)
    rec = recorder.Recorder(world.page, world.watcher, hand, tmp_path)
    run_id = rec.new_run('ep-write', 'start')
    original = rec._append  # pylint: disable=protected-access

    def failing_append(path, record):
        del path, record  # Unused.
        raise OSError('disk full')

    monkeypatch.setattr(rec, '_append', failing_append)
    before = world.watcher.observe()
    with pytest.raises(OSError):
        rec.record_step(run_id, before,
                        _offer(before, executor.CLICK, 'Next').action())
    monkeypatch.setattr(rec, '_append', original)
    world.watcher.wait_until_settled()
    before = world.watcher.observe()
    with pytest.raises(recorder.InvalidRunError):  # The click happened.
        rec.record_step(run_id, before,
                        _offer(before, executor.CLICK, 'Next').action())
    fresh = rec.new_run('ep-write', 'reset')
    t = rec.record_step(fresh, before,
                        _offer(before, executor.CLICK, 'Next').action())
    assert t.step == 0 and t.run_origin == 'reset'
    assert not (tmp_path / recorder.TRANSITIONS_FILE).read_text().count(
        run_id)  # Nothing from the invalid run was written.


def test_capture_is_marked_inconsistent_when_the_page_keeps_changing(
        open_page):
    _, watcher = open_page(_STUBBORN_PAGE)
    assert not watcher.observe().consistent


def test_stable_page_is_captured_consistently(world):
    assert world.watcher.observe().consistent


def test_unrecorded_reload_breaks_continuity(world, tmp_path):
    hand = executor.Executor(world.page, world.watcher, executor.FAST_SPEED)
    rec = recorder.Recorder(world.page, world.watcher, hand, tmp_path,
                            truth_fn=world.truth_bookings)
    run_id = rec.new_run('ep-forgot', 'start')
    before = world.watcher.observe()
    rec.record_step(run_id, before, _offer(before, 'click', 'Next').action())
    world.page.reload()  # The caller forgets new_run().
    world.page.wait_for_selector('body[data-ready=true]')
    world.watcher.wait_until_settled()
    before = world.watcher.observe()
    t = rec.record_step(run_id, before,
                        _offer(before, 'click', 'Next').action())
    assert not t.continuity.continuous
    assert any('page epoch changed' in d for d in t.continuity.differences)


def test_backend_change_between_steps_breaks_continuity(world, tmp_path):
    hand = executor.Executor(world.page, world.watcher, executor.FAST_SPEED)
    rec = recorder.Recorder(world.page, world.watcher, hand, tmp_path,
                            truth_fn=world.truth_bookings)
    run_id = rec.new_run('ep-backend', 'start')
    before = world.watcher.observe()
    rec.record_step(run_id, before, _offer(before, 'click', 'Next').action())
    world.page.request.post(  # Someone books behind the page's back.
        f'{world.base_url}/api/bookings',
        headers={'X-WAM-Session': world.session_id},
        data={'service': 'eye', 'doctor': 'dr-patel', 'date': '2026-11-25',
              'slot': '09:00', 'name': 'Other Patient',
              'email': 'other@example.com', 'phone': '5550000000',
              'consent': 'true'})
    before = world.watcher.observe()
    t = rec.record_step(run_id, before,
                        _offer(before, 'click', 'Next').action())
    assert not t.continuity.continuous
    assert t.continuity.differences == (
        'backend truth changed outside a recorded action',)


def test_observation_round_trip_keeps_consistency(world):
    observation = world.watcher.observe()
    rebuilt = observer.observation_from_dict(observation.to_dict())
    assert rebuilt == observation


# ARIA 1.0 combobox pattern: Chromium exposes the options under the custom
# combobox (option -> listbox -> combobox), so only the tag shows it is not
# a native <select>.
_ARIA_COMBOBOX_PAGE = """<!doctype html>
<body>
  <div role="combobox" aria-expanded="true" aria-owns="lb"
       aria-label="Fake dropdown">
    <input aria-label="Type to filter">
    <div id="lb" role="listbox">
      <div role="option" tabindex="-1">One</div>
      <div role="option" tabindex="-1">Two</div>
    </div>
  </div>
</body>"""


def test_aria_combobox_is_not_treated_as_native(open_page):
    page, watcher = open_page(_ARIA_COMBOBOX_PAGE)
    hand = executor.Executor(page, watcher, executor.FAST_SPEED)
    observation = watcher.observe()
    fake = next(e for e in observation.elements if e.role == 'combobox')
    assert fake.tag == 'div'
    assert not [a for a in executor.available_actions(observation)
                if a.kind == executor.SELECT]
    one = next(e for e in observation.elements if e.name == 'One')
    assert hand.execute(observation, executor.Action(
        executor.SELECT, one.node_id)).refusal is (
            executor.Refusal.UNSUPPORTED_ROLE)


def test_native_select_records_its_tag(world):
    assert world.watcher.observe().find('service').tag == 'select'


def test_missing_page_script_is_never_consistent(new_page):
    page = new_page()
    page.set_content('<button>Plain</button>')
    watcher = observer.attach(page)  # Too late: the script never ran here.
    assert page.evaluate('typeof window.__wamTemporal') == 'undefined'
    assert not watcher.observe().consistent


def test_older_schemas_stay_out_of_training():
    assert recorder.sequence_problems({'schema_version': 2}) == [
        'schema 2 lacks capture-consistency and native-control fields']
