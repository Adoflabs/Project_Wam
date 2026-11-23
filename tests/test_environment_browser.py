"""E0.1 acceptance tests: each required behaviour, driven through Chromium.

Outcomes are checked twice: in what the page shows (through the observer)
and in backend ground truth.
"""

# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

from browser_world import booking

_VALID = ('Sam Rivera', 'sam.rivera@example.com', '555-010-2030')


def _names(observation, role):
    return [e.name for e in observation.elements if e.role == role]


def test_service_selection_updates_doctors(world):
    observation = world.act(
        lambda: world.page.select_option('#service', 'dental'))
    doctor = observation.find('doctor')
    assert not doctor.states['disabled']
    options = [e.name for e in observation.elements
               if e.role == 'option' and e.parent_id == doctor.node_id]
    assert options == ['Select a doctor', 'Dr. Ngozi Okafor']


def test_date_selection_updates_slot_availability(world):
    observation = world.choose_appointment('checkup', 'dr-patel',
                                           '2026-11-24', slot=None)
    slots = {e.name: e.states['disabled'] for e in observation.elements
             if e.role == 'radio'}
    assert slots['09:30'] is True  # Prebooked.
    assert slots['09:00'] is False
    assert observation.find('slot-status').name == '6 times available.'


def test_loading_state_is_observable(world):
    world.act(lambda: world.page.select_option('#service', 'checkup'))
    world.act(lambda: world.page.select_option('#doctor', 'dr-patel'))
    world.page.select_option('#date', '2026-11-24')
    during = world.watcher.observe()  # Before the slow request returns.
    assert during.find('slot-status').name == 'Loading times…'
    assert during.temporal.pending_requests == 1
    world.watcher.wait_until_settled()
    after = world.watcher.observe()
    assert after.find('slot-status').name == '6 times available.'


def test_missing_time_shows_error_and_stays_on_step(world):
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', slot=None)
    observation = world.click('next-appointment')
    assert observation.find('error-slot').name == booking.MSG_SLOT_REQUIRED
    assert observation.find('slot-group').states['invalid']
    assert observation.find('step-indicator').name == 'Step 1 of 3'


def test_invalid_email_shows_error_and_creates_no_booking(world):
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', '09:00')
    world.click('next-appointment')
    world.fill_details(_VALID[0], 'sam@x', _VALID[2], consent=True)
    observation = world.click('next-details')
    assert observation.find('error-email').name == booking.MSG_EMAIL_INVALID
    assert observation.find('email').states['invalid']
    assert observation.find('error-name') is None
    assert observation.find('step-indicator').name == 'Step 2 of 3'
    assert not world.truth_bookings()


def test_valid_details_and_confirm_create_booking(world):
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', '09:00')
    world.click('next-appointment')
    world.fill_details(*_VALID, consent=True)
    review = world.click('next-details')
    assert 'Time: 09:00' in _names(review, 'listitem')
    assert not world.truth_bookings()  # Reviewing books nothing.

    done = world.click('confirm')
    assert done.find('step-indicator').name == 'Booking complete'
    assert done.find('reference').name == 'Your reference is BK-0001.'
    truth = world.truth_bookings()
    assert len(truth) == 1
    assert truth[0]['slot'] == '09:00'
    assert truth[0]['email'] == _VALID[1]


def test_booked_slot_is_unavailable_next_time(world):
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', '09:00')
    world.click('next-appointment')
    world.fill_details(*_VALID, consent=True)
    world.click('next-details')
    world.click('confirm')
    world.click('book-another')
    observation = world.choose_appointment('checkup', 'dr-patel',
                                           '2026-11-24', slot=None)
    slot = next(e for e in observation.elements
                if e.role == 'radio' and e.name == '09:00')
    assert slot.states['disabled']


def test_contested_slot_conflict(world):
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', '11:00')
    world.click('next-appointment')
    world.fill_details(*_VALID, consent=True)
    world.click('next-details')
    observation = world.click('confirm')
    assert observation.find('error-confirm').name == booking.MSG_SLOT_TAKEN
    assert observation.find('choose-another') is not None
    assert observation.find('confirm') is None
    assert not world.truth_bookings()

    observation = world.click('choose-another')
    slot = next(e for e in observation.elements
                if e.role == 'radio' and e.name == '11:00')
    assert slot.states['disabled']
    assert observation.find('step-indicator').name == 'Step 1 of 3'


def test_reset_restores_original_availability(world):
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', '09:00')
    world.click('next-appointment')
    world.fill_details(*_VALID, consent=True)
    world.click('next-details')
    world.click('confirm')
    world.reset()
    assert not world.truth_bookings()
    world.page.reload()
    world.page.wait_for_selector('body[data-ready=true]')
    observation = world.choose_appointment('checkup', 'dr-patel',
                                           '2026-11-24', slot=None)
    slot = next(e for e in observation.elements
                if e.role == 'radio' and e.name == '09:00')
    assert not slot.states['disabled']


def test_back_and_next_preserve_entered_values(world):
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', '10:00')
    world.click('next-appointment')
    world.fill_details(*_VALID, consent=True)
    back = world.click('back-details')
    assert back.find('service').value == 'General check-up'
    assert back.find('slot-10:00').states['checked']
    assert back.find('name') is None  # Hidden steps are not observed.

    forward = world.click('next-appointment')
    assert forward.find('name').value == _VALID[0]
    assert forward.find('email').value == _VALID[1]
    assert forward.find('consent').states['checked']


def test_enter_key_submits_step(world):
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', '09:00')
    world.click('next-appointment')
    world.act(lambda: world.page.fill('#name', _VALID[0]))
    observation = world.act(lambda: world.page.press('#name', 'Enter'))
    assert observation.find('error-email').name == booking.MSG_EMAIL_REQUIRED
