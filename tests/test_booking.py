# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

import datetime

import pytest

from browser_world.environment import booking

_VALID_DETAILS = {
    'name': 'Sam Rivera',
    'email': 'sam.rivera@example.com',
    'phone': '555-010-2030',
    'consent': 'true',
}
_FREE_SLOT = {
    'service': 'checkup',
    'doctor': 'dr-patel',
    'date': '2026-11-24',
    'slot': '09:00',
}


@pytest.fixture
def session() -> booking.BookingSession:
    return booking.BookingSession(booking.default_scenario())


def test_doctors_depend_on_service(session):
    assert [d.key for d in session.doctors('checkup')] == ['dr-patel',
                                                           'dr-kim']
    assert [d.key for d in session.doctors('dental')] == ['dr-okafor']
    assert not session.doctors('unknown')


def test_dates_follow_doctor_weekdays_and_exclude_today(session):
    scenario = session.scenario
    for doctor in scenario.doctors:
        dates = session.dates(doctor.key)
        assert dates
        for day in dates:
            assert day > scenario.today
            assert day <= scenario.today + datetime.timedelta(
                days=scenario.days_ahead)
            assert day.weekday() in doctor.weekdays


def test_slots_mark_prebooked_as_unavailable(session):
    slots = {s.time: s.available for s in session.slots('dr-patel',
                                                        '2026-11-24')}
    assert slots['09:30'] is False
    assert slots['14:00'] is False
    assert slots['09:00'] is True


def test_slots_empty_on_non_working_day(session):
    assert not session.slots('dr-kim', '2026-11-30')  # A Monday.


@pytest.mark.parametrize('email, valid', [
    ('a@b.co', True),
    ('sam.rivera@example.com', True),
    ('sam@x', False),
    ('sam.example.com', False),
    ('sam @example.com', False),
])
def test_email_validation(email, valid):
    assert booking.is_valid_email(email) is valid


@pytest.mark.parametrize('phone, valid', [
    ('5550102030', True),
    ('(555) 010-2030', True),
    ('555-010-203', False),
    ('555-010-2030x', False),
])
def test_phone_validation(phone, valid):
    assert booking.is_valid_phone(phone) is valid


def test_validate_appointment_reports_first_missing_choice(session):
    assert session.validate_appointment({}) == {
        'service': booking.MSG_SERVICE_REQUIRED}
    assert session.validate_appointment({'service': 'checkup'}) == {
        'doctor': booking.MSG_DOCTOR_REQUIRED}
    request = dict(_FREE_SLOT, slot='')
    assert session.validate_appointment(request) == {
        'slot': booking.MSG_SLOT_REQUIRED}
    request = dict(_FREE_SLOT, slot='09:30')
    assert session.validate_appointment(request) == {
        'slot': booking.MSG_SLOT_UNAVAILABLE}
    assert session.validate_appointment(_FREE_SLOT) == {}


def test_validate_details_reports_every_invalid_field(session):
    errors = session.validate_details(
        {'name': 'S', 'email': 'sam@x', 'phone': '', 'consent': 'false'})
    assert errors == {
        'name': booking.MSG_NAME_REQUIRED,
        'email': booking.MSG_EMAIL_INVALID,
        'phone': booking.MSG_PHONE_REQUIRED,
        'consent': booking.MSG_CONSENT_REQUIRED,
    }
    assert session.validate_details(_VALID_DETAILS) == {}


def test_book_creates_booking_and_takes_slot(session):
    created = session.book({**_FREE_SLOT, **_VALID_DETAILS})
    assert created.reference == 'BK-0001'
    assert session.bookings == (created,)
    slots = {s.time: s.available for s in session.slots('dr-patel',
                                                        '2026-11-24')}
    assert slots['09:00'] is False


def test_book_rejects_invalid_details_without_booking(session):
    with pytest.raises(booking.ValidationError) as excinfo:
        session.book({**_FREE_SLOT, **_VALID_DETAILS, 'email': 'bad'})
    assert excinfo.value.errors == {'email': booking.MSG_EMAIL_INVALID}
    assert not session.bookings


def test_booking_same_slot_twice_is_rejected(session):
    session.book({**_FREE_SLOT, **_VALID_DETAILS})
    with pytest.raises(booking.ValidationError) as excinfo:
        session.book({**_FREE_SLOT, **_VALID_DETAILS})
    assert excinfo.value.errors == {'slot': booking.MSG_SLOT_UNAVAILABLE}
    assert len(session.bookings) == 1


def test_contested_slot_is_taken_on_first_confirm(session):
    request = {**_FREE_SLOT, **_VALID_DETAILS, 'slot': '11:00'}
    slots = {s.time: s.available for s in session.slots('dr-patel',
                                                        '2026-11-24')}
    assert slots['11:00'] is True  # Looks free beforehand.
    with pytest.raises(booking.SlotTakenError):
        session.book(request)
    assert not session.bookings
    slots = {s.time: s.available for s in session.slots('dr-patel',
                                                        '2026-11-24')}
    assert slots['11:00'] is False


def test_reset_restores_initial_state(session):
    initial = session.slots('dr-patel', '2026-11-24')
    session.book({**_FREE_SLOT, **_VALID_DETAILS})
    with pytest.raises(booking.SlotTakenError):
        session.book({**_FREE_SLOT, **_VALID_DETAILS, 'slot': '11:00'})
    session.reset()
    assert session.slots('dr-patel', '2026-11-24') == initial
    assert not session.bookings
    with pytest.raises(booking.SlotTakenError):  # The race happens again.
        session.book({**_FREE_SLOT, **_VALID_DETAILS, 'slot': '11:00'})


def test_sessions_are_isolated():
    registry = booking.SessionRegistry(booking.default_scenario())
    registry.get('a').book({**_FREE_SLOT, **_VALID_DETAILS})
    assert len(registry.get('a').bookings) == 1
    assert not registry.get('b').bookings
    with pytest.raises(ValueError):
        registry.get('')
