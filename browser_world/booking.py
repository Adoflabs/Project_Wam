"""Appointment-booking rules and per-session state for the browser world.

This module is the environment's ground truth. It knows nothing about HTTP
or browsers: the server exposes it over an API, the web page renders it, and
the observer only ever sees what the page shows.

Determinism: every session's behaviour is a pure function of its `Scenario`
and the requests it has received. The scenario's `today` replaces the wall
clock, so the same scenario and request sequence always produce the same
results.

Typical usage example:

  registry = booking.SessionRegistry(booking.default_scenario())
  session = registry.get('worker-0')
  slots = session.slots('dr-patel', '2026-11-24')
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
import datetime
import re
import threading

# Validation messages are part of the observable world, so they are fixed
# strings that the page shows verbatim.
MSG_SERVICE_REQUIRED = 'Please choose a service.'
MSG_DOCTOR_REQUIRED = 'Please choose a doctor.'
MSG_DATE_REQUIRED = 'Please choose a date.'
MSG_SLOT_REQUIRED = 'Please choose a time.'
MSG_SLOT_UNAVAILABLE = 'That time is not available. Please choose another.'
MSG_NAME_REQUIRED = 'Please enter your full name.'
MSG_EMAIL_REQUIRED = 'Please enter your email address.'
MSG_EMAIL_INVALID = 'Please enter a valid email address.'
MSG_PHONE_REQUIRED = 'Please enter your phone number.'
MSG_PHONE_INVALID = 'Please enter a valid 10-digit phone number.'
MSG_CONSENT_REQUIRED = 'Please accept the privacy notice to continue.'
MSG_SLOT_TAKEN = ('Sorry, that time was just booked by someone else. '
                  'Please choose another time.')

APPOINTMENT_FIELDS = ('service', 'doctor', 'date', 'slot')
DETAIL_FIELDS = ('name', 'email', 'phone', 'consent')

_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$')
_PHONE_SEPARATORS_RE = re.compile(r'[\s\-().]')
_MIN_NAME_LENGTH = 2
_PHONE_DIGITS = 10


@dataclasses.dataclass(frozen=True)
class Service:
    """A bookable service, such as a general check-up.

    Attributes:
      key: Stable identifier used in requests.
      name: Human-readable name shown on the page.
    """

    key: str
    name: str


@dataclasses.dataclass(frozen=True)
class Doctor:
    """A doctor who offers some services on some weekdays.

    Attributes:
      key: Stable identifier used in requests.
      name: Human-readable name shown on the page.
      service_keys: Services this doctor offers.
      weekdays: Working days, using `datetime.date.weekday()` numbering
        (Monday is 0).
    """

    key: str
    name: str
    service_keys: frozenset[str]
    weekdays: frozenset[int]


# A slot is identified by (doctor key, ISO date, 'HH:MM').
SlotKey = tuple[str, str, str]

# Page layouts: which steps exist and in which order.
LAYOUTS = ('wizard', 'details_first', 'combined')

# Every piece of page wording, with the original Riverside Clinic text.
# `{i}`, `{n}` and `{ref}` are filled in by the page.
DEFAULT_TEXT: Mapping[str, str] = {
    'clinic': 'Riverside Clinic',
    'title': 'Book an appointment',
    'h_appointment': 'Choose your appointment',
    'h_details': 'Your details',
    'h_review': 'Review and confirm',
    'h_done': 'Appointment confirmed',
    'service': 'Service',
    'doctor': 'Doctor',
    'date': 'Date',
    'time': 'Time',
    'name': 'Full name',
    'email': 'Email address',
    'phone': 'Phone number',
    'notes': 'Notes for the doctor (optional)',
    'consent': 'I accept the privacy notice',
    'next': 'Next',
    'back': 'Back',
    'confirm': 'Confirm booking',
    'choose_another': 'Choose another time',
    'book_another': 'Book another appointment',
    'pick_service': 'Select a service',
    'pick_doctor': 'Select a doctor',
    'pick_date': 'Select a date',
    'times_prompt': 'Choose a date to see times.',
    'times_loading': 'Loading times…',
    'times_free': '{n} times available.',
    'times_none': 'No times available on this date.',
    'step': 'Step {i} of {n}',
    'done': 'Booking complete',
    'reference': 'Your reference is {ref}.',
}


@dataclasses.dataclass(frozen=True)
class Presentation:
    """How the booking page looks: its layout and all of its wording.

    Validation messages come from the server rules and do not vary.

    Attributes:
      layout: One of `LAYOUTS`: 'wizard' (appointment, details, review),
        'details_first' (details, appointment, review) or 'combined'
        (appointment and details on one step, then review).
      text: Wording for every key of `DEFAULT_TEXT`.
    """

    layout: str = 'wizard'
    text: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(DEFAULT_TEXT))


@dataclasses.dataclass(frozen=True)
class Scenario:
    """A complete, immutable description of one environment instance.

    Attributes:
      scenario_id: Identifier recorded with every collected episode.
      today: The environment's fixed "current" date.
      days_ahead: How many days, starting tomorrow, can be booked.
      services: Services offered.
      doctors: Doctors and their schedules.
      slot_times: Daily appointment start times, as 'HH:MM'.
      prebooked: Slots already taken when the session starts.
      contested: Slots that look free but are taken by another patient at
        the moment someone first tries to confirm them. This models a race
        that cannot be seen from the page beforehand.
      slot_latency_ms: Artificial delay on the time-slot request, so pages
        have an observable loading state.
      presentation: The page's layout and wording.
    """

    scenario_id: str
    today: datetime.date
    days_ahead: int
    services: tuple[Service, ...]
    doctors: tuple[Doctor, ...]
    slot_times: tuple[str, ...]
    prebooked: frozenset[SlotKey] = frozenset()
    contested: frozenset[SlotKey] = frozenset()
    slot_latency_ms: int = 0
    presentation: Presentation = dataclasses.field(
        default_factory=Presentation)


@dataclasses.dataclass(frozen=True)
class Slot:
    """One appointment time as offered to the patient."""

    time: str
    available: bool


@dataclasses.dataclass(frozen=True)
class Booking:
    """A confirmed appointment.

    Attributes:
      reference: Confirmation code shown to the patient.
      service: Service key.
      doctor: Doctor key.
      date: ISO date.
      slot: Start time, as 'HH:MM'.
      name: Patient's name.
      email: Patient's email address.
      phone: Patient's phone number.
    """

    reference: str
    service: str
    doctor: str
    date: str
    slot: str
    name: str
    email: str
    phone: str


class BookingError(Exception):
    """Base class for booking failures shown to the patient."""


class ValidationError(BookingError):
    """The request has missing or malformed fields.

    Attributes:
      errors: Mapping from field name to the message shown for it.
    """

    def __init__(self, errors: Mapping[str, str]):
        super().__init__(f'Invalid fields: {sorted(errors)}')
        self.errors = dict(errors)


class SlotTakenError(BookingError):
    """The chosen slot was booked by someone else."""


def default_scenario() -> Scenario:
    """Returns the fixed scenario used by the E0.1 tests and demo."""
    return Scenario(
        scenario_id='e01-default',
        today=datetime.date(2026, 11, 23),  # A Monday.
        days_ahead=7,
        services=(
            Service('checkup', 'General check-up'),
            Service('dental', 'Dental cleaning'),
            Service('eye', 'Eye exam'),
        ),
        doctors=(
            Doctor('dr-patel', 'Dr. Asha Patel',
                   frozenset({'checkup', 'eye'}), frozenset({0, 1, 2, 3, 4})),
            Doctor('dr-kim', 'Dr. Daniel Kim',
                   frozenset({'checkup'}), frozenset({1, 3, 5})),
            Doctor('dr-okafor', 'Dr. Ngozi Okafor',
                   frozenset({'dental'}), frozenset({0, 2, 4})),
        ),
        slot_times=('09:00', '09:30', '10:00', '10:30', '11:00', '14:00',
                    '14:30', '15:00'),
        prebooked=frozenset({
            ('dr-patel', '2026-11-24', '09:30'),
            ('dr-patel', '2026-11-24', '14:00'),
            ('dr-okafor', '2026-11-25', '10:00'),
        }),
        contested=frozenset({('dr-patel', '2026-11-24', '11:00')}),
        slot_latency_ms=120,
    )


def is_valid_email(email: str) -> bool:
    """Returns whether `email` looks like a deliverable address."""
    return bool(_EMAIL_RE.match(email.strip()))


def is_valid_phone(phone: str) -> bool:
    """Returns whether `phone` has exactly 10 digits after separators."""
    digits = _PHONE_SEPARATORS_RE.sub('', phone.strip())
    return digits.isdigit() and len(digits) == _PHONE_DIGITS


class BookingSession:
    """The mutable booking state of one isolated environment instance.

    Each data-collection worker gets its own session, so parallel workers
    can never see each other's bookings.
    """

    def __init__(self, scenario: Scenario):
        """Creates a session in the scenario's initial state.

        Args:
          scenario: The environment definition to run.
        """
        self._scenario = scenario
        self._services = {s.key: s for s in scenario.services}
        self._doctors = {d.key: d for d in scenario.doctors}
        self._taken: set[SlotKey] = set()
        self._contest_resolved: set[SlotKey] = set()
        self._bookings: list[Booking] = []
        self.reset()

    @property
    def scenario(self) -> Scenario:
        """The scenario this session runs."""
        return self._scenario

    @property
    def bookings(self) -> tuple[Booking, ...]:
        """Bookings confirmed in this session, oldest first."""
        return tuple(self._bookings)

    @property
    def taken(self) -> tuple[SlotKey, ...]:
        """Slots that are no longer free (prebooked, booked or lost in a
        race), sorted."""
        return tuple(sorted(self._taken))

    @property
    def contests_resolved(self) -> tuple[SlotKey, ...]:
        """Race slots whose race has already happened, sorted."""
        return tuple(sorted(self._contest_resolved))

    def reset(self) -> None:
        """Restores the scenario's initial state exactly."""
        self._taken = set(self._scenario.prebooked)
        self._contest_resolved = set()
        self._bookings = []

    def services(self) -> list[Service]:
        """Returns all services, in scenario order."""
        return list(self._scenario.services)

    def doctors(self, service_key: str) -> list[Doctor]:
        """Returns the doctors offering `service_key`, in scenario order."""
        return [
            d for d in self._scenario.doctors if service_key in d.service_keys
        ]

    def dates(self, doctor_key: str) -> list[datetime.date]:
        """Returns the bookable dates for a doctor.

        Dates run from tomorrow up to `days_ahead` days out, keeping only
        the doctor's working days. Unknown doctors have no dates.

        Args:
          doctor_key: The doctor to look up.

        Returns:
          Dates in ascending order.
        """
        doctor = self._doctors.get(doctor_key)
        if doctor is None:
            return []
        candidates = (
            self._scenario.today + datetime.timedelta(days=offset)
            for offset in range(1, self._scenario.days_ahead + 1)
        )
        return [day for day in candidates if day.weekday() in doctor.weekdays]

    def slots(self, doctor_key: str, date_iso: str) -> list[Slot]:
        """Returns the day's slots for a doctor, marking taken ones.

        Contested slots are reported as available until someone tries to
        confirm them; that is the point of a contested slot.

        Args:
          doctor_key: The doctor to look up.
          date_iso: Date as 'YYYY-MM-DD'. Dates the doctor does not work
            have no slots.

        Returns:
          Slots in time order.
        """
        if date_iso not in {d.isoformat() for d in self.dates(doctor_key)}:
            return []
        return [
            Slot(time, (doctor_key, date_iso, time) not in self._taken)
            for time in self._scenario.slot_times
        ]

    def validate_appointment(
            self, request: Mapping[str, str]) -> dict[str, str]:
        """Checks the appointment step: service, doctor, date and slot.

        Args:
          request: Field values keyed by `APPOINTMENT_FIELDS` names.

        Returns:
          Mapping from field name to error message; empty when valid. Only
          the first problem in dependency order is reported, because later
          choices are meaningless until earlier ones are fixed.
        """
        service = request.get('service', '')
        doctor = request.get('doctor', '')
        date_iso = request.get('date', '')
        slot = request.get('slot', '')
        if service not in self._services:
            return {'service': MSG_SERVICE_REQUIRED}
        if doctor not in {d.key for d in self.doctors(service)}:
            return {'doctor': MSG_DOCTOR_REQUIRED}
        if date_iso not in {d.isoformat() for d in self.dates(doctor)}:
            return {'date': MSG_DATE_REQUIRED}
        offered = {s.time: s.available for s in self.slots(doctor, date_iso)}
        if slot not in offered:
            return {'slot': MSG_SLOT_REQUIRED}
        if not offered[slot]:
            return {'slot': MSG_SLOT_UNAVAILABLE}
        return {}

    def validate_details(self, request: Mapping[str, str]) -> dict[str, str]:
        """Checks the patient-details step.

        Args:
          request: Field values keyed by `DETAIL_FIELDS` names. `consent`
            is 'true' when the box is ticked.

        Returns:
          Mapping from field name to error message for every invalid field;
          empty when valid.
        """
        errors = {}
        if len(request.get('name', '').strip()) < _MIN_NAME_LENGTH:
            errors['name'] = MSG_NAME_REQUIRED
        email = request.get('email', '').strip()
        if not email:
            errors['email'] = MSG_EMAIL_REQUIRED
        elif not is_valid_email(email):
            errors['email'] = MSG_EMAIL_INVALID
        phone = request.get('phone', '').strip()
        if not phone:
            errors['phone'] = MSG_PHONE_REQUIRED
        elif not is_valid_phone(phone):
            errors['phone'] = MSG_PHONE_INVALID
        if request.get('consent') != 'true':
            errors['consent'] = MSG_CONSENT_REQUIRED
        return errors

    def book(self, request: Mapping[str, str]) -> Booking:
        """Confirms an appointment.

        Args:
          request: All `APPOINTMENT_FIELDS` and `DETAIL_FIELDS` values.

        Returns:
          The new booking.

        Raises:
          ValidationError: A field is missing or malformed, or the slot was
            already shown as unavailable.
          SlotTakenError: The slot looked free but another patient took it.
        """
        errors = self.validate_appointment(request)
        errors.update(self.validate_details(request))
        if errors:
            raise ValidationError(errors)
        slot_key = (request['doctor'], request['date'], request['slot'])
        if (slot_key in self._scenario.contested
                and slot_key not in self._contest_resolved):
            self._contest_resolved.add(slot_key)
            self._taken.add(slot_key)
            raise SlotTakenError(MSG_SLOT_TAKEN)
        self._taken.add(slot_key)
        booking = Booking(
            reference=f'BK-{len(self._bookings) + 1:04d}',
            service=request['service'],
            doctor=request['doctor'],
            date=request['date'],
            slot=request['slot'],
            name=request['name'].strip(),
            email=request['email'].strip(),
            phone=request['phone'].strip(),
        )
        self._bookings.append(booking)
        return booking


class SessionRegistry:
    """Thread-safe map from session id to an isolated `BookingSession`.

    Session creation is guarded by a lock rather than relying on dict
    atomicity. Sessions themselves are not thread-safe; the server mutates
    them only from its single event-loop thread.
    """

    def __init__(self, scenario: Scenario | Callable[[str], Scenario]):
        """Creates an empty registry.

        Args:
          scenario: Scenario given to every new session, or a function from
            a session id to that session's scenario.
        """
        self._scenario_for: Callable[[str], Scenario] = (
            scenario if callable(scenario) else lambda _: scenario)
        self._sessions: dict[str, BookingSession] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> BookingSession:
        """Returns the session for `session_id`, creating it if needed.

        Raises:
          ValueError: `session_id` is empty.
        """
        if not session_id:
            raise ValueError('Session id must be non-empty.')
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = BookingSession(self._scenario_for(session_id))
                self._sessions[session_id] = session
            return session
