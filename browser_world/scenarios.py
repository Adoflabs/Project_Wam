"""Seeded booking scenarios in families of layout and wording.

A family fixes how the page looks (`booking.LAYOUTS` x three wordings); a
seed fixes the world behind it: date, services, doctors and their
schedules, slot times, which slots are taken, which slots will be lost in a
race, and response latency. The same (family, seed) always gives the same
scenario.

Families are assigned to train, validation and test before any data is
collected. The test families use a wording never seen in training, so a
model can only do well on them by understanding the page rather than
memorising its words.

Scenario details are hidden from the model: the page learns them only
through the server API, and records carry them only as metadata.

Typical usage example:

  scenario = scenarios.generate('harbor-combined', seed=7)
  goal = scenarios.goal(scenario, seed=7, outcome=scenarios.SUCCESS)
"""

from __future__ import annotations

import dataclasses
import datetime
import random

from browser_world import booking

SUCCESS = 'success'
CONFLICT = 'conflict'
INVALID_EMAIL = 'invalid_email'
NATURAL = 'natural'
OUTCOMES = (SUCCESS, CONFLICT, INVALID_EMAIL, NATURAL)

_HARBOR_TEXT = {
    'clinic': 'Harbor Health Centre',
    'title': 'Book a visit',
    'h_appointment': 'Pick a time',
    'h_details': 'About you',
    'h_review': 'Check your booking',
    'h_done': 'You are booked',
    'service': 'Type of visit',
    'doctor': 'Clinician',
    'date': 'Day',
    'time': 'Time slot',
    'name': 'Your name',
    'email': 'E-mail',
    'phone': 'Mobile number',
    'notes': 'Anything we should know? (optional)',
    'consent': 'I agree to the terms of service',
    'next': 'Continue',
    'back': 'Go back',
    'confirm': 'Book now',
    'choose_another': 'Pick a different time',
    'book_another': 'Make another booking',
    'pick_service': 'Choose a visit type',
    'pick_doctor': 'Choose a clinician',
    'pick_date': 'Choose a day',
    'times_prompt': 'Pick a day to see open times.',
    'times_loading': 'Checking open times…',
    'times_free': '{n} open times.',
    'times_none': 'Nothing open that day.',
    'step': 'Page {i} / {n}',
    'done': 'All done',
    'reference': 'Booking code: {ref}',
}

_CIVIC_TEXT = {
    'clinic': 'Civic Medical Practice',
    'title': 'Request an appointment',
    'h_appointment': 'Appointment preferences',
    'h_details': 'Patient information',
    'h_review': 'Confirm your request',
    'h_done': 'Request accepted',
    'service': 'Department',
    'doctor': 'Practitioner',
    'date': 'Preferred date',
    'time': 'Start time',
    'name': 'Patient name',
    'email': 'Contact email',
    'phone': 'Contact telephone',
    'notes': 'Additional information (optional)',
    'consent': 'I consent to the processing of my data',
    'next': 'Proceed',
    'back': 'Previous',
    'confirm': 'Submit request',
    'choose_another': 'Select another start time',
    'book_another': 'Submit another request',
    'pick_service': 'Select department',
    'pick_doctor': 'Select practitioner',
    'pick_date': 'Select date',
    'times_prompt': 'Select a date to list start times.',
    'times_loading': 'Retrieving start times…',
    'times_free': 'Start times available: {n}',
    'times_none': 'No start times on this date.',
    'step': 'Stage {i} of {n}',
    'done': 'Request complete',
    'reference': 'Reference number {ref}',
}

WORDINGS = {
    'riverside': dict(booking.DEFAULT_TEXT),
    'harbor': _HARBOR_TEXT,
    'civic': _CIVIC_TEXT,
}

# Every family is `<wording>-<layout>`. The split is fixed here, before any
# collection, and must not be changed after looking at results.
SPLITS = {
    'train': ('riverside-wizard', 'riverside-details_first',
              'riverside-combined', 'harbor-wizard', 'harbor-details_first'),
    'validation': ('harbor-combined',),
    'test': ('civic-wizard', 'civic-details_first', 'civic-combined'),
}
FAMILIES = tuple(f for families in SPLITS.values() for f in families)

_SERVICE_POOL = (
    ('checkup', 'General check-up'), ('dental', 'Dental cleaning'),
    ('eye', 'Eye exam'), ('physio', 'Physiotherapy'),
    ('derm', 'Skin consultation'), ('vaccine', 'Vaccination'),
)
_DOCTOR_POOL = (
    ('dr-patel', 'Dr. Asha Patel'), ('dr-kim', 'Dr. Daniel Kim'),
    ('dr-okafor', 'Dr. Ngozi Okafor'), ('dr-silva', 'Dr. Marta Silva'),
    ('dr-novak', 'Dr. Tomas Novak'), ('dr-haddad', 'Dr. Leila Haddad'),
    ('dr-ito', 'Dr. Ken Ito'), ('dr-mensah', 'Dr. Ama Mensah'),
)
_SLOT_POOL = tuple(f'{h:02d}:{m:02d}' for h in range(8, 17) for m in (0, 30))
_FIRST_NAMES = ('Sam', 'Alex', 'Priya', 'Jordan', 'Mei', 'Omar', 'Lena',
                'Kofi', 'Ines', 'Ravi')
_LAST_NAMES = ('Rivera', 'Chen', 'Okoro', 'Lindqvist', 'Haddad', 'Moreau',
               'Tanaka', 'Novak', 'Adeyemi', 'Costa')
_BASE_DATE = datetime.date(2026, 11, 2)


def split_of(family: str) -> str:
    """Returns 'train', 'validation' or 'test' for a family.

    Raises:
      ValueError: Unknown family.
    """
    for split, families in SPLITS.items():
        if family in families:
            return split
    raise ValueError(f'Unknown family {family!r}.')


def _rng(*parts: object) -> random.Random:
    return random.Random(':'.join(str(p) for p in parts))


def generate(family: str, seed: int) -> booking.Scenario:
    """Returns the scenario for a family and seed.

    Args:
      family: One of `FAMILIES`.
      seed: Any integer; the same seed always gives the same world.

    Returns:
      A scenario with at least one bookable slot and one contested slot.

    Raises:
      ValueError: Unknown family.
    """
    split_of(family)  # Validates the family.
    wording, layout = family.split('-', 1)
    rng = _rng('world', family, seed)
    today = _BASE_DATE + datetime.timedelta(days=rng.randint(0, 60))
    services = tuple(booking.Service(k, n) for k, n in
                     rng.sample(_SERVICE_POOL, rng.randint(2, 4)))
    doctors = _doctors(rng, services)
    slot_times = tuple(sorted(rng.sample(_SLOT_POOL, rng.randint(5, 9))))
    days_ahead = rng.randint(5, 10)
    draft = booking.Scenario(
        scenario_id=f'{family}-{seed}', today=today, days_ahead=days_ahead,
        services=services, doctors=doctors, slot_times=slot_times)
    open_slots = _all_slots(draft)
    taken_share = rng.uniform(0.1, 0.35)
    prebooked = frozenset(s for s in open_slots if rng.random() < taken_share)
    free = sorted(set(open_slots) - prebooked)
    contested = frozenset(rng.sample(free, min(len(free) - 1,
                                               rng.randint(1, 3))))
    return dataclasses.replace(
        draft, prebooked=prebooked, contested=contested,
        slot_latency_ms=rng.choice((0, 80, 150)),
        presentation=booking.Presentation(layout=layout,
                                          text=dict(WORDINGS[wording])))


def _doctors(rng: random.Random,
             services: tuple[booking.Service, ...]
             ) -> tuple[booking.Doctor, ...]:
    """Picks doctors so every service has at least one."""
    pool = rng.sample(_DOCTOR_POOL, rng.randint(len(services),
                                                len(services) + 2))
    doctors = []
    for i, (key, name) in enumerate(pool):
        offered = {services[i % len(services)].key}
        if rng.random() < 0.4:
            offered.add(rng.choice(services).key)
        weekdays = frozenset(rng.sample(range(7), rng.randint(3, 6)))
        doctors.append(booking.Doctor(key, name, frozenset(offered), weekdays))
    return tuple(doctors)


def _all_slots(scenario: booking.Scenario) -> list[booking.SlotKey]:
    """Every (doctor, date, time) the scenario offers, before bookings."""
    session = booking.BookingSession(scenario)
    return [(d.key, day.isoformat(), time)
            for d in scenario.doctors
            for day in session.dates(d.key)
            for time in scenario.slot_times]


@dataclasses.dataclass(frozen=True)
class Contact:
    """A synthetic patient. No real person's details are ever used."""

    name: str
    email: str
    phone: str


def contact(key: str) -> Contact:
    """Returns a synthetic, valid patient for any key string."""
    rng = _rng('contact', key)
    first, last = rng.choice(_FIRST_NAMES), rng.choice(_LAST_NAMES)
    number = rng.randint(10, 99)
    return Contact(
        name=f'{first} {last}',
        email=f'{first}.{last}{number}@example.com'.lower(),
        phone=f'555-01{rng.randint(0, 9)}-{rng.randint(1000, 9999)}')


@dataclasses.dataclass(frozen=True)
class Goal:
    """What a goal-directed decider tries to book, in page terms.

    Privileged decider knowledge: never part of the model input.

    Attributes:
      outcome: The outcome the goal is built to produce.
      service: Service name as shown in its dropdown.
      doctor: Doctor name as shown.
      date: Date label as shown, such as 'Tue 24 Nov'.
      slot: Time, such as '10:00'.
      contact: The patient's details; for INVALID_EMAIL the email is
        malformed.
    """

    outcome: str
    service: str
    doctor: str
    date: str
    slot: str
    contact: Contact


def goal(scenario: booking.Scenario, seed: int, outcome: str) -> Goal:
    """Returns a goal that leads to `outcome` in `scenario`.

    SUCCESS books a free, uncontested slot; CONFLICT picks a contested slot,
    which looks free until confirmed; INVALID_EMAIL picks a free slot but
    enters a malformed email; NATURAL picks uniformly among all slots that
    look free, so races happen only at their natural rate.

    Raises:
      ValueError: Unknown outcome.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f'Unknown outcome {outcome!r}.')
    rng = _rng('goal', scenario.scenario_id, seed, outcome)
    session = booking.BookingSession(scenario)
    free = sorted(set(_all_slots(scenario)) - scenario.prebooked
                  - scenario.contested)
    pool = free
    if outcome == CONFLICT:
        pool = sorted(scenario.contested)
    elif outcome == NATURAL:
        pool = sorted(set(free) | scenario.contested)
    doctor_key, date_iso, time = rng.choice(pool)
    doctor = next(d for d in scenario.doctors if d.key == doctor_key)
    service = rng.choice([s for s in scenario.services
                          if s.key in doctor.service_keys])
    day = next(d for d in session.dates(doctor_key)
               if d.isoformat() == date_iso)
    patient = contact(f'{scenario.scenario_id}:{seed}')
    if outcome == INVALID_EMAIL:
        patient = dataclasses.replace(
            patient, email=patient.email.replace('@example.com', '@example'))
    return Goal(outcome=outcome, service=service.name, doctor=doctor.name,
                date=day.strftime('%a %d %b'), slot=time, contact=patient)
