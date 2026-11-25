"""Numerical feature and supervision definitions for curated data."""

STATE_NAMES = (
    'checked',
    'disabled',
    'expanded',
    'focused',
    'invalid',
    'readonly',
    'required',
    'selected',
)

ACTION_KINDS = (
    'click',
    'type',
    'select',
    'check',
    'scroll',
)

VALUE_SOURCES = (
    'unchanged',
    'typed',
    'option',
    'cleared',
    'other',
)

OUTCOMES = (
    'no_change',
    'alert_created',
    'alert_resolved',
    'alert_persisting',
    'heading_changed',
    'booking_created',
    'race_lost',
    'scroll_changed',
)

MAX_NAME_WORDS = 8
MAX_CHARS = 32
PAD = 0
UNK = 1
