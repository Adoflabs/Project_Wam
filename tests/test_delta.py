"""Tests for observation deltas against the live booking page."""

# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

from browser_world import booking
from browser_world import delta


def test_no_change_gives_empty_delta(world):
    first = world.watcher.observe()
    second = world.watcher.observe()
    change = delta.diff(first, second)
    assert change == {'url_changed': False, 'added': [], 'removed': [],
                      'changed': []}


def test_dependent_dropdown_change(world):
    before = world.watcher.observe()
    after = world.act(lambda: world.page.select_option('#service', 'dental'))
    change = delta.diff(before, after)
    changed = {c['key']: c['fields'] for c in change['changed']}
    assert changed['service']['value'] == ['Select a service',
                                           'Dental cleaning']
    assert changed['doctor']['disabled'] == [True, False]
    added = {(a['role'], a['name']) for a in change['added']}
    assert ('option', 'Dr. Ngozi Okafor') in added


def test_validation_error_appears_and_moves_layout(world):
    before = world.watcher.observe()
    after = world.click('next-appointment')
    change = delta.diff(before, after)
    assert {'key': 'error-service', 'role': 'alert',
            'name': booking.MSG_SERVICE_REQUIRED} in change['added']
    changed = {c['key']: c['fields'] for c in change['changed']}
    assert changed['service']['invalid'] == [False, True]
    assert 'box' in changed['doctor']  # Pushed down by the error message.


def test_rerendered_slots_match_by_key(world):
    world.choose_appointment('checkup', 'dr-patel', '2026-11-24', slot=None)
    before = world.watcher.observe()
    world.act(lambda: world.page.select_option('#date', '2026-11-25'))
    after = world.act(lambda: world.page.select_option('#date',
                                                       '2026-11-24'))
    change = delta.diff(before, after)
    keys = {c['key'] for c in change['changed']}
    assert 'slot-09:00' not in keys  # New node, same observable element.
    assert not [a for a in change['added'] if a['role'] == 'radio']
