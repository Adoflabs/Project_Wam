"""E0.3a-2: a complete booking, executed and recorded as one continuous run.

Every step goes through the executor and the recorder. The decider below
knows only the goal (which names to pick and what to type) and chooses
from `available_actions`; when the next target is not offered because it
is off-screen, it scrolls. No selectors, no direct state changes.
"""

# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

from browser_world import executor
from browser_world import recorder

# The goal, as (kind, accessible name, text or desired checked state).
BOOKING_PLAN = (
    (executor.SELECT, 'General check-up', None),
    (executor.SELECT, 'Dr. Asha Patel', None),
    (executor.SELECT, 'Tue 24 Nov', None),
    (executor.CHECK, '10:00', True),
    (executor.CLICK, 'Next', None),
    (executor.TYPE, 'Full name', 'Sam Rivera'),
    (executor.TYPE, 'Email address', 'sam.rivera@example.com'),
    (executor.TYPE, 'Phone number', '555-010-2030'),
    (executor.CHECK, 'I accept the privacy notice', True),
    (executor.CLICK, 'Next', None),
    (executor.CLICK, 'Confirm booking', None),
)
_MAX_SCROLLS_PER_STEP = 5


def decide(observation, kind, name, value):
    """Returns the action for the goal step, or a scroll if its target is
    not offered yet, or None if neither is possible."""
    for offer in executor.available_actions(observation):
        if offer.kind != kind or offer.name != name:
            continue
        if kind == executor.CHECK and offer.checked != value:
            continue
        return offer.action(value if kind == executor.TYPE else None)
    for offer in executor.available_actions(observation):
        if offer.kind == executor.SCROLL and offer.scroll_y > 0:
            return offer.action()
    return None


def run_booking(world, rec):
    """Executes BOOKING_PLAN in one run and returns its transitions."""
    run_id = rec.new_run('booking', 'start')
    transitions = []
    for kind, name, value in BOOKING_PLAN:
        for _ in range(_MAX_SCROLLS_PER_STEP + 1):
            world.watcher.wait_until_settled()
            observation = world.watcher.observe()
            action = decide(observation, kind, name, value)
            assert action is not None, f'cannot reach {kind} {name}'
            outcome = rec.record_step(run_id, observation, action)
            assert isinstance(outcome, recorder.Transition), outcome
            transitions.append(outcome)
            if action.kind != executor.SCROLL:
                break
    return transitions


def test_complete_booking_is_one_continuous_recorded_run(make_world,
                                                         tmp_path):
    world = make_world(viewport_height=600)
    hand = executor.Executor(world.page, world.watcher, executor.FAST_SPEED)
    rec = recorder.Recorder(world.page, world.watcher, hand, tmp_path,
                            truth_fn=world.truth_bookings)
    transitions = run_booking(world, rec)

    kinds = [t.action.kind for t in transitions]
    assert executor.SCROLL in kinds  # The short window forced scrolling.
    for kind in (executor.SELECT, executor.CHECK, executor.TYPE,
                 executor.CLICK):
        assert kind in kinds

    saved = recorder.read_jsonl(rec.transitions_path)
    assert len(saved) == len(transitions)
    assert len({r['run_id'] for r in saved}) == 1
    assert [r['step'] for r in saved] == list(range(len(saved)))
    assert all(not recorder.sequence_problems(r) for r in saved), [
        recorder.sequence_problems(r) for r in saved]
    for previous, current in zip(transitions, transitions[1:]):
        assert not recorder.differences(recorder.canonical(previous.after),
                                        recorder.canonical(current.before))

    # Dependencies the world model should learn, seen in the records.
    service = transitions[0]
    assert service.before.find('doctor').states['disabled']
    assert not service.after.find('doctor').states['disabled']

    final = transitions[-1]
    assert final.after.find('reference').name == 'Your reference is BK-0001.'
    assert final.truth['before'] == []
    booked = final.truth['after']
    assert len(booked) == 1
    assert (booked[0]['doctor'], booked[0]['date'], booked[0]['slot'],
            booked[0]['email']) == ('dr-patel', '2026-11-24', '10:00',
                                    'sam.rivera@example.com')
    assert not (tmp_path / recorder.REFUSALS_FILE).exists()
