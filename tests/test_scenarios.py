"""E0.5: seeded scenario families, run end to end in real Chromium."""

# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

import dataclasses
import json

import pytest
from starlette import testclient

from browser_world import booking
from browser_world import executor
from browser_world import recorder
from browser_world import scenarios
from browser_world import server
from browser_world import goal_policy
from tests import run_checks


def test_generation_is_deterministic_and_seeds_differ():
    first = scenarios.generate('harbor-combined', 3)
    assert first == scenarios.generate('harbor-combined', 3)
    assert first != scenarios.generate('harbor-combined', 4)
    assert scenarios.contact('a') == scenarios.contact('a')
    assert scenarios.contact('a') != scenarios.contact('b')


def test_splits_are_fixed_disjoint_and_hold_out_a_wording():
    all_families = [f for fs in scenarios.SPLITS.values() for f in fs]
    assert len(all_families) == len(set(all_families)) == 9
    train_wordings = {f.split('-', maxsplit=1)[0]
                      for f in scenarios.SPLITS['train']}
    test_wordings = {f.split('-', maxsplit=1)[0]
                     for f in scenarios.SPLITS['test']}
    assert not train_wordings & test_wordings
    assert {f.split('-', 1)[1] for f in scenarios.SPLITS['test']} == set(
        booking.LAYOUTS)
    with pytest.raises(ValueError):
        scenarios.split_of('nope-wizard')


@pytest.mark.parametrize('family', scenarios.FAMILIES)
def test_every_family_offers_free_and_contested_slots(family):
    for seed in range(5):
        scenario = scenarios.generate(family, seed)
        wording, layout = family.split('-', 1)
        assert scenario.presentation.layout == layout
        assert scenario.presentation.text == scenarios.WORDINGS[wording]
        assert set(scenario.presentation.text) == set(booking.DEFAULT_TEXT)
        assert scenario.contested
        assert not scenario.contested & scenario.prebooked
        for outcome in scenarios.OUTCOMES:
            goal = scenarios.goal(scenario, seed, outcome)
            assert goal.contact.email.split('@')[1].startswith('example')


def test_the_race_slot_varies_across_seeds():
    times = {t for seed in range(20)
             for _, _, t in scenarios.generate('riverside-wizard',
                                               seed).contested}
    assert len(times) > 5  # Not one fixed time that always fails.


def test_goals_aim_at_the_right_kind_of_slot():
    scenario = scenarios.generate('civic-wizard', 11)
    session = booking.BookingSession(scenario)
    names = {d.name: d.key for d in scenario.doctors}
    for outcome in scenarios.OUTCOMES:
        goal = scenarios.goal(scenario, 11, outcome)
        day = next(d for d in session.dates(names[goal.doctor])
                   if d.strftime('%a %d %b') == goal.date)
        key = (names[goal.doctor], day.isoformat(), goal.slot)
        assert (key in scenario.contested) == (outcome == scenarios.CONFLICT)
        assert key not in scenario.prebooked
    invalid = scenarios.goal(scenario, 11, scenarios.INVALID_EMAIL)
    assert not booking.is_valid_email(invalid.contact.email)


def test_each_session_gets_its_scenarios_presentation():
    chosen = scenarios.generate('civic-details_first', 2)
    app = server.create_app(
        lambda sid: chosen if sid == 'civic' else booking.default_scenario())
    client = testclient.TestClient(app)
    civic = client.get('/api/presentation',
                       headers={server.SESSION_HEADER: 'civic'}).json()
    default = client.get('/api/presentation',
                         headers={server.SESSION_HEADER: 'other'}).json()
    assert civic['layout'] == 'details_first'
    assert civic['text']['next'] == 'Proceed'
    assert default == {'layout': 'wizard', 'text': booking.DEFAULT_TEXT}


def _metadata(family, seed):
    return {'scenario_id': f'{family}-{seed}', 'family': family,
            'split': scenarios.split_of(family), 'seed': seed}


def _run(make_scenario_world, tmp_path, family, seed, outcome):
    scenario = scenarios.generate(family, seed)
    world = make_scenario_world(scenario, viewport_height=600)
    hand = executor.Executor(world.page, world.watcher, executor.FAST_SPEED)
    rec = recorder.Recorder(world.page, world.watcher, hand, tmp_path,
                            truth_fn=world.truth_bookings)
    goal = scenarios.goal(scenario, seed, outcome)
    ended, transitions = goal_policy.run_episode(
        world.watcher, rec, scenario, goal, _metadata(family, seed))
    return scenario, goal, ended, transitions, world


@pytest.mark.parametrize('family, seed', [
    ('harbor-details_first', 5),
    ('riverside-combined', 8),
    ('civic-wizard', 13),
])
def test_successful_booking_in_different_families(make_scenario_world,
                                                  tmp_path, family, seed):
    scenario, goal, ended, transitions, world = _run(
        make_scenario_world, tmp_path, family, seed, scenarios.SUCCESS)
    assert ended == goal_policy.DONE
    saved = run_checks.assert_one_continuous_run(tmp_path,
                                                   len(transitions))
    text = scenario.presentation.text
    first = transitions[0].before
    assert any(e.role == 'heading' and e.name == text['title']
               for e in first.elements)
    booked = world.truth_bookings()
    assert len(booked) == 1
    assert (booked[0]['slot'], booked[0]['email']) == (goal.slot,
                                                       goal.contact.email)
    assert saved[-1]['truth']['after'] == booked
    assert saved[0]['scenario'] == _metadata(family, seed)


def test_conflict_outcome_is_recorded(make_scenario_world, tmp_path):
    _, _, ended, transitions, world = _run(
        make_scenario_world, tmp_path, 'harbor-wizard', 4, scenarios.CONFLICT)
    assert ended == goal_policy.CONFLICT
    run_checks.assert_one_continuous_run(tmp_path, len(transitions))
    confirm = transitions[-1]
    assert confirm.action.kind == executor.CLICK
    assert any(e.role == 'alert' and e.name == booking.MSG_SLOT_TAKEN
               for e in confirm.after.elements)
    assert world.truth_bookings() == []


def test_validation_outcome_is_recorded(make_scenario_world, tmp_path):
    _, _, ended, transitions, world = _run(
        make_scenario_world, tmp_path, 'civic-combined', 6,
        scenarios.INVALID_EMAIL)
    assert ended == goal_policy.VALIDATION_ERROR
    run_checks.assert_one_continuous_run(tmp_path, len(transitions))
    assert any(e.role == 'alert' and e.name == booking.MSG_EMAIL_INVALID
               for e in transitions[-1].after.elements)
    assert world.truth_bookings() == []


def test_hidden_scenario_details_stay_out_of_model_input(make_scenario_world,
                                                         tmp_path):
    scenario, _, _, transitions, _ = _run(
        make_scenario_world, tmp_path, 'riverside-wizard', 21,
        scenarios.SUCCESS)
    for transition in transitions:
        text = json.dumps(recorder.model_input(transition))
        for secret in (scenario.scenario_id, 'riverside-wizard', 'family',
                       'seed', 'contested', 'prebooked'):
            assert secret not in text
        assert transition.before.url.split('?')[1].startswith('session=')
    assert dataclasses.asdict(scenario.presentation)['layout'] == 'wizard'
