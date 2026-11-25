"""E0.4: automated collection, branches and dataset validation."""

# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

import json
import sys
import types

import pytest

from browser_world.collection import collector
from browser_world.interaction import executor
from browser_world.collection import goal_policy
from browser_world.interaction import observer
from browser_world.collection import recorder
from browser_world.environment import scenarios


@pytest.fixture
def make_collector(new_page, scenario_server, tmp_path):
    url, assignments = scenario_server

    def factory():
        page = new_page(700)
        watcher = observer.attach(page)
        return collector.Collector(page, watcher, url, assignments,
                                   tmp_path, executor.FAST_SPEED)

    return factory


def _records(tmp_path):
    found = []
    for split in scenarios.SPLITS:
        path = tmp_path / split / recorder.TRANSITIONS_FILE
        if path.exists():
            found.extend(recorder.read_jsonl(path))
    return found


def test_intents_count_only_when_the_page_shows_them(world):
    intent = goal_policy.Intent(executor.SELECT, 'General check-up')
    name = goal_policy.Intent(executor.TYPE, 'Full name', 'Sam Rivera')
    before = world.watcher.observe()
    assert goal_policy.satisfied(before, intent) is False
    assert goal_policy.satisfied(before, name) is None  # On a hidden step.
    after = world.act(lambda: world.page.select_option('#service',
                                                       'checkup'))
    assert goal_policy.satisfied(after, intent) is True
    assert goal_policy.pending(after, [intent, name]) == [name]


def test_noisy_collection_records_detours_and_still_books(make_collector,
                                                          tmp_path):
    collect = make_collector()
    collect.collect(40, seed_base=20_000,
                    policy_weights={collector.NOISY: 1.0})
    tags = {}
    for episode in collect.episodes:
        for tag, count in episode.tags.items():
            tags[tag] = tags.get(tag, 0) + count
    assert set(tags) - {'goal'}, tags  # At least one kind of detour.
    assert any(e.ended == goal_policy.DONE for e in collect.episodes)
    report = collector.validate_raw(tmp_path)
    assert not report['problems'] and report['records'] >= 40


def test_exploration_is_bounded_and_complete(make_collector, tmp_path):
    collect = make_collector()
    collect.collect(10, seed_base=21_000,
                    policy_weights={collector.EXPLORE: 1.0})
    for episode in collect.episodes:
        assert episode.status == collector.COMPLETE
        assert episode.ended in ('step_budget', goal_policy.DONE)
        assert episode.steps <= collector._EXPLORE_STEPS[1]  # pylint: disable=protected-access
    assert not collector.validate_raw(tmp_path)['problems']


def test_branch_group_starts_every_branch_from_the_same_verified_state(
        make_collector, tmp_path):
    collect = make_collector()
    collect.collect(1, seed_base=22_000,
                    policy_weights={collector.GOAL: 1.0}, branch_every=1)
    branches = [e for e in collect.episodes if e.branch]
    assert len(branches) == 3
    assert all(e.status == collector.COMPLETE for e in branches), [
        (e.status, e.error) for e in branches]
    assert len({e.branch['point_fingerprint'] for e in branches}) == 1
    assert len({tuple(e.branch['action'][:4]) for e in branches}) == 3
    records = _records(tmp_path)
    assert len(records) == 3
    assert {r['run_origin'] for r in records} == {'branch'}
    assert len({r['run_id'] for r in records}) == 3
    first = records[0]['before']
    for record in records[1:]:
        assert not recorder.differences(
            recorder.canonical(observer.observation_from_dict(first)),
            recorder.canonical(observer.observation_from_dict(
                record['before'])))
        assert record['truth']['before'] == records[0]['truth']['before']
    assert {'bookings', 'taken', 'contests_resolved'} == set(
        records[0]['truth']['before'])
    assert not collector.validate_raw(tmp_path)['problems']


def test_validation_recomputes_boundaries_instead_of_trusting_flags(
        make_collector, tmp_path):
    collect = make_collector()
    collect.collect(5, seed_base=23_000,
                    policy_weights={collector.GOAL: 1.0}, branch_every=0)
    split = collect.episodes[0].split
    path = tmp_path / split / recorder.TRANSITIONS_FILE
    records = recorder.read_jsonl(path)
    assert not collector.validate_raw(tmp_path)['problems']
    tampered = records[1]
    tampered['before']['elements'][0]['name'] = 'Something else'
    assert tampered['continuity']['continuous'] is True  # Flag still says ok.
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    problems = collector.validate_raw(tmp_path)['problems']
    assert any('boundary' in p for p in problems), problems


def test_manifest_splits_and_hidden_metadata(make_collector, tmp_path):
    collect = make_collector()
    collect.collect(25, seed_base=24_000)
    manifest = json.loads((tmp_path / collector.MANIFEST_FILE).read_text())
    assert manifest['schema_version'] == recorder.SCHEMA_VERSION
    assert len(manifest['episodes']) == len(collect.episodes)
    for episode in manifest['episodes']:
        assert episode['seed'] >= 24_000
        assert scenarios.split_of(episode['family']) == episode['split']
    for record in _records(tmp_path):
        assert record['schema_version'] == 3
        assert scenarios.split_of(record['scenario']['family']) == (
            record['scenario']['split'])
    report = collector.coverage(tmp_path, collect.episodes)
    assert report['transitions'] >= 25
    assert not collector.validate_raw(tmp_path)['problems']


def test_peak_process_rss_mib_platforms(monkeypatch):
    fake_resource = types.SimpleNamespace(
        RUSAGE_SELF=0,
        getrusage=lambda _who: types.SimpleNamespace(ru_maxrss=2048),
    )
    monkeypatch.setattr(collector, 'resource', fake_resource)

    monkeypatch.setattr(sys, 'platform', 'linux')
    assert collector._peak_process_rss_mib() == 2.0  # pylint: disable=protected-access

    monkeypatch.setattr(sys, 'platform', 'win32')
    assert collector._peak_process_rss_mib() is None  # pylint: disable=protected-access

    monkeypatch.setattr(sys, 'platform', 'darwin')
    assert collector._peak_process_rss_mib() is None  # pylint: disable=protected-access

    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(collector, 'resource', None)
    assert collector._peak_process_rss_mib() is None  # pylint: disable=protected-access
