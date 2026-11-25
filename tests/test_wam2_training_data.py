"""Tests for recorded experience windows and rollout integrity."""

from __future__ import annotations

from browser_world.interaction import executor
from browser_world.wam2 import model
from browser_world.wam2 import observation
from browser_world.wam2 import training_data
from tests import wam2_helpers


def test_load_real_train_and_validation_windows() -> None:
    train = training_data.load_experience_windows(split='train', max_horizon=3)
    val = training_data.load_experience_windows(
        split='validation', max_horizon=3
    )

    assert train.transition_count == 343
    assert val.transition_count == 82
    assert len(train.windows[1]) == 343
    assert len(val.windows[1]) == 82
    assert len(train.windows[2]) > 0
    assert len(train.windows[3]) > 0

    for horizon in (1, 2, 3):
        for win in train.windows[horizon] + val.windows[horizon]:
            assert len(win.observations) == horizon + 1
            assert len(win.actions) == horizon
            assert len(win.pointers) == horizon
            assert len(win.steps) == horizon
            assert list(win.steps) == list(
                range(win.steps[0], win.steps[0] + horizon)
            )


def test_no_cross_branch_or_episode_joins() -> None:
    val = training_data.load_experience_windows(
        split='validation', max_horizon=3
    )
    branch_runs = {
        rid for rid, grp in val.branch_groups.items() if grp is not None
    }
    assert len(branch_runs) == 3
    for win in val.windows[2] + val.windows[3]:
        assert win.run_id not in branch_runs


def test_rollout_stops_at_unknown_future_entity() -> None:
    enc = wam2_helpers.FakeSemanticEncoder()
    role_map, roles = observation.load_role_vocab()
    world = model.WorldActionModel(role_count=len(roles), slot_count=32)

    obs0 = wam2_helpers.make_observation([
        wam2_helpers.make_element(
            1, 'combobox', 'Service', match_key='svc'
        ),
        wam2_helpers.make_element(
            2, 'option', 'Skin', parent_id=1, match_key='opt1'
        ),
    ])
    # obs1 introduces a brand-new option node_id=99 not in obs0.
    obs1 = wam2_helpers.make_observation([
        wam2_helpers.make_element(
            1, 'combobox', 'Service', match_key='svc'
        ),
        wam2_helpers.make_element(
            2, 'option', 'Skin', parent_id=1, match_key='opt1'
        ),
        wam2_helpers.make_element(
            99, 'option', 'Dr. X', parent_id=1, match_key='drx'
        ),
    ])
    obs2 = obs1
    win = training_data.ExperienceWindow(
        observations=(obs0, obs1, obs2),
        actions=(
            executor.Action(executor.SELECT, target=2),
            executor.Action(executor.SELECT, target=99),
        ),
        pointers=((10.0, 10.0), (20.0, 20.0)),
        run_id='synthetic-run',
        steps=(0, 1),
    )

    res = training_data.rollout_imagined_window(
        world, win, enc, role_map, slot_count=32
    )
    assert res.completed_steps == 1
    assert res.terminated_at_step == 1
    assert res.termination_reason == 'unknown_future_entity'
    assert len(res.predictions) == 1
