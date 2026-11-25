"""Tests for temporal target consistency, validation, and checkpoints."""

from __future__ import annotations

import pathlib

import pytest
import torch

from browser_world.interaction import executor
from browser_world.wam2 import model
from browser_world.wam2 import observation
from browser_world.wam2 import train
from browser_world.wam2 import training_data
from tests import wam2_helpers


def test_context_adaptation_gradients_and_checkpoint_reload(
    tmp_path: pathlib.Path,
) -> None:
    enc = wam2_helpers.FakeSemanticEncoder()
    role_map, roles = observation.load_role_vocab()
    world = model.WorldActionModel(role_count=len(roles), slot_count=18)

    obs0 = wam2_helpers.make_observation([
        wam2_helpers.make_element(
            1, 'combobox', 'Service', match_key='svc', tag='select'
        ),
        wam2_helpers.make_element(
            2, 'option', 'Skin', parent_id=1, match_key='opt'
        ),
    ])
    obs1 = wam2_helpers.make_observation([
        wam2_helpers.make_element(
            1, 'combobox', 'Service', match_key='svc', tag='select'
        ),
        wam2_helpers.make_element(
            2, 'option', 'Skin', parent_id=1,
            states={'selected': True}, match_key='opt',
        ),
    ])
    obs2 = wam2_helpers.make_observation([
        wam2_helpers.make_element(
            1, 'combobox', 'Service', match_key='svc', tag='select'
        ),
        wam2_helpers.make_element(
            2, 'option', 'Skin', parent_id=1,
            states={'selected': True}, match_key='opt',
        ),
    ])
    win = training_data.ExperienceWindow(
        observations=(obs0, obs1, obs2),
        actions=(
            executor.Action(executor.SELECT, target=2),
            executor.Action(executor.SCROLL, scroll_y=50.0),
        ),
        pointers=((10.0, 10.0), (20.0, 20.0)),
        run_id='synth',
        steps=(0, 1),
    )
    encoded = train.encode_windows((win,), enc, role_map, slot_count=18)
    loss_res = train._window_training_loss(world, encoded[0], enc)
    loss_res.loss.backward()

    ev_grads = [
        p.grad for p in world.evidence.parameters() if p.grad is not None
    ]
    ad_grads = [
        p.grad for p in world.adaptation.parameters() if p.grad is not None
    ]
    assert ev_grads and any(g.abs().sum().item() > 0 for g in ev_grads)
    assert ad_grads and any(g.abs().sum().item() > 0 for g in ad_grads)

    val_rep = train.evaluate_validation(world, {2: encoded}, enc)
    assert val_rep['horizons']['H=2']['evaluated_windows'] == 1

    ckpt_path = tmp_path / 'best.pt'
    cfg = train.TrainConfig(out_dir=tmp_path)
    sha = train.save_checkpoint(
        ckpt_path, world, roles, cfg, {'train': 'deadbeef'}, {'git_head': ''},
        1, {'train_loss': 1.0}, val_rep,
    )
    loaded = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    assert 'train_metrics' in loaded and 'val_metrics' in loaded
    reloaded = train.verify_checkpoint_reload(ckpt_path, torch.device('cpu'))
    assert reloaded['sha256'] == sha
    assert reloaded['param_count'] == sum(p.numel() for p in world.parameters())


def test_config_validation_rejects_invalid_and_empty_inputs(
    tmp_path: pathlib.Path,
) -> None:
    _, roles = observation.load_role_vocab()
    with pytest.raises(ValueError, match='epochs'):
        train.validate_train_config(
            train.TrainConfig(epochs=0, out_dir=tmp_path), roles, 10, 5
        )
    with pytest.raises(ValueError, match='trajectory_batch_size'):
        train.validate_train_config(
            train.TrainConfig(trajectory_batch_size=0, out_dir=tmp_path),
            roles, 10, 5,
        )
    with pytest.raises(ValueError, match='grad_accum'):
        train.validate_train_config(
            train.TrainConfig(grad_accum=0, out_dir=tmp_path), roles, 10, 5
        )
    with pytest.raises(ValueError, match='lr'):
        train.validate_train_config(
            train.TrainConfig(lr=0.0, out_dir=tmp_path), roles, 10, 5
        )
    with pytest.raises(ValueError, match='Training windows'):
        train.validate_train_config(
            train.TrainConfig(out_dir=tmp_path), roles, 0, 5
        )
    with pytest.raises(ValueError, match='Validation windows'):
        train.validate_train_config(
            train.TrainConfig(out_dir=tmp_path), roles, 10, 0
        )


def test_temporal_consistency_and_no_future_leakage() -> None:
    enc = wam2_helpers.FakeSemanticEncoder()
    role_map, roles = observation.load_role_vocab()
    world = model.WorldActionModel(role_count=len(roles), slot_count=18)
    world.eval()

    obs0 = wam2_helpers.make_observation([
        wam2_helpers.make_element(
            1, 'button', 'Next', match_key='nxt'
        ),
    ])
    obs1 = wam2_helpers.make_observation([
        wam2_helpers.make_element(
            1, 'button', 'Next', states={'disabled': True}, match_key='nxt'
        ),
    ])
    obs2_a = wam2_helpers.make_observation([
        wam2_helpers.make_element(
            1, 'button', 'Next', states={'disabled': False}, match_key='nxt'
        ),
    ])
    obs2_b = wam2_helpers.make_observation([
        wam2_helpers.make_element(
            1, 'button', 'Different', states={'disabled': True}, match_key='nxt'
        ),
    ])
    win_a = training_data.ExperienceWindow(
        observations=(obs0, obs1, obs2_a),
        actions=(
            executor.Action(executor.CLICK, target=1),
            executor.Action(executor.SCROLL, scroll_y=10.0),
        ),
        pointers=((10.0, 10.0), (10.0, 10.0)),
        run_id='r',
        steps=(0, 1),
    )
    win_b = training_data.ExperienceWindow(
        observations=(obs0, obs1, obs2_b),
        actions=win_a.actions,
        pointers=win_a.pointers,
        run_id='r',
        steps=(0, 1),
    )
    enc_a = train.encode_windows((win_a,), enc, role_map, slot_count=18)[0]
    enc_b = train.encode_windows((win_b,), enc, role_map, slot_count=18)[0]

    # Step-0 imagined prediction must be identical regardless of obs2.
    with torch.no_grad():
        b0_a = observation.observe(world, enc_a.aligned_obs[0], previous=None)
        b0_b = observation.observe(world, enc_b.aligned_obs[0], previous=None)
        lat_a = world.encode_action(
            b0_a, enc_a.step0_action.kind, enc_a.step0_action.target,
            enc_a.step0_action.text, enc_a.step0_action.checked,
            enc_a.step0_action.scroll, enc_a.step0_action.pointer,
        )
        lat_b = world.encode_action(
            b0_b, enc_b.step0_action.kind, enc_b.step0_action.target,
            enc_b.step0_action.text, enc_b.step0_action.checked,
            enc_b.step0_action.scroll, enc_b.step0_action.pointer,
        )
        p_a = world.imagine(b0_a, lat_a, enc_a.step0_action.target)
        p_b = world.imagine(b0_b, lat_b, enc_b.step0_action.target)
        torch.testing.assert_close(p_a.belief.latent, p_b.belief.latent)
