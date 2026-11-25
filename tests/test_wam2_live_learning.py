"""Tests for WAM2 online experience learning and checkpoint retention."""

from __future__ import annotations

import pathlib

import torch

from browser_world.interaction import executor
from browser_world.wam2 import live_learning
from browser_world.wam2 import model
from browser_world.wam2 import observation
from browser_world.wam2 import planning
from tests import wam2_helpers


def _build_learner(
    tmp_path: pathlib.Path,
) -> tuple[
    live_learning.LiveLearner,
    dict[str, int],
    wam2_helpers.FakeSemanticEncoder,
]:
    del tmp_path
    torch.manual_seed(0)
    role_to_idx, _ = observation.load_role_vocab()
    world = model.WorldActionModel(role_count=len(role_to_idx), slot_count=32)
    encoder = wam2_helpers.FakeSemanticEncoder()
    learner = live_learning.LiveLearner(
        world,
        encoder,
        role_to_idx,
        live_learning.LearningConfig(learning_rate=5e-3),
    )
    return learner, role_to_idx, encoder


def _make_sample(
    learner: live_learning.LiveLearner,
    role_to_idx: dict[str, int],
    encoder: wam2_helpers.FakeSemanticEncoder,
    *,
    approved_id: int | None,
    rejected_ids: tuple[int, ...],
    literal_span: tuple[int, int] | None = None,
    goal_satisfied: bool | None = None,
) -> live_learning.LearningSample:
    before_obs = wam2_helpers.make_observation([
        wam2_helpers.make_element(10, 'textbox', 'Name', value=''),
        wam2_helpers.make_element(11, 'textbox', 'Email', value=''),
        wam2_helpers.make_element(12, 'button', 'Continue'),
    ])
    after_obs = wam2_helpers.make_observation([
        wam2_helpers.make_element(10, 'textbox', 'Name', value='Rahul'),
        wam2_helpers.make_element(11, 'textbox', 'Email', value=''),
        wam2_helpers.make_element(12, 'button', 'Continue'),
    ])
    b_al = observation.encode_observation(
        before_obs, encoder, role_to_idx, slot_count=32
    )
    a_al = observation.encode_observation(
        after_obs, encoder, role_to_idx, slot_count=32, previous=b_al
    )
    inst = 'Enter Rahul in Name'
    toks, mask, _ = encoder.encode((inst,), max_tokens=32)
    # Build 4 word-level tokens so 'Rahul' at (6, 11) has an exact token span.
    toks = toks.expand(1, 4, -1).clone()
    mask = torch.ones((1, 4), dtype=torch.bool)
    offsets = torch.tensor([[[0, 5], [6, 11], [12, 14], [15, 19]]])
    cands = (
        planning.CandidateAction(
            0, executor.Action(executor.TYPE, 10, text='Rahul'), 0, (6, 11), 0.0
        ),
        planning.CandidateAction(
            1, executor.Action(executor.TYPE, 11, text='Rahul'), 1, (6, 11), 0.0
        ),
        planning.CandidateAction(
            2, executor.Action(executor.CLICK, 12), 2, None, 0.0
        ),
    )
    fb = live_learning.ActionFeedback(
        decision_id='dec-1',
        approved_candidate_id=approved_id,
        rejected_candidate_ids=rejected_ids,
        literal_span=literal_span,
        goal_satisfied=goal_satisfied,
    )
    return live_learning.LearningSample(
        sample_id='s-1',
        decision_id='dec-1',
        before_aligned=b_al,
        after_aligned=a_al,
        executed_candidate=cands[0],
        candidates=cands,
        pointer=(0.0, 0.0),
        instruction=inst,
        instruction_tokens=toks,
        instruction_mask=mask,
        token_offsets=offsets[0],
        feedback=fb,
    )


def test_learn_updates_heads_freezes_backbone_and_improves_preference(
    tmp_path: pathlib.Path,
) -> None:
    learner, role_to_idx, encoder = _build_learner(tmp_path)
    frozen_before = {
        name: p.detach().clone()
        for name, p in learner.world.named_parameters()
        if not p.requires_grad
    }
    sample = _make_sample(
        learner,
        role_to_idx,
        encoder,
        approved_id=0,
        rejected_ids=(1,),
        literal_span=(6, 11),
        goal_satisfied=False,
    )

    with torch.no_grad():
        b0 = observation.observe(learner.world, sample.before_aligned)
        _, prop_before, _ = learner.world.evaluate_goal(
            b0, sample.instruction_tokens, sample.instruction_mask
        )
        margin_before = float(
            (prop_before[0, 0, 1] - prop_before[0, 1, 1]).item()
        )

    report = learner.learn_from_sample(sample)
    assert report.status == 'weight_update_accepted'
    assert report.model_revision_before == 0
    assert report.model_revision_after == 1
    assert report.changed_parameter_tensors > 0
    assert report.parameter_update_norm > 0.0
    assert 'proposal' in report.supervised_objectives
    assert 'pairwise_ranking' in report.supervised_objectives
    assert 'copy' in report.supervised_objectives
    assert 'content_source' in report.supervised_objectives

    for name, p in learner.world.named_parameters():
        if name in frozen_before:
            assert torch.equal(p.detach(), frozen_before[name])

    with torch.no_grad():
        b1 = observation.observe(learner.world, sample.before_aligned)
        _, prop_after, _ = learner.world.evaluate_goal(
            b1, sample.instruction_tokens, sample.instruction_mask
        )
        margin_after = float(
            (prop_after[0, 0, 1] - prop_after[0, 1, 1]).item()
        )
    assert margin_after > margin_before

    ckpt_path = learner.save_checkpoint(tmp_path / 'live_ckpt')
    reloaded, ckpt_meta = live_learning.load_from_live_checkpoint(
        ckpt_path, torch.device('cpu')
    )
    assert ckpt_meta['model_revision'] == 1
    for k, v in learner.world.state_dict().items():
        assert torch.equal(v, reloaded.state_dict()[k])


def test_incomplete_episode_produces_no_return_target() -> None:
    assert live_learning.return_targets_for_episode(3, 'incomplete') is None
    assert live_learning.return_targets_for_episode(3, None) is None
    ret_ok = live_learning.return_targets_for_episode(3, 'success')
    assert ret_ok is not None
    assert len(ret_ok) == 3
    assert ret_ok[2] == 1.0


def test_rejected_update_restores_state_and_revision(
    tmp_path: pathlib.Path,
) -> None:
    learner, role_to_idx, encoder = _build_learner(tmp_path)
    sample = _make_sample(
        learner, role_to_idx, encoder, approved_id=0, rejected_ids=()
    )
    # Force reference degradation rejection by setting negative tolerance.
    learner.config = live_learning.LearningConfig(
        reference_degradation_tolerance=-1.0, reference_absolute_eps=-1.0
    )
    learner._reference_windows = [
        type('W', (), {
            'observations': (
                sample.before_aligned.observation,
                sample.after_aligned.observation,
            ),
            'actions': (sample.executed_candidate.action,),
            'pointers': ((0.0, 0.0),),
        })()
    ]
    snap_before = {
        k: v.detach().clone() for k, v in learner.world.state_dict().items()
    }
    rep = learner.learn_from_sample(sample)
    assert rep.status == 'weight_update_rejected'
    assert rep.model_revision_after == 0
    for k, v in learner.world.state_dict().items():
        assert torch.equal(v, snap_before[k])
