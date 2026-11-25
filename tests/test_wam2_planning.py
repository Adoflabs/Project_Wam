"""Tests for WAM2 model-driven planning and candidate prediction."""

from __future__ import annotations

import torch

from browser_world.interaction import executor
from browser_world.wam2 import model
from browser_world.wam2 import observation
from browser_world.wam2 import planning
from tests import wam2_helpers


def _setup_obs_and_model() -> tuple[
    model.WorldActionModel,
    wam2_helpers.FakeSemanticEncoder,
    observation.AlignedObservation,
    model.Belief,
]:
    torch.manual_seed(0)
    role_to_idx, _ = observation.load_role_vocab()
    world = model.WorldActionModel(role_count=len(role_to_idx), slot_count=32)
    encoder = wam2_helpers.FakeSemanticEncoder()
    obs = wam2_helpers.make_observation([
        wam2_helpers.make_element(10, 'textbox', 'Name', value=''),
        wam2_helpers.make_element(11, 'button', 'Continue'),
    ])
    aligned = observation.encode_observation(
        obs, encoder, role_to_idx, slot_count=32
    )
    belief = observation.observe(world, aligned)
    return world, encoder, aligned, belief


def test_decode_copy_spans_preserves_unicode_and_rejects_invalid() -> None:
    inst = 'Enter Rahul 🎉 in Name'
    # 4 tokens: 'Enter', 'Rahul ', '🎉', 'in Name'
    offsets = torch.tensor([[0, 5], [6, 12], [12, 13], [14, 21]])
    mask = torch.tensor([True, True, True, False])
    starts = torch.tensor([0.1, 3.0, 0.2, 5.0])
    ends = torch.tensor([0.1, 0.2, 4.0, 5.0])

    spans = planning.decode_copy_spans(
        inst, starts, ends, offsets, mask, max_tokens=2, top_k=3
    )
    assert spans[0] == (6, 13)
    assert inst[spans[0][0]:spans[0][1]] == 'Rahul 🎉'
    for s, e in spans:
        assert (s, e) != (14, 21)


def test_enumerate_candidates_excludes_invalid() -> None:
    world, _, aligned, _ = _setup_obs_and_model()
    del world
    affordances = (
        executor.Affordance(executor.TYPE, 10, 'textbox', 'Name'),
        executor.Affordance(executor.CLICK, 11, 'button', 'Continue'),
        executor.Affordance(executor.CLICK, 999, 'button', 'Ghost'),
        executor.Affordance(executor.SCROLL, None, 'page', 'scroll down',
                            scroll_y=300.0),
    )
    cands = planning.enumerate_candidates(
        aligned,
        'Enter Rahul',
        ((6, 11),),
        affordances,
    )
    assert len(cands) == 3
    assert cands[0].action.kind == executor.TYPE
    assert cands[0].action.text == 'Rahul'
    assert cands[0].target_slot == 0
    assert cands[1].action.kind == executor.CLICK
    assert cands[1].target_slot == 1
    assert cands[2].action.kind == executor.SCROLL
    assert cands[2].target_slot is None


def test_predict_candidates_uses_same_belief_without_executor() -> None:
    world, encoder, aligned, belief = _setup_obs_and_model()
    cands = planning.enumerate_candidates(
        aligned,
        'Enter Rahul',
        ((6, 11),),
        tuple(executor.available_actions(aligned.observation)),
    )
    preds = planning.predict_candidates(
        world, belief, cands, aligned, encoder, batch_size=2
    )
    assert len(preds) == len(cands)
    assert all(p.decision_score is None for p in preds)
    assert not torch.equal(
        preds[0].prediction.belief.latent,
        preds[1].prediction.belief.latent,
    )


def test_untrained_goal_outputs_cannot_drive_selection() -> None:
    world, encoder, aligned, belief = _setup_obs_and_model()
    cands = planning.enumerate_candidates(
        aligned,
        'Enter Rahul',
        ((6, 11),),
        tuple(executor.available_actions(aligned.observation)),
    )
    preds = planning.predict_candidates(world, belief, cands, aligned, encoder)
    toks, mask, _ = encoder.encode(('Enter Rahul',), max_tokens=32)

    untrained = planning.score_predictions(
        world,
        preds,
        toks,
        mask,
        satisfaction_trained=False,
        return_trained=False,
    )
    assert planning.select_candidate(untrained) is None

    trained = planning.score_predictions(
        world,
        preds,
        toks,
        mask,
        satisfaction_trained=True,
        return_trained=True,
    )
    picked = planning.select_candidate(trained)
    assert picked is not None
    best = max(trained, key=lambda p: float(p.decision_score or -1e9))
    assert picked.candidate.candidate_id == best.candidate.candidate_id


def test_imagined_only_slot_cannot_become_real_target() -> None:
    _, _, aligned, _ = _setup_obs_and_model()
    imagined_cand = planning.CandidateAction(
        candidate_id=0,
        action=executor.Action(executor.CLICK, target=10),
        target_slot=15,
        instruction_span=None,
        proposal_logit=1.0,
    )
    assert planning.resolve_candidate_action(imagined_cand, aligned) is None
