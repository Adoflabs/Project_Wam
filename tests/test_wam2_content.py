"""Tests for exact content representation, pointer head, and resolver."""

from __future__ import annotations

import pathlib

import pytest
import torch

from browser_world.wam2 import content
from browser_world.wam2 import losses
from browser_world.wam2 import model
from browser_world.wam2 import observation
from browser_world.wam2 import training_data
from tests import wam2_helpers


def test_exact_text_invariants_and_unicode_preservation() -> None:
    empty = content.ExactText(content.ContentStatus.KNOWN, '')
    absent = content.ExactText(content.ContentStatus.ABSENT, None)
    unk = content.ExactText(content.ContentStatus.UNKNOWN, None)
    assert empty.text == ''
    assert absent != unk

    uni = content.ExactText(
        content.ContentStatus.KNOWN, 'शहर में बेंगलुरु — 15:30'
    )
    assert uni.text == 'शहर में बेंगलुरु — 15:30'

    with pytest.raises(ValueError, match='KNOWN'):
        content.ExactText(content.ContentStatus.KNOWN, None)
    with pytest.raises(ValueError, match='ABSENT'):
        content.ExactText(content.ContentStatus.ABSENT, '')
    with pytest.raises(ValueError, match='UNKNOWN'):
        content.ExactText(content.ContentStatus.UNKNOWN, 'x')


def test_observation_content_and_legal_masks() -> None:
    enc = wam2_helpers.FakeSemanticEncoder()
    role_map, _ = observation.load_role_vocab()
    obs = wam2_helpers.make_observation([
        wam2_helpers.make_element(
            1, 'textbox', 'Your name', value='', match_key='name'
        ),
        wam2_helpers.make_element(
            2, 'button', 'Next', value=None, match_key='nxt'
        ),
    ])
    al = observation.encode_observation(obs, enc, role_map, slot_count=20)
    assert al.exact_content[0].name == content.ExactText(
        content.ContentStatus.KNOWN, 'Your name'
    )
    assert al.exact_content[0].value == content.ExactText(
        content.ContentStatus.KNOWN, ''
    )
    assert al.exact_content[1].value == content.ExactText(
        content.ContentStatus.ABSENT, None
    )
    assert al.exact_content[2].name == content.ExactText(
        content.ContentStatus.UNKNOWN, None
    )

    mask_no_act = content.legal_source_mask(
        al.exact_content, al.inputs.present, None, torch.device('cpu')
    )
    assert mask_no_act.shape == (1, 20, 2, 45)
    assert not bool(mask_no_act[0, 0, 1, content.ACTION_TEXT])
    assert bool(mask_no_act[0, 0, 1, content.KEEP])
    # Slot 2 was unoccupied: KEEP and COPY_* from slot 2 must be False.
    assert not bool(mask_no_act[0, 2, 0, content.KEEP])
    assert not bool(
        mask_no_act[0, 0, 0, content.FIXED_SOURCE_COUNT + 2 * 2]
    )
    # Absent value on slot 1 cannot be copied as a literal string.
    assert not bool(
        mask_no_act[0, 0, 1, content.FIXED_SOURCE_COUNT + 2 * 1 + 1]
    )


def test_targets_multiple_sources_and_set_valued_loss() -> None:
    unk = content.ExactText(content.ContentStatus.UNKNOWN, None)
    absent = content.ExactText(content.ContentStatus.ABSENT, None)
    known = lambda s: content.ExactText(content.ContentStatus.KNOWN, s)

    prev = (
        content.EntityContent(known('City'), known('Delhi')),
        content.EntityContent(known('Mumbai'), absent),
        content.EntityContent(unk, unk),
    )
    prev_p = torch.tensor([[1.0, 1.0, 0.0]])
    after = (
        content.EntityContent(known('City'), known('Mumbai')),
        content.EntityContent(known('Mumbai'), absent),
        content.EntityContent(known('New Opt'), absent),
    )
    after_p = torch.tensor([[1.0, 1.0, 1.0]])
    ex_known = torch.ones((1, 3), dtype=torch.bool)

    legal = content.legal_source_mask(
        prev, prev_p, 'Mumbai', torch.device('cpu')
    )
    targets, elig = content.build_content_targets(
        prev, prev_p, after, after_p, ex_known, 'Mumbai', torch.device('cpu')
    )
    # Slot 0 value 'Mumbai': valid from ACTION_TEXT and COPY_NAME_FROM_SLOT_1.
    assert bool(targets[0, 0, 1, content.ACTION_TEXT])
    assert bool(targets[0, 0, 1, content.FIXED_SOURCE_COUNT + 2 * 1])
    assert not bool(targets[0, 0, 1, content.KEEP])
    # Slot 2 new entity 'New Opt' unavailable -> UNKNOWN.
    assert bool(targets[0, 2, 0, content.UNKNOWN])

    logits = torch.zeros((1, 3, 2, 11), requires_grad=True)
    loss = losses.set_valued_content_loss(logits, legal, targets, elig)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    # Exact arithmetic check on 1-field 2-target vs 4-legal case.
    l_1 = torch.tensor([[[[1.0, 2.0, 0.0, 3.0]]]], requires_grad=True)
    leg_1 = torch.tensor([[[[True, True, False, True]]]])
    tgt_1 = torch.tensor([[[[True, True, False, False]]]])
    el_1 = torch.tensor([[[True]]])
    out = losses.set_valued_content_loss(l_1, leg_1, tgt_1, el_1)
    expected = (
        torch.logsumexp(torch.tensor([1.0, 2.0, 3.0]), dim=0)
        - torch.logsumexp(torch.tensor([1.0, 2.0]), dim=0)
    )
    torch.testing.assert_close(out, expected)


def test_recursive_resolver_and_independent_branches() -> None:
    known = lambda s: content.ExactText(content.ContentStatus.KNOWN, s)
    absent = content.ExactText(content.ContentStatus.ABSENT, None)
    unk = content.ExactText(content.ContentStatus.UNKNOWN, None)

    prev = (
        content.EntityContent(known('Name'), known('')),
        content.EntityContent(known('Doctor'), absent),
        content.EntityContent(unk, unk),
    )
    prev_p = torch.tensor([[1.0, 1.0, 0.0]])
    pred_p = torch.tensor([[1.0, 1.0, 1.0]])

    # Branch A: types 'Ada', keeps Doctor.
    sel_a = torch.tensor([[[content.KEEP, content.ACTION_TEXT],
                           [content.KEEP, content.KEEP],
                           [content.UNKNOWN, content.ABSENT]]])
    res_a = content.resolve_content(prev, prev_p, 'Ada', sel_a, pred_p)
    # Branch B: copies Doctor name into Name value.
    sel_b = torch.tensor([[[content.KEEP, content.FIXED_SOURCE_COUNT + 2 * 1],
                           [content.KEEP, content.KEEP],
                           [content.KEEP, content.ABSENT]]])
    res_b = content.resolve_content(prev, prev_p, None, sel_b, pred_p)

    assert res_a[0].value == known('Ada')
    assert res_b[0].value == known('Doctor')
    assert res_a[2].name == unk
    assert res_b[2].name == unk  # KEEP on new slot resolves to UNKNOWN.
    assert prev[0].value == known('')  # Original branch untouched.


def test_model_pointer_dims_checkpoint_compat_and_imagined_guard() -> None:
    ckpt_path = pathlib.Path('artifacts/wam2/dynamics_v2_corrected/best.pt')
    orig_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    world, _ = content.load_from_dynamics_checkpoint(
        ckpt_path, torch.device('cpu')
    )
    assert sum(p.numel() for p in world.parameters()) == 3589686
    for k, v in orig_ckpt['model_state_dict'].items():
        torch.testing.assert_close(world.state_dict()[k], v)

    enc = wam2_helpers.FakeSemanticEncoder()
    role_map, _ = observation.load_role_vocab()
    obs = wam2_helpers.make_observation([
        wam2_helpers.make_element(1, 'button', 'Next', match_key='nxt')
    ])
    al = observation.encode_observation(obs, enc, role_map)
    belief = observation.observe(world, al)
    act = world.encode_action(
        belief, torch.tensor([4]), torch.tensor([-1]),
        torch.zeros((1, 384)), torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.1]]), torch.tensor([[0.0, 0.0]]),
    )
    pred = world.imagine(belief, act, torch.tensor([-1]))
    assert pred.content_source_logits.shape == (1, 128, 2, 261)

    # Imagined unoccupied slot 5 must be rejected as an execution target.
    with pytest.raises(ValueError, match='Imagined or unoccupied slot'):
        training_data.resolve_observed_node_id(5, al)
