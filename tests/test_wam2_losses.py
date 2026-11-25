"""Tests for masked loss math, direction-aware losses, and feedback."""

from __future__ import annotations

import pytest
import torch

from browser_world.interaction import executor
from browser_world.wam2 import losses
from browser_world.wam2 import model
from browser_world.wam2 import observation
from tests import wam2_helpers


def _devices() -> list[torch.device]:
    devs = [torch.device('cpu')]
    if torch.cuda.is_available():
        devs.append(torch.device('cuda'))
    return devs


@pytest.mark.parametrize('device', _devices())
def test_masked_mean_exact_cases_cpu_and_cuda(device: torch.device) -> None:
    # Case A: latent features [1, 3, 256] with [1, 3, 1] mask.
    out_a = losses.masked_mean(
        torch.ones((1, 3, 256), device=device),
        torch.ones((1, 3, 1), dtype=torch.bool, device=device),
    )
    torch.testing.assert_close(out_a, torch.tensor(1.0, device=device))

    # Case B: birth-state features [1, 3, 8] with [1, 3, 1] mask.
    out_b = losses.masked_mean(
        torch.ones((1, 3, 8), device=device),
        torch.ones((1, 3, 1), dtype=torch.bool, device=device),
    )
    torch.testing.assert_close(out_b, torch.tensor(1.0, device=device))

    # Case C: geometry and flags [1, 3, 4] with [1, 3, 1] mask.
    out_c = losses.masked_mean(
        torch.ones((1, 3, 4), device=device),
        torch.ones((1, 3, 1), dtype=torch.bool, device=device),
    )
    torch.testing.assert_close(out_c, torch.tensor(1.0, device=device))

    # Case D: partial supervision and exact gradients.
    vals_d = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        device=device,
        requires_grad=True,
    )
    mask_d = torch.tensor(
        [[True], [False], [True]], dtype=torch.bool, device=device
    )
    out_d = losses.masked_mean(vals_d, mask_d)
    torch.testing.assert_close(out_d, torch.tensor(3.5, device=device))
    out_d.backward()
    expected_grad = torch.tensor(
        [[0.25, 0.25], [0.0, 0.0], [0.25, 0.25]], device=device
    )
    assert vals_d.grad is not None
    torch.testing.assert_close(vals_d.grad, expected_grad)

    # Case E: empty supervision returns finite 0 with zero gradients.
    vals_e = torch.tensor([[1.0, 2.0]], device=device, requires_grad=True)
    empty_mask = torch.zeros((1, 2), dtype=torch.bool, device=device)
    out_e = losses.masked_mean(vals_e, empty_mask)
    assert torch.isfinite(out_e)
    torch.testing.assert_close(out_e, torch.tensor(0.0, device=device))
    out_e.backward()
    assert vals_e.grad is not None
    torch.testing.assert_close(vals_e.grad, torch.zeros_like(vals_e))

    # Case F: non-broadcastable mask raises RuntimeError.
    with pytest.raises(RuntimeError):
        losses.masked_mean(
            torch.ones((1, 3, 4), device=device),
            torch.ones((1, 2, 4), dtype=torch.bool, device=device),
        )

    # Case G: exact full-shape mask.
    vals_g = torch.tensor([[2.0, 4.0], [6.0, 8.0]], device=device)
    mask_g = torch.tensor([[True, False], [False, True]], device=device)
    torch.testing.assert_close(
        losses.masked_mean(vals_g, mask_g), torch.tensor(5.0, device=device)
    )

    # Case H: batch dimension with differing valid counts.
    vals_h = torch.tensor(
        [[[1.0, 3.0], [5.0, 7.0]], [[9.0, 11.0], [13.0, 15.0]]],
        device=device,
    )
    mask_h = torch.tensor(
        [[[True], [False]], [[True], [True]]], dtype=torch.bool, device=device
    )
    torch.testing.assert_close(
        losses.masked_mean(vals_h, mask_h),
        torch.tensor(52.0 / 6.0, device=device),
    )


def test_masked_mean_rejects_non_bool_and_device_mismatch() -> None:
    with pytest.raises(TypeError, match='bool'):
        losses.masked_mean(torch.ones((2, 2)), torch.ones((2, 2)))
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match='device'):
            losses.masked_mean(
                torch.ones((2, 2), device='cuda'),
                torch.ones((2, 2), dtype=torch.bool, device='cpu'),
            )


def test_controlled_directional_state_and_existence_strata() -> None:
    # Verify exact logits/targets mapping for 0->0, 0->1, 1->0, 1->1.
    dummy = torch.zeros((1, 2, 1))
    z2o = torch.full((1, 2, 8), -2.0)
    o2z = torch.full((1, 2, 8), 2.0)
    b = model.Belief(
        torch.zeros((1, 2, 256)), torch.ones((1, 2)),
        torch.zeros((1, 2, 8)), torch.zeros((1, 2, 4)),
        torch.zeros((1, 2, 4)), torch.zeros((1, 2, 4)),
        torch.zeros((1, 2, 4)), torch.zeros((1, 2, 3)),
        torch.zeros((1, 64)),
    )
    pred = model.Prediction(
        b, torch.zeros((1, 2)), o2z, z2o, torch.zeros((1, 2, 8)),
        torch.zeros((1, 2, 4)), torch.zeros((1, 2, 4)),
        torch.zeros((1, 2, 4)), torch.zeros((1, 2, 3)),
        torch.zeros((1, 2, 2, 9)),
    )
    prev_states = torch.zeros((1, 2, 8))
    prev_states[0, 1, 0] = 1.0  # slot 1 feat 0 starts at 1
    next_states = torch.zeros((1, 2, 8))
    next_states[0, 0, 0] = 1.0  # slot 0 feat 0: 0 -> 1
    next_states[0, 1, 0] = 0.0  # slot 1 feat 0: 1 -> 0 (rest 0->0)
    nxt_inputs = observation.ObservationInputs(
        names=dummy, values=dummy, roles=dummy, states=next_states,
        flags=dummy, boxes=dummy, affordances=dummy, parents=dummy,
        present=torch.ones((1, 2)),
        box_known=torch.ones((1, 2), dtype=torch.bool),
        matched=torch.ones((1, 2), dtype=torch.bool),
        existence_known=torch.ones((1, 2), dtype=torch.bool),
        element_to_slot=(0, 1), slot_to_element=(0, 1),
        parent_known=torch.ones((1, 2), dtype=torch.bool),
    )
    dir_l, sup = losses.directional_state_loss(
        pred, prev_states, torch.ones((1, 2)), nxt_inputs
    )
    # 1->1 is unsupported (support=0) and excluded from average.
    assert sup == {'0->0': 14, '0->1': 1, '1->0': 1, '1->1': 0}
    expected_00 = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(-2.0), torch.tensor(0.0)
    )
    expected_01 = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(-2.0), torch.tensor(1.0)
    )
    expected_10 = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(2.0), torch.tensor(1.0)
    )
    torch.testing.assert_close(
        dir_l, (expected_00 + expected_01 + expected_10) / 3.0
    )


def test_directional_state_strata_and_structured_feedback() -> None:
    enc = wam2_helpers.FakeSemanticEncoder()
    role_map, roles = observation.load_role_vocab()
    world = model.WorldActionModel(role_count=len(roles), slot_count=18)

    obs0 = wam2_helpers.make_observation([
        wam2_helpers.make_element(
            1, 'combobox', 'Service', match_key='svc',
            states={'disabled': False, 'focused': True},
        ),
        wam2_helpers.make_element(
            2, 'combobox', 'Day', match_key='day',
            states={'disabled': True},
        ),
    ])
    obs1 = wam2_helpers.make_observation([
        wam2_helpers.make_element(
            1, 'combobox', 'Service', match_key='svc',
            states={'disabled': True, 'focused': True},
        ),
        wam2_helpers.make_element(
            2, 'combobox', 'Day', match_key='day',
            states={'disabled': False},
        ),
        wam2_helpers.make_element(
            3, 'button', 'Next', match_key='nxt',
        ),
    ])
    al0 = observation.encode_observation(obs0, enc, role_map, slot_count=18)
    al1 = observation.encode_observation(
        obs1, enc, role_map, slot_count=18, previous=al0
    )
    b0 = observation.observe(world, al0, previous=None)
    act_in = observation.encode_action(
        executor.Action(executor.SCROLL, scroll_y=100.0), al0, enc, (0.0, 0.0)
    )
    act_lat = world.encode_action(
        b0, act_in.kind, act_in.target, act_in.text,
        act_in.checked, act_in.scroll, act_in.pointer,
    )
    pred = world.imagine(b0, act_lat, act_in.target)
    b1 = observation.observe(world, al1, previous=b0)

    step_l = losses.step_dynamics_losses(
        pred, b0.present, b0.states, al1.inputs, b1.latent
    )
    assert step_l.direction_support['0->0'] > 0
    assert step_l.direction_support['0->1'] > 0
    assert step_l.direction_support['1->0'] > 0
    assert step_l.direction_support['1->1'] > 0
    assert step_l.existence_support['persistence'] == 2
    assert step_l.existence_support['birth'] == 1
    assert step_l.existence_support['empty'] == 15

    inv_l, _, _ = losses.inverse_dynamics_loss(world, b0, b1, act_in)
    (step_l.total + inv_l).backward()
    assert all(
        torch.isfinite(p.grad).all()
        for p in world.parameters() if p.grad is not None
    )

    b0_mod = model.Belief(
        b0.latent.detach(), b0.present.detach(),
        1.0 - b0.states.detach(), b0.roles.detach(), b0.flags.detach(),
        b0.boxes.detach(), b0.affordances.detach(), b0.parents.detach(),
        b0.context.detach(),
    )
    b0_det = model.Belief(
        b0.latent.detach(), b0.present.detach(), b0.states.detach(),
        b0.roles.detach(), b0.flags.detach(), b0.boxes.detach(),
        b0.affordances.detach(), b0.parents.detach(), b0.context.detach(),
    )
    p_orig = world.imagine(b0_det, act_lat.detach(), act_in.target)
    p_mod = world.imagine(b0_mod, act_lat.detach(), act_in.target)
    assert not torch.allclose(p_orig.belief.latent, p_mod.belief.latent)
