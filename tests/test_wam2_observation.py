"""Unit and real-browser integration tests for wam2 observation/action."""

from __future__ import annotations

from playwright import sync_api
import pytest
import torch

from browser_world.dataset import schema
from browser_world.environment import booking
from browser_world.environment import scenarios
from browser_world.interaction import executor
from browser_world.interaction import observer
from browser_world.wam2 import model
from browser_world.wam2 import observation
from browser_world.wam2 import semantic
from tests import wam2_helpers


def test_architectural_constants_contract() -> None:
    assert (
        model.SLOT_DIM
        == model.MECHANISM_DIM + model.STATE_DIM + model.CONTENT_DIM
        == 256
    )
    assert model.STATE_COUNT == len(schema.STATE_NAMES) == len(
        observer.STATE_NAMES
    )
    assert tuple(schema.STATE_NAMES) == tuple(observer.STATE_NAMES)
    assert model.ACTION_COUNT == len(schema.ACTION_KINDS) == 5
    assert model.TEXT_DIM == semantic.TEXT_DIM == 384
    world = model.WorldActionModel(role_count=16, slot_count=128)
    assert sum(p.numel() for p in world.parameters()) == 3589686


def test_slot_assignment_persistence_new_removed_and_capacity() -> None:
    enc = wam2_helpers.FakeSemanticEncoder()
    role_map, _ = observation.load_role_vocab()

    obs0 = wam2_helpers.make_observation([
        wam2_helpers.make_element(1, 'heading', 'Title', match_key='h1'),
        wam2_helpers.make_element(2, 'button', 'Save', match_key='save'),
        wam2_helpers.make_element(3, 'combobox', 'Service', match_key='svc'),
    ])
    al0 = observation.encode_observation(obs0, enc, role_map, slot_count=20)
    assert al0.inputs.element_to_slot == (0, 1, 2)
    assert not al0.inputs.matched.any()

    # Reordered entities, one removed ('save'), one new ('opt1').
    obs1 = wam2_helpers.make_observation([
        wam2_helpers.make_element(3, 'combobox', 'Service', match_key='svc'),
        wam2_helpers.make_element(
            4, 'option', 'Skin', parent_id=3, match_key='opt1'
        ),
        wam2_helpers.make_element(1, 'heading', 'Title', match_key='h1'),
    ])
    al1 = observation.encode_observation(
        obs1, enc, role_map, slot_count=20, previous=al0
    )
    # 'Service' keeps 2, 'Title' keeps 0, removed slot 1 is NOT reused.
    assert al1.inputs.element_to_slot == (2, 3, 0)
    assert bool(al1.inputs.matched[0, 0]) and bool(al1.inputs.matched[0, 2])
    assert not bool(al1.inputs.matched[0, 1])
    assert not bool(al1.inputs.matched[0, 3])
    assert float(al1.inputs.present[0, 1]) == 0.0
    assert bool(al1.allocated_slots[0, 1])

    # Initial observation exceeding slot_count - 16 must raise.
    too_many = wam2_helpers.make_observation(
        [wam2_helpers.make_element(i + 1, 'button', f'B{i}') for i in range(5)]
    )
    with pytest.raises(ValueError, match='exceeding initial limit'):
        observation.encode_observation(too_many, enc, role_map, slot_count=20)


def test_ambiguous_correspondence_and_capacity_exhaustion() -> None:
    enc = wam2_helpers.FakeSemanticEncoder()
    role_map, _ = observation.load_role_vocab()
    obs0 = wam2_helpers.make_observation([
        wam2_helpers.make_element(10, 'button', 'Next'),
        wam2_helpers.make_element(11, 'button', 'Next'),
    ])
    al0 = observation.encode_observation(obs0, enc, role_map, slot_count=18)
    obs1 = wam2_helpers.make_observation(
        [wam2_helpers.make_element(20, 'button', 'Next')]
    )
    al1 = observation.encode_observation(
        obs1, enc, role_map, slot_count=18, previous=al0
    )
    # Ambiguous disappearance & appearance are masked.
    assert not bool(al1.inputs.existence_known[0, 0])
    assert not bool(al1.inputs.existence_known[0, 1])
    assert not bool(al1.inputs.existence_known[0, 2])

    # Exhaust slot_count=17 (1 initial + 16 reserved, add 17th fails).
    al_tiny = observation.encode_observation(
        wam2_helpers.make_observation(
            [wam2_helpers.make_element(1, 'heading', 'A', match_key='a')]
        ),
        enc,
        role_map,
        slot_count=17,
    )
    with pytest.raises(ValueError, match='Slot capacity'):
        observation.encode_observation(
            wam2_helpers.make_observation([
                wam2_helpers.make_element(i + 2, 'heading', f'N{i}')
                for i in range(17)
            ]),
            enc,
            role_map,
            slot_count=17,
            previous=al_tiny,
        )


def test_features_geometry_parents_affordances_action_and_grads() -> None:
    enc = wam2_helpers.FakeSemanticEncoder()
    role_map, roles = observation.load_role_vocab()
    obs0 = wam2_helpers.make_observation(
        [
            wam2_helpers.make_element(
                10, 'combobox', 'Service', match_key='svc', tag='select'
            ),
            wam2_helpers.make_element(
                11, 'option', 'Skin', parent_id=10,
                states={'focused': True},
                box=observer.Box(-50.0, 600.0, 200.0, 50.0),
                match_key='opt',
            ),
            wam2_helpers.make_element(
                12, 'button', 'MissingParent', parent_id=999, box=None
            ),
            wam2_helpers.make_element(13, 'button', 'Next', match_key='nxt'),
        ],
        scroll_y=100.0,
    )
    al0 = observation.encode_observation(obs0, enc, role_map, slot_count=32)

    assert float(al0.inputs.roles[0, 0, roles.index('combobox')]) == 1.0
    assert float(al0.inputs.states[0, 1, 3]) == 1.0
    assert torch.allclose(
        al0.inputs.boxes[0, 1],
        torch.tensor([-0.05, 1.0, 0.2, 0.1]),
    )
    assert bool(al0.inputs.box_known[0, 1]) is True
    assert bool(al0.inputs.box_known[0, 2]) is False
    assert float(al0.inputs.parents[0, 0, 0]) == 1.0
    assert float(al0.inputs.parents[0, 1, 1]) == 1.0
    assert bool(al0.inputs.parent_known[0, 1]) is True
    assert float(al0.inputs.parents[0, 2, 0]) == 1.0
    assert bool(al0.inputs.parent_known[0, 2]) is False

    assert float(al0.inputs.affordances[0, 1, 2]) == 1.0
    assert float(al0.inputs.affordances[0, 2, 0]) == 0.0
    assert float(al0.inputs.affordances[0, 3, 0]) == 1.0

    sel_act = executor.Action(executor.SELECT, target=11)
    sel_in = observation.encode_action(sel_act, al0, enc, (500.0, 250.0))
    assert int(sel_in.kind.item()) == 2
    assert int(sel_in.target.item()) == 1
    assert torch.allclose(sel_in.pointer, torch.tensor([[0.5, 0.5]]))

    scr_act = executor.Action(executor.SCROLL, target=None, scroll_y=250.0)
    scr_in = observation.encode_action(scr_act, al0, enc, (0.0, 0.0))
    assert int(scr_in.kind.item()) == 4
    assert int(scr_in.target.item()) == -1
    assert torch.allclose(scr_in.scroll, torch.tensor([[0.5]]))

    world = model.WorldActionModel(role_count=len(roles), slot_count=32)
    belief = observation.observe(world, al0, previous=None)
    act_lat = world.encode_action(
        belief, sel_in.kind, sel_in.target, sel_in.text,
        sel_in.checked, sel_in.scroll, sel_in.pointer,
    )
    assert list(act_lat.shape) == [1, 128]
    pred = world.imagine(belief, act_lat, sel_in.target)
    loss = pred.present_logits.sum() + pred.zero_to_one_logits.sum()
    loss.backward()
    grads = [p.grad for p in world.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_real_browser_cuda_observation_and_imagine_integration(
    browser: sync_api.Browser,
    scenario_server: tuple[str, dict[str, booking.Scenario]],
) -> None:
    if not torch.cuda.is_available():
        pytest.skip(
            'CUDA required for real browser SemanticEncoder integration.'
        )

    device = torch.device('cuda')
    enc = semantic.SemanticEncoder(device)
    role_map, roles = observation.load_role_vocab()
    world = model.WorldActionModel(
        role_count=len(roles), slot_count=model.SLOT_COUNT
    ).to(device)

    base_url, assignments = scenario_server
    assignments['wam2-integ'] = scenarios.generate('riverside-wizard', 90000)

    context = browser.new_context(viewport={'width': 1280, 'height': 700})
    try:
        page = context.new_page()
        watcher = observer.attach(page)
        hand = executor.Executor(page, watcher, executor.FAST_SPEED)
        page.goto(f'{base_url}/?session=wam2-integ')
        page.wait_for_selector('body[data-ready=true]')
        watcher.wait_until_settled()
        before_obs = watcher.observe()

        al_before = observation.encode_observation(before_obs, enc, role_map)
        belief_before = observation.observe(world, al_before, previous=None)
        assert list(belief_before.latent.shape) == [1, 128, 256]
        assert int(al_before.inputs.present.sum().item()) == len(
            before_obs.elements
        )

        opt_aff = next(
            a for a in executor.available_actions(before_obs)
            if a.kind == executor.SELECT and a.name == 'Skin consultation'
        )
        act = opt_aff.action()
        act_in = observation.encode_action(act, al_before, enc, hand.pointer)
        act_lat = world.encode_action(
            belief_before, act_in.kind, act_in.target, act_in.text,
            act_in.checked, act_in.scroll, act_in.pointer,
        )
        pred = world.imagine(belief_before, act_lat, act_in.target)
        assert torch.isfinite(pred.belief.latent).all()

        result = hand.execute(before_obs, act)
        assert result.executed
        watcher.wait_until_settled()
        after_obs = watcher.observe()

        al_after = observation.encode_observation(
            after_obs, enc, role_map, previous=al_before
        )
        belief_after = observation.observe(
            world, al_after, previous=belief_before
        )
        assert torch.isfinite(belief_after.latent).all()
        assert al_after.inputs.matched.any()
        assert int(al_after.allocated_slots.sum().item()) >= len(
            before_obs.elements
        )
    finally:
        context.close()
