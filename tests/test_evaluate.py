"""Unit tests for validation decoding and metric calculation."""

# pylint: disable=missing-function-docstring

import math

import torch
from torch.utils import data as torch_data

from browser_world.dataset import schema
from browser_world.dataset import training_data
from browser_world.experiments import evaluate
from browser_world.model import world_model


def _synthetic_example(run_id: str, step: int, num_elems: int, target: int):
    elements = []
    target_elems = []
    for i in range(num_elems):
        elements.append({
            'role': 2,
            'name_words': [0] * 8,
            'name_chars': [0] * 32,
            'value_chars': [0] * 32,
            'has_value': True,
            'states': [0] * 8,
            'flags': [1, 1, 0, 0],
            'box': [0.1, 0.1, 0.2, 0.05],
            'box_mask': True,
            'parent': -1,
            'name_bytes': [65 + i],
            'value_bytes': [66],
        })
        target_elems.append({
            'known': True,
            'removed': False,
            'changed': i == 1,
            'matched': True,
            'next_states': [0] * 8,
            'value_source': 0,
            'value_mask': True,
            'box_delta': [0.0, 0.0, 0.0, 0.0],
            'box_mask': True,
        })
    return {
        'meta': {
            'run_id': run_id,
            'step': step,
            'split': 'validation',
            'family': 'f1',
            'action_kind': 'click',
        },
        'inputs': {
            'state': {'elements': elements, 'page': [0.0, 0.0, 1.0, 1.0]},
            'action': {
                'kind': schema.ACTION_KINDS.index('click'),
                'target': target,
                'text_chars': [0] * 32,
                'text_bytes': [],
                'checked': 0,
                'scroll': 0.0,
                'pointer': [0.2, 0.2],
            },
        },
        'targets': {
            'elements': target_elems,
            'new': [{'role': 3, 'name_words': [0] * 8, 'parent_before': -1}],
            'outcomes': [1, 0, 0, 0, 0, 0, 0, 0],
        },
    }


def test_decode_predictions_thresholds_argmax_and_padding():
    element_mask = torch.tensor([[[True, True, False]]])
    predictions = {
        'changed_logits': torch.tensor([[[0.0, -0.1, 10.0]]]),
        'removed_logits': torch.tensor([[[-0.5, 0.2, 10.0]]]),
        'next_state_logits': torch.tensor([[[
            [0.0, -1.0, 1.0, -0.1, 0.5, -0.5, 0.0, -2.0],
            [-1.0] * 8,
            [5.0] * 8,
        ]]]),
        'value_logits': torch.tensor([[[
            [0.1, 0.9, 0.2, 0.0, -1.0],
            [0.5, 0.1, 0.2, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.0, 9.0],
        ]]]),
        'box_delta': torch.tensor([[[
            [0.01, -0.02, 0.0, 0.005],
            [0.0, 0.0, 0.0, 0.0],
            [9.0, 9.0, 9.0, 9.0],
        ]]]),
        'outcome_logits': torch.tensor([[[
            0.0, -0.1, 2.0, -3.0, 0.1, -0.5, -1.0, 0.0,
        ]]]),
        'new_count': torch.tensor([[1.75]]),
    }

    decoded = evaluate.decode_predictions(predictions, element_mask)
    assert len(decoded) == 1
    item = decoded[0]
    assert item['changed'] == [1, 0]
    assert item['removed'] == [0, 1]
    assert item['next_states'][0] == [1, 0, 1, 0, 1, 0, 1, 0]
    assert item['value_source'] == [1, 3]
    assert len(item['box_delta']) == 2
    assert math.isclose(item['new_count'], 1.75, rel_tol=1e-6)
    assert item['outcomes'] == [1, 0, 1, 0, 1, 0, 0, 1]


def test_evaluate_model_identity_matching_and_metrics():
    ex_a = _synthetic_example('run-a', 0, num_elems=3, target=0)
    ex_b = _synthetic_example('run-b', 0, num_elems=2, target=0)
    # Reverse order in examples vs dataset to prove identity matching.
    dataset = training_data.Windows([[ex_a], [ex_b]])
    loader = torch_data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=training_data.collate,
    )
    torch.manual_seed(0)
    model = world_model.WorldModel(num_roles=16)
    metrics = evaluate.evaluate_model(
        model,
        loader,
        dataset,
        [ex_b, ex_a],
        torch.device('cpu'),
    )
    assert 'total' in metrics['loss']
    assert 'changed_element' in metrics['baseline_format']
    assert 'f1' in metrics['non_target_changed']
    assert 'f1' in metrics['surviving_non_target_changed']
    assert 'f1' in metrics['beyond_action_copy']
    assert isinstance(metrics['removed_not_changed_count'], int)
    assert set(metrics['state_changes']) == set(schema.STATE_NAMES)
    assert set(metrics['per_action']) == set(schema.ACTION_KINDS)
    assert set(metrics['per_run']) == {'run-a', 'run-b'}
    assert set(metrics['per_outcome']) == set(schema.OUTCOMES)
    assert len(metrics['per_role_new_count_mae']) == 15


def test_model_inputs_action_blind_invariance():
    ex_a = _synthetic_example('r1', 0, num_elems=3, target=0)
    ex_b = _synthetic_example('r1', 0, num_elems=3, target=2)
    ex_b['inputs']['action'] = {
        'kind': schema.ACTION_KINDS.index('type'),
        'target': 2,
        'text_chars': [0] * 32,
        'text_bytes': [115, 97, 109],
        'checked': 2,
        'scroll': 0.5,
        'pointer': [0.9, 0.8],
    }
    ds_a = training_data.Windows([[ex_a]])
    ds_b = training_data.Windows([[ex_b]])
    batch_a = training_data.collate([ds_a[0]])
    batch_b = training_data.collate([ds_b[0]])
    snap_a = {k: v.clone() for k, v in batch_a.items()}

    blind_a = evaluate.model_inputs(batch_a, action_blind=True)
    blind_b = evaluate.model_inputs(batch_b, action_blind=True)

    for k, v in batch_a.items():
        assert torch.equal(v, snap_a[k])

    for k in world_model.INPUT_KEYS:
        if k == 'action_text_bytes':
            assert (blind_a[k] == 0).all()
            assert (blind_b[k] == 0).all()
        else:
            assert torch.equal(blind_a[k], blind_b[k])
