"""Reference baselines and metrics over curated targets.

Any learned world model must beat these before it counts:

  * no-change: predicts that nothing happens.
  * action-copy: applies only the action's direct effect (typed text, the
    chosen option, the requested check state, the requested scroll) and
    predicts no other consequence: no dependent updates, no alerts, no
    step changes, no bookings.

Metrics respect every target mask: ambiguous elements, unmatched elements
and values that do not exist are never scored.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import pathlib
from typing import Any

from browser_world import curate
from browser_world import training_data

Example = Mapping[str, Any]
Prediction = dict[str, Any]
_CHECKED = curate.STATE_NAMES.index('checked')
_TYPED = curate.VALUE_SOURCES.index('typed')
_OPTION = curate.VALUE_SOURCES.index('option')


def predict_no_change(example: Example) -> Prediction:
    """Predicts an unchanged page."""
    elements = example['inputs']['state']['elements']
    outcomes = [0] * len(curate.OUTCOMES)
    outcomes[curate.OUTCOMES.index('no_change')] = 1
    return {'changed': [0] * len(elements), 'removed': [0] * len(elements),
            'next_states': [list(e['states']) for e in elements],
            'value_source': [0] * len(elements),
            'box_delta': [[0.0] * 4 for _ in elements],
            'new_count': 0, 'outcomes': outcomes}


def _dropdown_of(elements: Sequence[Mapping[str, Any]], index: int) -> int:
    """Walks parents from an option to its dropdown (role id unknown here,
    so the nearest ancestor with a value is taken)."""
    parent = elements[index]['parent']
    while parent >= 0 and not elements[parent]['has_value']:
        parent = elements[parent]['parent']
    return parent


def predict_action_copy(example: Example) -> Prediction:
    """Predicts only the action's direct effect."""
    prediction = predict_no_change(example)
    elements = example['inputs']['state']['elements']
    action = example['inputs']['action']
    kind = curate.ACTION_KINDS[action['kind']]
    target = action['target']
    if kind == 'type':
        prediction['changed'][target] = 1
        prediction['value_source'][target] = _TYPED
    elif kind == 'check':
        prediction['changed'][target] = 1
        prediction['next_states'][target][_CHECKED] = int(
            action['checked'] == 2)
    elif kind == 'select':
        dropdown = _dropdown_of(elements, target)
        if dropdown >= 0:
            prediction['changed'][dropdown] = 1
            prediction['value_source'][dropdown] = _OPTION
    elif kind == 'scroll':
        for i, element in enumerate(elements):
            if element['box_mask'] and not element['flags'][3]:  # Pinned.
                prediction['box_delta'][i] = [0.0, -action['scroll'], 0.0,
                                              0.0]
                prediction['changed'][i] = 1
        prediction['outcomes'][curate.OUTCOMES.index('scroll_changed')] = 1
    if kind != 'click':
        prediction['outcomes'][curate.OUTCOMES.index('no_change')] = 0
    return prediction


def _f1(true_pos: int, false_pos: int, false_neg: int) -> dict[str, float]:
    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos \
        else 0.0
    recall = true_pos / (true_pos + false_neg) if true_pos + false_neg \
        else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {'precision': round(precision, 3), 'recall': round(recall, 3),
            'f1': round(f1, 3)}


def evaluate(examples: Sequence[Example],
             predict: Callable[[Example], Prediction]) -> dict[str, Any]:
    """Scores predictions against masked curated targets."""
    changed = [0, 0, 0]  # True positives, false positives, false negatives.
    removed = [0, 0, 0]
    states_right = states_total = 0
    changed_states_right = changed_states_total = 0
    values_right = values_total = 0
    box_error = box_total = 0.0
    new_error = 0.0
    outcome_counts = {name: [0, 0, 0] for name in curate.OUTCOMES}
    for example in examples:
        prediction = predict(example)
        for i, target in enumerate(example['targets']['elements']):
            if not target['known']:
                continue
            for counts, key in ((changed, 'changed'), (removed, 'removed')):
                truth, guess = bool(target[key]), bool(prediction[key][i])
                counts[0] += truth and guess
                counts[1] += guess and not truth
                counts[2] += truth and not guess
            if target['matched']:
                same = prediction['next_states'][i] == target['next_states']
                states_right += same
                states_total += 1
                if target['changed']:
                    changed_states_right += same
                    changed_states_total += 1
            if target['value_mask']:
                values_right += (prediction['value_source'][i]
                                 == target['value_source'])
                values_total += 1
            if target['box_mask']:
                box_error += sum(abs(p - t) for p, t in zip(
                    prediction['box_delta'][i], target['box_delta'])) / 4
                box_total += 1
        new_error += abs(prediction['new_count']
                         - len(example['targets']['new']))
        for name, truth, guess in zip(curate.OUTCOMES,
                                      example['targets']['outcomes'],
                                      prediction['outcomes']):
            counts = outcome_counts[name]
            counts[0] += bool(truth) and bool(guess)
            counts[1] += bool(guess) and not truth
            counts[2] += bool(truth) and not guess
    count = max(len(examples), 1)
    return {
        'examples': len(examples),
        'changed_element': _f1(*changed),
        'removed_element': _f1(*removed),
        'next_states_exact_all': round(states_right / max(states_total, 1),
                                       3),
        'next_states_exact_on_changed': round(
            changed_states_right / max(changed_states_total, 1), 3),
        'value_source_accuracy': round(values_right / max(values_total, 1),
                                       3),
        'box_delta_mae': round(box_error / max(box_total, 1), 4),
        'new_count_mae': round(new_error / count, 3),
        'outcome_f1': {name: _f1(*c)['f1']
                       for name, c in outcome_counts.items()},
    }


def run(curated_dir: pathlib.Path) -> dict[str, Any]:
    """Evaluates both baselines on the validation and test splits."""
    results = {}
    for split in ('validation', 'test'):
        examples = training_data.load_examples(curated_dir, split)
        results[split] = {
            'no_change': evaluate(examples, predict_no_change),
            'action_copy': evaluate(examples, predict_action_copy)}
    return results


if __name__ == '__main__':
    print(json.dumps(run(pathlib.Path('data/curated/c01')), indent=1))
