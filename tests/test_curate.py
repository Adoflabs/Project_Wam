"""C0.1: strict validation, alignment, curation, tensors and baselines."""

# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

import json
import pathlib
import shutil

import pytest

from browser_world import baselines
from browser_world import collector
from browser_world import curate
from browser_world import executor
from browser_world import observer
from browser_world import recorder
from browser_world import training_data


@pytest.fixture(scope='module')
def raw_dir(browser, scenario_server, tmp_path_factory):
    """A small real dataset: a branch group plus noisy and exploratory
    episodes, collected once for the module."""
    url, assignments = scenario_server
    out = tmp_path_factory.mktemp('raw')
    context = browser.new_context(viewport={'width': 1280, 'height': 700})
    page = context.new_page()
    watcher = observer.attach(page)
    collect = collector.Collector(page, watcher, url, assignments, out,
                                  executor.FAST_SPEED)
    collect.collect(14, seed_base=40_000)
    context.close()
    return out


def _copy(raw_dir, tmp_path):
    target = tmp_path / 'raw'
    shutil.copytree(raw_dir, target)
    return target


def _rewrite(path: pathlib.Path, change):
    records = recorder.read_jsonl(path)
    change(records)
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))


def _first_split(raw):
    for split in curate.SPLITS:
        path = raw / split / recorder.TRANSITIONS_FILE
        if path.exists() and recorder.read_jsonl(path):
            return split, path
    raise AssertionError('No records.')


def test_real_dataset_passes(raw_dir):
    report = curate.validate(raw_dir)
    assert not report.problems, report.problems


def _problems_after(raw_dir, tmp_path, change, split=None):
    raw = _copy(raw_dir, tmp_path)
    found_split, path = _first_split(raw)
    if split is not None:
        path = raw / split / recorder.TRANSITIONS_FILE
    del found_split
    _rewrite(path, change)
    return curate.validate(raw).problems


def test_corrupted_target_index_is_caught(raw_dir, tmp_path):
    def change(records):
        record = next(r for r in records if r['target'])
        record['target']['index'] = (record['target']['index'] + 1) % len(
            record['before']['elements'])
    problems = _problems_after(raw_dir, tmp_path, change)
    assert any('target' in p for p in problems), problems


def test_wrong_family_assignment_is_caught(raw_dir, tmp_path):
    def change(records):
        records[0]['scenario']['family'] = 'civic-wizard'  # A test family.
    problems = _problems_after(raw_dir, tmp_path, change)
    assert any('family' in p for p in problems), problems


def test_missing_manifest_is_caught(raw_dir, tmp_path):
    raw = _copy(raw_dir, tmp_path)
    (raw / collector.MANIFEST_FILE).unlink()
    assert 'manifest.json is missing' in curate.validate(raw).problems


def test_altered_branch_starting_truth_is_caught(raw_dir, tmp_path):
    def change(records):
        branches = [r for r in records if r['scenario'].get('branch')]
        assert len(branches) >= 2
        branches[1]['truth']['before']['taken'].append(['x', 'y', 'z'])
    problems = _problems_after(raw_dir, tmp_path, change, split='train')
    assert any('starting backend truth differs' in p for p in problems), (
        problems)


def test_run_duplicated_across_splits_is_caught(raw_dir, tmp_path):
    raw = _copy(raw_dir, tmp_path)
    train = recorder.read_jsonl(raw / 'train' / recorder.TRANSITIONS_FILE)
    other = raw / 'validation' / recorder.TRANSITIONS_FILE
    other.parent.mkdir(exist_ok=True)
    with other.open('a', encoding='utf-8') as file:
        file.write(json.dumps(train[0]) + '\n')
    problems = curate.validate(raw).problems
    assert any('in train and validation' in p for p in problems), problems


def test_executor_mismatch_duplicate_steps_and_schema_are_caught(raw_dir,
                                                                tmp_path):
    def change(records):
        records[0]['result']['action'] = dict(records[0]['action'],
                                              text='tampered')
        records.append(dict(records[1]))
        records[2]['schema_version'] = 2
    problems = _problems_after(raw_dir, tmp_path, change)
    joined = ' | '.join(problems)
    assert 'different action' in joined
    assert 'duplicate steps' in joined
    assert 'schema 2' in joined


def test_unsettled_record_is_excluded_not_fatal(raw_dir, tmp_path):
    def change(records):
        records[0]['settled'] = False
    raw = _copy(raw_dir, tmp_path)
    _, path = _first_split(raw)
    _rewrite(path, change)
    report = curate.validate(raw)
    assert not report.problems
    assert list(report.exclusions.values()) == [
        'page did not settle before the after-observation']


def _element(node_id, role, name, parent=None, key=None, value=None):
    return observer.Element(
        node_id=node_id, role=role, name=name, value=value,
        states={s: False for s in observer.STATE_NAMES}, parent_id=parent,
        match_key=key, tag='div', box=observer.Box(0, 0, 10, 10),
        visible=True, in_viewport=True, hit_point=None, hit_testable=None,
        obstructed=False, clickable=False, pinned=False, paint_order=1)


def _observation(elements, epoch=1):
    motion = observer.Motion(1280, 800, 0, 0, 1280, 800, 0, 1)
    temporal = observer.Temporal(0, 0, 0, 0, 0, 0, True)
    return observer.Observation(1, epoch, 1, '', True, 'http://x/', 't',
                                tuple(elements), motion, temporal)


def test_unnamed_twins_align_by_node_within_a_document():
    before = _observation([_element(1, 'form', ''), _element(2, 'form', '')])
    after = _observation([_element(2, 'form', ''), _element(1, 'form', '')])
    alignment = curate.align(before, after)
    assert alignment.before_to_after == (1, 0)
    assert alignment.tiers == {'same_node': 2}


def test_unnamed_twins_across_documents_are_ambiguous_not_merged():
    before = _observation([_element(1, 'form', ''), _element(2, 'form', '')])
    after = _observation([_element(7, 'form', ''), _element(8, 'form', '')],
                         epoch=2)
    alignment = curate.align(before, after)
    assert alignment.before_to_after == (-1, -1)
    assert alignment.ambiguous_before == frozenset({0, 1})
    assert alignment.ambiguous_after == frozenset({0, 1})


def test_rerendered_keyed_element_and_new_and_removed_elements():
    before = _observation([_element(1, 'radio', '09:00', key='slot-09:00'),
                           _element(2, 'alert', 'Old error')])
    after = _observation([_element(9, 'radio', '09:00', key='slot-09:00'),
                          _element(10, 'alert', 'New error')])
    alignment = curate.align(before, after)
    assert alignment.before_to_after == (0, -1)
    assert alignment.after_to_before == (0, -1)
    assert alignment.tiers == {'page_key': 1}


@pytest.fixture(scope='module')
def curated_dir(raw_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp('curated')
    curate.curate(raw_dir, out)
    return out


def test_inputs_hold_no_privileged_information(curated_dir):
    for split in curate.SPLITS:
        for example in training_data.load_examples(curated_dir, split):
            text = json.dumps(example['inputs'])
            for forbidden in ('node_id', 'match_key', 'scenario', 'family',
                              'seed', 'policy', 'truth', 'run_id', 'goal',
                              'tag'):
                assert f'"{forbidden}"' not in text


def test_vocabulary_is_fitted_on_training_data_only(curated_dir):
    vocab = json.loads((curated_dir / 'vocab.json').read_text())
    manifest = json.loads((curated_dir / 'manifest.json').read_text())
    assert manifest['config']['vocab_fitted_on'] == 'train only'
    civic_only = {'proceed', 'practitioner', 'department'}
    assert not civic_only & set(vocab['words'])
    assert manifest['source_files'] and manifest['curated_files']


def test_windows_stay_inside_runs_and_never_invent_history(curated_dir):
    examples = training_data.load_examples(curated_dir, 'train')
    for window in training_data.windows(examples, 3):
        assert len({e['meta']['run_id'] for e in window}) == 1
        steps = [e['meta']['step'] for e in window]
        assert steps == list(range(steps[0], steps[0] + 3))
    gapped = [e for e in examples if e['meta']['step'] != 1]
    for window in training_data.windows(gapped, 2):
        assert 1 not in [e['meta']['step'] for e in window]
        assert window[0]['meta']['step'] != 0  # 0 cannot jump to 2.


def test_collate_pads_with_masks(curated_dir):
    data = training_data.Windows.load(curated_dir, 'train', 3)
    batch = training_data.collate([data[0], data[1]])
    b, k, n = batch['role'].shape
    assert (b, k) == (2, 3)
    assert batch['element_mask'].shape == (2, 3, n)
    assert batch['name_chars'].shape == (2, 3, n, curate.MAX_CHARS)
    padded = ~batch['element_mask']
    assert (batch['parent'][padded] == -1).all()
    assert not batch['t_known'][padded].any()
    assert batch['t_outcomes'].shape == (2, 3, len(curate.OUTCOMES))


def test_baselines_score_and_action_copy_beats_no_change(curated_dir):
    examples = training_data.load_examples(curated_dir, 'train')
    still = baselines.evaluate(examples, baselines.predict_no_change)
    copy = baselines.evaluate(examples, baselines.predict_action_copy)
    assert still['changed_element']['f1'] == 0.0
    assert copy['changed_element']['f1'] > 0.0
    assert copy['value_source_accuracy'] >= still['value_source_accuracy']
