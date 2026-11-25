"""C0.1: strict validation, alignment, curation, tensors and baselines."""

# Test functions need no docstrings, and pytest injects fixtures by name.
# pylint: disable=missing-function-docstring,redefined-outer-name

import copy
import json
import pathlib
import shutil

import pytest

import torch

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


def _decode_bytes(ids):
    return bytes(b - 1 for b in ids if b != curate.BYTE_PAD_ID).decode('utf-8')


def test_encode_bytes_ascii_unicode_empty_null_and_long():
    assert curate.encode_bytes('AB!') == [66, 67, 34]
    assert curate.encode_bytes(None) == []
    assert curate.encode_bytes('') == []

    null_text = 'a\x00b'
    null_ids = curate.encode_bytes(null_text)
    assert null_ids == [98, 1, 99]
    assert curate.BYTE_PAD_ID not in null_ids
    assert _decode_bytes(null_ids + [0, 0]) == null_text

    for text in ('नमस्ते दुनिया', '₹1,250.50 €99 £10', '🚀✨🧪', 'हिंदी ₹100 🌍'):
        ids = curate.encode_bytes(text)
        assert _decode_bytes(ids) == text

    over_32_chars = 'Appointment confirmed with Dr. Daniel Kim at 09:30'
    assert len(over_32_chars) > 32
    assert _decode_bytes(curate.encode_bytes(over_32_chars)) == over_32_chars

    over_128_bytes = 'नमस्ते-' * 20
    assert len(over_128_bytes.encode('utf-8')) > 128
    assert _decode_bytes(curate.encode_bytes(over_128_bytes)) == over_128_bytes

    with pytest.raises(UnicodeEncodeError):
        curate.encode_bytes('\ud800')


@pytest.fixture(scope='module')
def bytes_curated_dir(raw_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp('bytes_curated')
    curate.curate(raw_dir, out, input_encoding='utf8_bytes_v1')
    return out


def test_dataset_destination_protection(raw_dir, tmp_path):
    raw_copy = _copy(raw_dir, tmp_path / 'src')
    sentinel_text = 'do not overwrite'

    for mode in curate.INPUT_ENCODINGS:
        nonempty = tmp_path / f'nonempty_{mode}'
        nonempty.mkdir()
        sentinel = nonempty / 'sentinel.txt'
        sentinel.write_text(sentinel_text, encoding='utf-8')
        with pytest.raises(FileExistsError):
            curate.curate(raw_copy, nonempty, input_encoding=mode)
        assert sentinel.read_text(encoding='utf-8') == sentinel_text

    empty_dest = tmp_path / 'empty_dest'
    empty_dest.mkdir()
    curate.curate(raw_copy, empty_dest, input_encoding='utf8_bytes_v1')
    assert (empty_dest / 'manifest.json').exists()

    with pytest.raises(ValueError, match='overlap'):
        curate.curate(raw_copy, raw_copy / 'nested', input_encoding='legacy')
    with pytest.raises(ValueError, match='overlap'):
        curate.curate(raw_copy, raw_copy, input_encoding='utf8_bytes_v1')


def test_bytes_inputs_hold_no_privileged_information(bytes_curated_dir):
    for split in curate.SPLITS:
        for example in training_data.load_examples(bytes_curated_dir, split):
            text = json.dumps(example['inputs'])
            for forbidden in ('node_id', 'match_key', 'scenario', 'family',
                              'seed', 'policy', 'truth', 'run_id', 'goal',
                              'tag'):
                assert f'"{forbidden}"' not in text


@pytest.mark.parametrize(
    'bad_ids',
    [
        [-1],
        [0],
        [257],
        [1.9],
        [True],
        ['1'],
        'invalid',
        None,
    ],
)
def test_invalid_byte_ids_rejected(bytes_curated_dir, bad_ids):
    base = training_data.load_examples(bytes_curated_dir, 'train')[0]

    bad_name = copy.deepcopy(base)
    bad_name['inputs']['state']['elements'][0]['name_bytes'] = bad_ids
    with pytest.raises(ValueError):
        training_data.Windows([[bad_name]])

    bad_value = copy.deepcopy(base)
    bad_value['inputs']['state']['elements'][0]['value_bytes'] = bad_ids
    with pytest.raises(ValueError):
        training_data.Windows([[bad_value]])

    bad_action = copy.deepcopy(base)
    bad_action['inputs']['action']['text_bytes'] = bad_ids
    with pytest.raises(ValueError):
        training_data.Windows([[bad_action]])


def test_valid_boundary_byte_ids_accepted(bytes_curated_dir):
    base = copy.deepcopy(
        training_data.load_examples(bytes_curated_dir, 'train')[0]
    )
    base['inputs']['state']['elements'][0]['name_bytes'] = [1, 256]
    base['inputs']['state']['elements'][0]['value_bytes'] = [256, 1]
    base['inputs']['action']['text_bytes'] = [1, 256]
    ds = training_data.Windows([[base]])
    batch = training_data.collate([ds[0]])
    assert batch['name_bytes'][0, 0, 0, :2].tolist() == [1, 256]
    assert batch['value_bytes'][0, 0, 0, :2].tolist() == [256, 1]
    assert batch['action_text_bytes'][0, 0, :2].tolist() == [1, 256]


def test_pad_and_pad_2d_contracts():
    vec = torch.tensor([4, 5], dtype=torch.int64)
    assert torch.equal(training_data._pad(vec, 2, 0), vec)
    padded_vec = training_data._pad(vec, 4, -1)
    assert torch.equal(padded_vec, torch.tensor([4, 5, -1, -1],
                                                dtype=torch.int64))
    assert padded_vec.dtype == torch.int64
    assert padded_vec.device == vec.device
    with pytest.raises(ValueError, match='smaller'):
        training_data._pad(vec, 1, 0)

    value = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    actual = training_data._pad_2d(value, 3, 3, 0)
    expected = torch.tensor(
        [[1, 2, 0], [3, 4, 0], [0, 0, 0]],
        dtype=torch.long,
    )
    assert torch.equal(actual, expected)
    assert actual.dtype == torch.long
    assert actual.device == value.device

    zero_cols = torch.empty((3, 0), dtype=torch.long)
    with pytest.raises(ValueError):
        training_data._pad_2d(zero_cols, 1, 1, 0)
    expanded_zero_cols = training_data._pad_2d(zero_cols, 3, 2, 0)
    assert expanded_zero_cols.shape == (3, 2)
    assert (expanded_zero_cols == 0).all()

    with pytest.raises(ValueError, match='two-dimensional'):
        training_data._pad_2d(vec, 2, 2, 0)


def test_manifest_compatibility_legacy_and_bytes(raw_dir, tmp_path):
    legacy_out = tmp_path / 'legacy_out'
    bytes_out = tmp_path / 'bytes_out'
    legacy_m = curate.curate(raw_dir, legacy_out, input_encoding='legacy')
    bytes_m = curate.curate(raw_dir, bytes_out,
                            input_encoding='utf8_bytes_v1')

    for key in ('schema_version', 'source_files', 'vocab_sizes', 'examples',
                'exclusions', 'label_coverage', 'alignment'):
        assert legacy_m[key] == bytes_m[key]

    assert 'source_commit' in legacy_m
    assert 'input_encoding' not in legacy_m
    assert bytes_m['input_encoding'] == 'utf8_bytes_v1'
    assert bytes_m['byte_padding_id'] == curate.BYTE_PAD_ID
    assert bytes_m['byte_vocab_size'] == curate.BYTE_VOCAB_SIZE
    assert bytes_m['config']['byte_truncation'] is False
    assert 'vocab_sha256' in bytes_m


def test_bytes_collate_dynamic_padding_empty_and_mixed_rejection(
        curated_dir, bytes_curated_dir):
    byte_ds = training_data.Windows.load(bytes_curated_dir, 'train', 1)
    legacy_ds = training_data.Windows.load(curated_dir, 'train', 1)

    with pytest.raises(ValueError, match='mix'):
        training_data.collate([byte_ds[0], legacy_ds[0]])

    # Dynamic collation of distinct element counts and byte lengths across K=3.
    byte_k3 = training_data.Windows.load(bytes_curated_dir, 'train', 3)
    selected_indices = [0, len(byte_k3) - 1]
    batch_k3 = training_data.collate([byte_k3[i] for i in selected_indices])
    b3, k3, n3, l_name3 = batch_k3['name_bytes'].shape
    assert (b3, k3) == (2, 3)
    assert batch_k3['name_bytes'].dtype == torch.int64
    assert batch_k3['value_bytes'].dtype == torch.int64
    assert batch_k3['action_text_bytes'].dtype == torch.int64
    assert batch_k3['element_mask'].shape == (2, 3, n3)
    assert batch_k3['value_bytes'].shape[:3] == (2, 3, n3)
    assert batch_k3['action_text_bytes'].shape[:2] == (2, 3)
    assert l_name3 >= 1
    assert batch_k3['value_bytes'].shape[3] >= 1
    assert batch_k3['action_text_bytes'].shape[2] >= 1

    # Verify every example and time step in the K=3 batch.
    examples = training_data.load_examples(bytes_curated_dir, 'train')
    by_key = {(e['meta']['run_id'], e['meta']['step']): e for e in examples}
    for b_idx, ds_idx in enumerate(selected_indices):
        for k_idx in range(k3):
            meta = byte_k3.meta[ds_idx][k_idx]
            example = by_key[(meta['run_id'], meta['step'])]
            elems = example['inputs']['state']['elements']
            elem_count = len(elems)
            assert batch_k3['element_mask'][b_idx, k_idx, :elem_count].all()
            assert not batch_k3['element_mask'][b_idx, k_idx, elem_count:].any()
            if elem_count < n3:
                assert (batch_k3['name_bytes'][b_idx, k_idx, elem_count:]
                        == curate.BYTE_PAD_ID).all()
                assert (batch_k3['value_bytes'][b_idx, k_idx, elem_count:]
                        == curate.BYTE_PAD_ID).all()
            for e_idx, elem in enumerate(elems):
                nb = batch_k3['name_bytes'][b_idx, k_idx, e_idx].tolist()
                n_len = len(elem['name_bytes'])
                assert nb[:n_len] == elem['name_bytes']
                assert all(x == curate.BYTE_PAD_ID for x in nb[n_len:])
                assert _decode_bytes(nb) == _decode_bytes(elem['name_bytes'])

                vb = batch_k3['value_bytes'][b_idx, k_idx, e_idx].tolist()
                v_len = len(elem['value_bytes'])
                assert vb[:v_len] == elem['value_bytes']
                assert all(x == curate.BYTE_PAD_ID for x in vb[v_len:])
                assert _decode_bytes(vb) == _decode_bytes(elem['value_bytes'])

            ab = batch_k3['action_text_bytes'][b_idx, k_idx].tolist()
            a_len = len(example['inputs']['action']['text_bytes'])
            assert ab[:a_len] == example['inputs']['action']['text_bytes']
            assert all(x == curate.BYTE_PAD_ID for x in ab[a_len:])
            assert _decode_bytes(ab) == _decode_bytes(
                example['inputs']['action']['text_bytes'])

    # All-empty byte sequences must still collate to minimum width 1.
    empty_example = {
        'meta': {'run_id': 'r1', 'step': 0, 'split': 'train',
                 'family': 'f1', 'action_kind': 'scroll'},
        'inputs': {
            'state': {
                'elements': [{
                    'role': 2, 'name_words': [0] * 8,
                    'name_chars': [0] * 32, 'value_chars': [0] * 32,
                    'has_value': False, 'states': [0] * 8, 'flags': [0] * 4,
                    'box': [0.0] * 4, 'box_mask': False, 'parent': -1,
                    'name_bytes': [], 'value_bytes': [],
                }],
                'page': [0.0] * 4,
            },
            'action': {
                'kind': 4, 'target': -1, 'text_chars': [0] * 32,
                'checked': 0, 'scroll': 0.0, 'pointer': [0.0, 0.0],
                'text_bytes': [],
            },
        },
        'targets': {
            'elements': [{
                'known': True, 'removed': False, 'changed': False,
                'matched': True, 'next_states': [0] * 8, 'value_source': 0,
                'value_mask': False, 'box_delta': [0.0] * 4, 'box_mask': False,
            }],
            'new': [],
            'outcomes': [0] * 8,
        },
    }
    empty_ds = training_data.Windows([[empty_example]])
    empty_batch = training_data.collate([empty_ds[0]])
    assert empty_batch['name_bytes'].shape == (1, 1, 1, 1)
    assert empty_batch['value_bytes'].shape == (1, 1, 1, 1)
    assert empty_batch['action_text_bytes'].shape == (1, 1, 1)
    assert (empty_batch['name_bytes'] == curate.BYTE_PAD_ID).all()
    assert (empty_batch['value_bytes'] == curate.BYTE_PAD_ID).all()
    assert (empty_batch['action_text_bytes'] == curate.BYTE_PAD_ID).all()

    # Partially byte-encoded records must be rejected explicitly.
    partial_missing_elem = json.loads(json.dumps(empty_example))
    del partial_missing_elem['inputs']['state']['elements'][0]['value_bytes']
    with pytest.raises(ValueError, match='missing element byte fields'):
        training_data.Windows([[partial_missing_elem]])

    partial_missing_action = json.loads(json.dumps(empty_example))
    del partial_missing_action['inputs']['action']['text_bytes']
    with pytest.raises(ValueError, match='missing action text_bytes'):
        training_data.Windows([[partial_missing_action]])
