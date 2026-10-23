"""A real tiny full-trained checkpoint may use the original fixed validation only with proof."""
import hashlib
import json
from pathlib import Path
import shutil

import datasets as ds
import numpy as np
import pytest
from tokenizers import Tokenizer, models, processors

from evaluation import EvaluationConfig, evaluate_code
from kaggle_experiments import KaggleConfig, file_sha256, run_experiment, write_json
from tools.evaluate_fixed_bundle import run


def seal(root, *, derive_length=False):
    files = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size,
                  sha256=file_sha256(p)) for p in sorted(root.rglob('*'))
             if p.is_file() and p.name != 'bundle_manifest.json']
    write_json(root/'bundle_manifest.json', dict(schema=1, files=files,
        selection_sha256=file_sha256(root/'selection/data_manifest.json'),
        **({'derive_length': True} if derive_length else {})))
    return file_sha256(root/'bundle_manifest.json')


def row_sha(rows):
    return hashlib.sha256(''.join(json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n'
                                  for row in rows).encode()).hexdigest()


@pytest.fixture
def full_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    original, full = tmp_path/'original', tmp_path/'full'
    original.mkdir()
    full.mkdir()
    vocab = {'<unk>': 0, '<bos>': 1, '<eos>': 2, '<pad>': 3,
             **{f'question{i}': i+4 for i in range(32)},
             **{f'def solve():\n    return {i}': i+36 for i in range(32)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.add_special_tokens(['<unk>', '<bos>', '<eos>', '<pad>'])
    tokenizer.post_processor = processors.TemplateProcessing(single='$A', pair='$A $B')
    for root in (original, full):
        tokenizer.save(str(root/'tokenizer.json'))
    def rows(offset, count):
        return ds.Dataset.from_list([dict(input_ids=[1, i+4, 2], attention_mask=[1, 1, 1],
            labels=[i+36, 2]) for i in range(offset, offset+count)])
    data = ds.DatasetDict(train=rows(0, 12), val=rows(20, 6))
    data.save_to_disk(str(full/'data'))
    fixed_indices = dict(train=[7, 1, 9, 3], val=[4, 1])
    fixed = ds.DatasetDict({name: split.select(fixed_indices[name]).map(lambda row: {'length': 3})
                            for name, split in data.items()})
    fixed.save_to_disk(str(original/'data'))
    for root, counts, indices, strategy in (
            (original, (4, 2), fixed_indices, 'fixed_selection'),
            (full, (12, 6), dict(train=list(range(12)), val=list(range(6))), 'full_original_order')):
        details = {}
        for name, values in indices.items():
            (root/'selection').mkdir(exist_ok=True)
            path = root/f'selection/{name}_indices.npy'
            np.save(path, values)
            details[name] = dict(source_rows=len(data[name]), selected_rows=len(values),
                                 indices_sha256=file_sha256(path))
        value = dict(identity=dict(train_rows=counts[0], validation_rows=counts[1], seed=42,
            tokenizer_sha256=file_sha256(root/'tokenizer.json')), splits=details, strategy=strategy)
        if root == full:
            original_selection = json.loads((original/'selection/data_manifest.json').read_text())
            value['provenance'] = dict(original_bundle_sha256=file_sha256(original/'bundle_manifest.json'),
                historical_selected_row_proof=dict(original_bundle_sha256=file_sha256(original/'bundle_manifest.json'),
                    original_selection_sha256=file_sha256(original/'selection/data_manifest.json'),
                    original_selection=original_selection))
        write_json(root/'selection/data_manifest.json', value)
        seal(root, derive_length=root == full)
    trial = tmp_path/'trial'
    run_experiment(full, KaggleConfig(architecture='bart', output_dir=trial,
        micro_batch=1, global_batch=2, precision='no', use_cpu=True, tiny_model=True,
        expected_rows=(12, 6), expected_manifest_sha=file_sha256(full/'bundle_manifest.json'),
        max_steps=2, eval_steps=1, early_stopping_patience=5, gradient_checkpointing=False))
    golden_input = ds.Dataset.from_list([dict(sample_row=i,
        question=tokenizer.decode(row['input_ids'], skip_special_tokens=True),
        code=tokenizer.decode(row['labels'], skip_special_tokens=True),
        generated_code=tokenizer.decode(row['labels'], skip_special_tokens=True))
        for i, row in enumerate(fixed['val'])])
    golden_result = evaluate_code(golden_input, EvaluationConfig(workers=1, output_dir=tmp_path/'golden-results'))
    fields = ('sample_row', 'evaluable', 'reference_status', 'reference_result_json')
    golden = tmp_path/'reference_identity.json'
    write_json(golden, dict(validation_sha256=row_sha(fixed['val']),
        rows=[{key: row[key] for key in fields} for row in golden_result.predictions]))
    return dict(original=original, full=full, trial=trial, golden=golden)


def arguments(fixture):
    return dict(architecture='bart', device='cpu', batch_size=2, max_new_tokens=3, num_beams=1,
        expected_manifest_sha=file_sha256(fixture['original']/'bundle_manifest.json'),
        expected_rows=(4, 2), training_bundle_root=fixture['full'],
        expected_training_manifest_sha=file_sha256(fixture['full']/'bundle_manifest.json'),
        expected_training_rows=(12, 6), reference_identity=fixture['golden'])


def test_full_checkpoint_evaluates_exact_fixed_rows_with_both_identities(full_checkpoint, tmp_path):
    f = full_checkpoint
    output = tmp_path/'evaluation'
    metrics = run(f['original'], f['trial']/'final', output, **arguments(f))
    config = json.loads((output/'config.json').read_text())
    assert config['bundle_sha256'] == file_sha256(f['original']/'bundle_manifest.json')
    assert config['training_bundle_sha256'] == file_sha256(f['full']/'bundle_manifest.json')
    assert config['validation_mapping']['verified_rows'] == 2
    assert config['validation_mapping']['full_validation_rows'] == 6
    assert config['validation_mapping']['validation_sha256'] == metrics['validation_sha256']
    assert config['checkpoint_step'] == json.loads((f['trial']/'complete.json').read_text())['best_global_step']
    assert config['training_global_step'] == 2
    assert config['reference_identity_sha256'] == file_sha256(f['golden'])
    assert metrics['num_evaluated'] == 2
    assert metrics['reference_identity_verified'] is True
    assert json.loads((output/'progress.json').read_text())['phase'] == 'complete'


def test_default_path_still_rejects_a_checkpoint_from_another_bundle(full_checkpoint, tmp_path):
    f = full_checkpoint
    with pytest.raises(AssertionError):
        run(f['original'], f['trial']/'final', tmp_path/'default', architecture='bart', device='cpu',
            expected_manifest_sha=file_sha256(f['original']/'bundle_manifest.json'), expected_rows=(4, 2))


@pytest.mark.parametrize('missing', ['training_bundle_root', 'expected_training_manifest_sha', 'reference_identity'])
def test_cross_bundle_proof_requires_all_explicit_inputs(full_checkpoint, tmp_path, missing):
    f = full_checkpoint
    kwargs = arguments(f)
    kwargs.pop(missing)
    with pytest.raises(ValueError, match='training|reference|explicit'):
        run(f['original'], f['trial']/'final', tmp_path/'incomplete', **kwargs)


def reseal_training(f):
    sha = seal(f['full'], derive_length=True)
    manifest = json.loads((f['trial']/'run_manifest.json').read_text())
    manifest['bundle_sha256'] = sha
    manifest['selection_sha256'] = file_sha256(f['full']/'selection/data_manifest.json')
    manifest['config']['expected_manifest_sha'] = sha
    write_json(f['trial']/'run_manifest.json', manifest)
    completed = json.loads((f['trial']/'complete.json').read_text())
    write_json(Path(completed['best_model_checkpoint'])/'run_manifest.json', manifest)


@pytest.mark.parametrize('column', ['input_ids', 'attention_mask', 'labels'])
def test_every_mapped_row_is_checked_before_limit(full_checkpoint, tmp_path, column):
    f = full_checkpoint
    data = ds.load_from_disk(str(f['full']/'data'))
    validation = data['val'].to_list()
    # The second original validation row maps to source row1; limit=1 must not
    # permit this mismatch outside the generation prefix to go undetected.
    validation[1][column][1 if column == 'input_ids' else 0] += 1
    changed = ds.DatasetDict(train=data['train'], val=ds.Dataset.from_list(validation))
    changed.save_to_disk(str(f['full']/'changed-data'))
    shutil.rmtree(f['full']/'data')
    (f['full']/'changed-data').rename(f['full']/'data')
    reseal_training(f)
    with pytest.raises(ValueError, match='row|validation|mask'):
        run(f['original'], f['trial']/'final', tmp_path/'mismatch', limit=1, **arguments(f))


@pytest.mark.parametrize('change', ['source_tokenizer', 'final_weights', 'best_step', 'full_order'])
def test_incompatible_checkpoint_or_source_is_rejected(full_checkpoint, tmp_path, change):
    f = full_checkpoint
    if change in ('source_tokenizer', 'final_weights'):
        path = f['trial']/('source_tokenizer.json' if change == 'source_tokenizer' else 'final/model.safetensors')
        with path.open('ab') as stream:
            stream.write(b' ')
    elif change == 'best_step':
        value = json.loads((f['trial']/'complete.json').read_text())
        value['best_global_step'] = 999
        write_json(f['trial']/'complete.json', value)
    else:
        path = f['full']/'selection/val_indices.npy'
        np.save(path, [1, 0, 2, 3, 4, 5])
        value = json.loads((f['full']/'selection/data_manifest.json').read_text())
        value['splits']['val']['indices_sha256'] = file_sha256(path)
        write_json(f['full']/'selection/data_manifest.json', value)
        reseal_training(f)
    with pytest.raises(ValueError):
        run(f['original'], f['trial']/'final', tmp_path/'invalid', **arguments(f))


def test_golden_reference_drift_blocks_final_metrics_and_completion(full_checkpoint, tmp_path):
    f = full_checkpoint
    golden = json.loads(f['golden'].read_text())
    golden['rows'][1]['reference_result_json'] = '999'
    write_json(f['golden'], golden)
    output = tmp_path/'golden-drift'
    with pytest.raises(ValueError, match='reference mismatch'):
        run(f['original'], f['trial']/'final', output, **arguments(f))
    assert not (output/'metrics.json').exists()
    assert json.loads((output/'progress.json').read_text())['phase'] != 'complete'
