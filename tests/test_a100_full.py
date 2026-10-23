"""Full-data preparation uses real Arrow; only GPU training is replaced."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path

import datasets as ds
import numpy as np
import pytest

from test_a100_diversity import make_sources


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def preparer():
    assert importlib.util.find_spec('tools.prepare_a100_full'), 'Full-cache preparer is missing'
    from tools import prepare_a100_full
    return prepare_a100_full


@pytest.fixture
def source_proof(tmp_path):
    original, cache = make_sources(tmp_path)
    full = ds.load_from_disk(str(cache))
    selection = json.loads((original/'selection/data_manifest.json').read_text())
    fingerprints = {name: full[name]._fingerprint for name in ('train', 'val')}
    provenance = dict(original_bundle_sha256=digest(original/'bundle_manifest.json'),
        original_selection_sha256=digest(original/'selection/data_manifest.json'),
        original_selection=selection, full_cache_path=str(cache.resolve()),
        full_cache={name: dict(rows=count, fingerprint=fingerprints[name])
                    for name, count in (('train', 12), ('val', 6))},
        verified_original_rows=dict(train=4, val=2),
        original_rows_sha256=dict(train='a'*64, val='b'*64), validation_rows_sha256='b'*64)
    stage2 = tmp_path/'stage2_data'
    put(stage2/'complete.json', dict(schema=1, status='complete',
        request=dict(base_rows=[4, 2], full_rows=[12, 6], repeat_factor=4, seed=42, global_batch=256),
        provenance=provenance))
    kwargs = dict(full_rows=(12, 6), base_rows=(4, 2), expected_fingerprints=fingerprints,
        expected_original_manifest_sha=digest(original/'bundle_manifest.json'),
        expected_tokenizer_sha=digest(original/'tokenizer.json'))
    return original, cache, stage2, kwargs


def prepare(source_proof, output):
    original, cache, stage2, kwargs = source_proof
    return preparer().prepare_full_bundle(cache, original, stage2, output, **kwargs)


def test_preparation_preserves_arrow_inodes_order_and_seals_full_identity(source_proof, tmp_path):
    """Copying or resampling Arrow rows, losing inventory, or including test is a bug."""
    original, cache, stage2, _ = source_proof
    output = tmp_path/'full_bundle'
    result = prepare(source_proof, output)
    manifest = json.loads((output/'bundle_manifest.json').read_text())
    selection = json.loads((output/'selection/data_manifest.json').read_text())
    assert manifest['derive_length'] is True
    assert json.loads((output/'data/dataset_dict.json').read_text()) == {'splits': ['train', 'val']}
    assert not (output/'data/test').exists()
    assert (output/'tokenizer.json').stat().st_ino == (original/'tokenizer.json').stat().st_ino
    for name, count in (('train', 12), ('val', 6)):
        assert np.load(output/f'selection/{name}_indices.npy').tolist() == list(range(count))
        for source in (cache/name).iterdir():
            if source.name in ('state.json', 'dataset_info.json') or source.suffix == '.arrow':
                assert (output/'data'/name/source.name).stat().st_ino == source.stat().st_ino
        assert selection['splits'][name]['selected_rows'] == count
        assert selection['splits'][name]['source_rows'] == count
    assert selection['identity']['seed'] == 42
    assert selection['identity']['train_rows'] == 12
    assert selection['identity']['validation_rows'] == 6
    assert selection['provenance']['historical_selected_row_proof']['verified_original_rows'] == {'train': 4, 'val': 2}
    assert selection['provenance']['source_files']
    for entry in manifest['files']:
        assert digest(output/entry['path']) == entry['sha256']
        assert (output/entry['path']).stat().st_size == entry['bytes']
    assert result['bundle_manifest_sha256'] == digest(output/'bundle_manifest.json')
    raw = ds.load_from_disk(str(output/'data'))
    assert 'length' not in raw['train'].column_names
    assert raw['train']['input_ids'] == ds.load_from_disk(str(cache))['train']['input_ids']


def test_prepared_bundle_is_loadable_with_derived_lengths(source_proof, tmp_path):
    from kaggle_experiments import load_bundle
    output = tmp_path/'full_bundle'
    result = prepare(source_proof, output)
    loaded, _, _ = load_bundle(output, expected_rows=(12, 6),
                               expected_manifest_sha=result['bundle_manifest_sha256'])
    assert loaded['train']['length'] == [3]*12
    assert loaded['val']['length'] == [3]*6


@pytest.mark.parametrize('change', ['cache_path', 'row_count', 'fingerprint', 'selected_proof', 'original_sha'])
def test_preparation_rejects_invalid_historical_source_proof_before_publishing(source_proof, tmp_path, change):
    path = source_proof[2]/'complete.json'
    value = json.loads(path.read_text())
    proof = value['provenance']
    if change == 'cache_path':
        proof['full_cache_path'] += '-other'
    elif change == 'row_count':
        proof['full_cache']['train']['rows'] = 11
    elif change == 'fingerprint':
        proof['full_cache']['val']['fingerprint'] = 'wrong'
    elif change == 'selected_proof':
        proof['verified_original_rows']['train'] = 3
    else:
        proof['original_bundle_sha256'] = 'f'*64
    put(path, value)
    output = tmp_path/'full_bundle'
    with pytest.raises(ValueError):
        prepare(source_proof, output)
    assert not output.exists()


def test_preparation_rejects_duplicate_or_partial_output(source_proof, tmp_path):
    output = tmp_path/'full_bundle'
    prepare(source_proof, output)
    with pytest.raises(FileExistsError):
        prepare(source_proof, output)
    other = tmp_path/'other_bundle'
    other.with_name('.other_bundle.partial').mkdir()
    with pytest.raises(FileExistsError):
        prepare(source_proof, other)


def test_preparation_rejects_indexed_dataset_instead_of_flattening(source_proof, tmp_path, monkeypatch):
    real_load = ds.load_from_disk
    cache = source_proof[1]
    def indexed(path, **kwargs):
        result = real_load(path, **kwargs)
        if Path(path) == cache:
            result['train'] = result['train'].select([1, 0, *range(2, 12)])
        return result
    monkeypatch.setattr(ds, 'load_from_disk', indexed)
    with pytest.raises(ValueError, match='indices|indexed'):
        prepare(source_proof, tmp_path/'full_bundle')


def runner():
    assert importlib.util.find_spec('tools.run_a100_full'), 'Full-data runner is missing'
    from tools import run_a100_full
    return run_a100_full


def launch_fixture(tmp_path, monkeypatch):
    module = runner()
    bundle = tmp_path/'bundle'
    bundle.mkdir()
    stage2 = tmp_path/'stage2'
    put(stage2/'stage2_complete.json', dict(schema=1, stage=2, status='complete', architecture='bart',
        ranking=['distinct4m', 'repeat4m'],
        models=dict(distinct4m={'pass@1': .687}, repeat4m={'pass@1': .676})))
    monkeypatch.setattr(module, 'verify_full_bundle', lambda path: dict(bundle_sha256='a'*64,
        selection_sha256='b'*64, tokenizer_sha256='c'*64))
    monkeypatch.setattr(module, 'GPU_LOCK_DIRECTORY', tmp_path/'gpu_lock')
    return module, bundle, stage2


def test_full_run_freezes_science_outside_fresh_trial_and_records_completion(tmp_path, monkeypatch):
    import kaggle_experiments
    module, bundle, stage2 = launch_fixture(tmp_path, monkeypatch)
    observed = []
    def train(path, config, *, resume_from_checkpoint=None):
        assert path == bundle
        assert not Path(config.output_dir).exists()
        observed.append((config, resume_from_checkpoint))
        result = dict(architecture='bart', global_step=5000, precision='fp16', world_size=1,
            effective_batch=256, metrics={'eval_token_loss': .25})
        put(Path(config.output_dir)/'complete.json', result)
        return result
    monkeypatch.setattr(kaggle_experiments, 'run_experiment', train)
    output = tmp_path/'full_run'
    result = module.run_full(bundle, output, stage2)
    config, resume = observed[0]
    assert (config.architecture, config.expected_rows, config.micro_batch, config.global_batch,
        config.learning_rate, config.num_train_epochs, config.eval_steps, config.early_stopping_patience,
        config.precision, config.gradient_checkpointing, config.torch_compile, config.seed,
        config.max_steps, resume) == ('bart', (11593060, 118302), 128, 256, .0002, 4., 5000, 5,
                                     'fp16', False, False, 42, -1, None)
    request = json.loads((output/'run_request.json').read_text())
    assert request['config']['expected_manifest_sha'] == 'a'*64
    assert request['initialization'] == 'from_scratch'
    assert result['status'] == 'complete'
    assert json.loads((output/'queue_state.json').read_text())['status'] == 'complete'
    with pytest.raises(FileExistsError):
        module.run_full(bundle, output, stage2)
    assert len(observed) == 1


def test_full_run_records_failure_and_never_retries(tmp_path, monkeypatch):
    import kaggle_experiments
    module, bundle, stage2 = launch_fixture(tmp_path, monkeypatch)
    calls = []
    def fail(*args, **kwargs):
        calls.append(True)
        raise FloatingPointError('Nonfinite loss')
    monkeypatch.setattr(kaggle_experiments, 'run_experiment', fail)
    output = tmp_path/'full_run'
    with pytest.raises(FloatingPointError):
        module.run_full(bundle, output, stage2)
    assert json.loads((output/'queue_state.json').read_text())['status'] == 'failed'
    with pytest.raises(FileExistsError):
        module.run_full(bundle, output, stage2)
    assert len(calls) == 1


def test_full_run_rejects_changed_stage2_winner_before_training(tmp_path, monkeypatch):
    module, bundle, stage2 = launch_fixture(tmp_path, monkeypatch)
    path = stage2/'stage2_complete.json'
    proof = json.loads(path.read_text())
    proof['ranking'] = ['repeat4m', 'distinct4m']
    put(path, proof)
    with pytest.raises(ValueError, match='ranking|winner'):
        module.run_full(bundle, tmp_path/'full_run', stage2)
