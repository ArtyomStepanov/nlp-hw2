"""Full-cache loading and resumable best-model selection on real tiny CPU runs."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import datasets as ds
import pytest
from safetensors.torch import load_file
import torch

import kaggle_experiments as k
from test_kaggle_experiments import bundle  # Reuse the real checksummed Arrow fixture.


def reseal(root, **metadata):
    entries = [dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                    sha256=k.file_sha256(path))
               for path in sorted(root.rglob('*'))
               if path.is_file() and path.name != 'bundle_manifest.json']
    k.write_json(root/'bundle_manifest.json', dict(schema=1, files=entries,
        selection_sha256=k.file_sha256(root/'selection/data_manifest.json'), **metadata))
    return k.file_sha256(root/'bundle_manifest.json')


def raw_bundle(root, *, long_source=False):
    data = ds.load_from_disk(str(root/'data'))
    raw = {name: split.remove_columns('length') for name, split in data.items()}
    rows = raw['train'].to_list()
    rows[0]['input_ids'] = rows[0]['input_ids'][:1] + rows[0]['input_ids'][-1:]
    rows[0]['attention_mask'] = [1, 1]
    rows[0]['labels'] = [rows[0]['labels'][0]]*9 + rows[0]['labels'][-1:]
    rows[1]['input_ids'] = rows[1]['input_ids'][:1] + [rows[1]['input_ids'][1]]*7 + rows[1]['input_ids'][-1:]
    rows[1]['attention_mask'] = [1]*9
    if long_source:
        rows[0]['input_ids'] = [1]*257
        rows[0]['attention_mask'] = [1]*257
    raw['train'] = ds.Dataset.from_list(rows)
    ds.DatasetDict(raw).save_to_disk(str(root/'raw-data'))
    shutil.rmtree(root/'data')
    (root/'raw-data').rename(root/'data')
    return reseal(root, derive_length=True)


def test_default_config_manifest_preserves_historical_resume_identity(tmp_path):
    assert hasattr(k, 'config_manifest'), 'Canonical resume-compatible config serializer is missing'
    config = k.KaggleConfig(architecture='bart', output_dir=tmp_path/'run')
    historical = asdict(config)
    historical.pop('output_dir')
    historical.pop('early_stopping_patience', None)
    historical['expected_rows'] = [1_000_000, 20_000]
    assert k.config_manifest(config) == historical
    config.early_stopping_patience = 5
    assert k.config_manifest(config)['early_stopping_patience'] == 5


@pytest.mark.parametrize('patience', [0, -1, True, 1.5])
def test_invalid_early_stopping_patience_is_rejected(patience, tmp_path):
    with pytest.raises(ValueError, match='patience'):
        k.KaggleConfig(architecture='bart', output_dir=tmp_path/'run',
                       early_stopping_patience=patience)


def test_pinned_raw_cache_derives_target_aware_lengths_without_rewriting(bundle, monkeypatch):
    sha = raw_bundle(bundle)
    original_load = ds.load_from_disk
    loaded = []
    def capture(*args, **kwargs):
        data = original_load(*args, **kwargs)
        loaded.append(data)
        return data
    monkeypatch.setattr(ds, 'load_from_disk', capture)
    before = {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
              for path in bundle.rglob('*') if path.is_file()}
    data, _, _ = k.load_bundle(bundle, expected_rows=(8, 4), expected_manifest_sha=sha)
    assert data['train']['length'][:] == [10, 9, 3, 3, 3, 3, 3, 3]
    for name in ('train', 'val'):
        source = loaded[0][name]
        assert data[name].cache_files == source.cache_files
        assert [buffer.address if buffer else None for buffer in data[name].data.column('input_ids').chunk(0).buffers()] == [
            buffer.address if buffer else None for buffer in source.data.column('input_ids').chunk(0).buffers()]
    assert before == {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                      for path in bundle.rglob('*') if path.is_file()}


def test_raw_cache_length_derivation_requires_pinned_manifest(bundle):
    raw_bundle(bundle)
    with pytest.raises(ValueError, match='pinned|expected_manifest_sha'):
        k.load_bundle(bundle, expected_rows=(8, 4))


def test_raw_cache_rejects_lengths_above_existing_256_contract(bundle):
    sha = raw_bundle(bundle, long_source=True)
    with pytest.raises(ValueError, match='256|length'):
        k.load_bundle(bundle, expected_rows=(8, 4), expected_manifest_sha=sha)


def test_raw_cache_rejects_index_views_before_add_column_can_flatten(bundle, monkeypatch):
    sha = raw_bundle(bundle)
    original_load = ds.load_from_disk
    def indexed(*args, **kwargs):
        data = original_load(*args, **kwargs)
        data['train'] = data['train'].select([1, 0, 2, 3, 4, 5, 6, 7])
        return data
    monkeypatch.setattr(ds, 'load_from_disk', indexed)
    with pytest.raises(ValueError, match='indices|indexed|index view'):
        k.load_bundle(bundle, expected_rows=(8, 4), expected_manifest_sha=sha)


def plateau_metric(monkeypatch):
    # Supply a deterministic validation plateau while retaining real forward,
    # backward, optimizer, token-statistics and checkpoint behavior.
    import training
    original = training._TokenLossMetric
    class Plateau(original):
        def __call__(self, prediction, compute_result=False):
            result = super().__call__(prediction, compute_result=compute_result)
            if compute_result:
                result['token_loss'] = 1.
            return result
    monkeypatch.setattr(training, '_TokenLossMetric', Plateau)
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')


def tiny_config(output, **changes):
    return k.KaggleConfig(architecture='bart', output_dir=output,
        micro_batch=1, global_batch=2, precision='no', use_cpu=True,
        tiny_model=True, expected_rows=(8, 4), max_steps=10, eval_steps=2,
        learning_rate=.002, gradient_checkpointing=False,
        early_stopping_patience=2, **changes)


def assert_best_and_latest_are_distinct(output, best_step, latest_step):
    best = load_file(str(output/f'checkpoint-{best_step}/model.safetensors'))
    latest = load_file(str(output/f'checkpoint-{latest_step}/model.safetensors'))
    final = load_file(str(output/'final/model.safetensors'))
    assert set(final) == set(best) == set(latest)
    assert all(torch.equal(final[name], best[name]) for name in best)
    assert any(not torch.equal(latest[name], best[name]) for name in best)
    for filename in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth',
                     'trainer_state.json', 'run_manifest.json'):
        assert (output/f'checkpoint-{latest_step}'/filename).stat().st_size > 0


def test_patience_restores_from_checkpoint_and_final_exports_best_weights(bundle, tmp_path, monkeypatch):
    plateau_metric(monkeypatch)
    output = tmp_path/'run'
    original_factory = k._safety_and_artifacts_callback
    class Interrupted(Exception):
        pass
    def interrupt_factory(*args, **kwargs):
        callback = original_factory(*args, **kwargs)
        original_save = callback.on_save
        def on_save(args, state, control, **kwargs):
            original_save(args, state, control, **kwargs)
            if state.global_step == 4:
                raise Interrupted()
        callback.on_save = on_save
        return callback
    monkeypatch.setattr(k, '_safety_and_artifacts_callback', interrupt_factory)
    config = tiny_config(output)
    with pytest.raises(Interrupted):
        k.run_experiment(bundle, config)
    state = json.loads((output/'checkpoint-4/trainer_state.json').read_text())
    assert state['stateful_callbacks']['EarlyStoppingCallback']['attributes']['early_stopping_patience_counter'] == 1
    assert not (output/'complete.json').exists()
    monkeypatch.setattr(k, '_safety_and_artifacts_callback', original_factory)
    result = k.run_experiment(bundle, config, resume_from_checkpoint=output/'checkpoint-4')
    assert result['global_step'] == 6  # Reset patience would incorrectly stop at 8.
    assert result['best_global_step'] == 2
    assert result['best_metric'] == 1.
    assert Path(result['best_model_checkpoint']).name == 'checkpoint-2'
    assert result['final_weights_source'] == 'best_eval_token_loss'
    assert_best_and_latest_are_distinct(output, 2, 6)


def test_off_period_final_step_keeps_latest_full_checkpoint_before_best_restore(bundle, tmp_path, monkeypatch):
    plateau_metric(monkeypatch)
    output = tmp_path/'off-period'
    config = tiny_config(output)
    config.max_steps = 5
    config.early_stopping_patience = 5
    result = k.run_experiment(bundle, config)
    assert result['global_step'] == 5
    assert result['best_global_step'] == 2
    assert_best_and_latest_are_distinct(output, 2, 5)
    # A completed portable checkpoint can also be exported to a fresh folder;
    # its latest resumable state remains in the original checkpoint directory.
    config.output_dir = tmp_path/'resumed-export'
    resumed = k.run_experiment(bundle, config, resume_from_checkpoint=output/'checkpoint-5')
    assert Path(resumed['latest_checkpoint']) == output/'checkpoint-5'
    assert Path(resumed['latest_checkpoint']).is_dir()
    restored = load_file(str(config.output_dir/'final/model.safetensors'))
    best = load_file(str(output/'checkpoint-2/model.safetensors'))
    assert all(torch.equal(restored[name], best[name]) for name in best)


def test_completion_marker_is_not_published_when_final_artifact_fails(bundle, tmp_path, monkeypatch):
    plateau_metric(monkeypatch)
    import plotting
    def fail_plot(*args, **kwargs):
        raise OSError('simulated final artifact failure')
    monkeypatch.setattr(plotting, 'plot_training_history', fail_plot)
    output = tmp_path/'partial'
    config = tiny_config(output)
    config.max_steps = 1
    with pytest.raises(OSError, match='artifact failure'):
        k.run_experiment(bundle, config)
    assert not (output/'complete.json').exists()
