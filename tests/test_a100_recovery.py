"""Recovery contract tests; all checkpoints are tiny and no GPU work is launched."""
from argparse import Namespace
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest

from test_a100_stage2 import GOLDEN, put, stage1_fixture, training_fixture


def api():
    assert importlib.util.find_spec('tools.recover_a100_stage2'), 'Recovery runner is missing'
    from tools import recover_a100_stage2
    return recover_a100_stage2


def failed_fixture(tmp_path):
    failed = tmp_path/'failed'
    trial = failed/'trials/repeat4m'
    config, identity = training_fixture(trial)
    (trial/'complete.json').unlink()
    checkpoint = trial/'checkpoint-14000'
    checkpoint.mkdir()
    put(checkpoint/'run_manifest.json', json.loads((trial/'run_manifest.json').read_text()))
    put(checkpoint/'trainer_state.json', dict(global_step=14000, log_history=[dict(step=14000, loss=.3)]))
    put(checkpoint/'config.json', json.loads((trial/'final/config.json').read_text()))
    for name in ('tokenizer.json', 'model.safetensors', 'optimizer.pt', 'scheduler.pt', 'scaler.pt', 'rng_state.pth'):
        (checkpoint/name).write_bytes(b'tiny saved state')
    put(failed/'queue_state.json', dict(status='failed', stage=2, variant='repeat4m',
        phase='training', architecture='bart', error='RuntimeError: repeat4m training exited 1', completed=[]))
    from tools import run_a100_stage2
    put(failed/'run_request.json', dict(sources=run_a100_stage2._sources()))
    proof = dict(completion_sha256='f'*64, bundles={
        'repeat4m': identity,
        'distinct4m': dict(identity, bundle_sha256='d'*64, selection_sha256='e'*64)})
    return failed, checkpoint, proof


def test_recovery_preserves_every_frozen_setting_and_full_schedule(tmp_path):
    runner = api()
    failed, checkpoint, proof = failed_fixture(tmp_path)
    recovery = runner.verify_failed_stage2(failed, proof, 'bart')
    manifest = json.loads((checkpoint/'run_manifest.json').read_text())
    config = runner.make_recovery_config(manifest, tmp_path/'new')
    settings = asdict(config)
    settings.pop('output_dir')
    settings['expected_rows'] = list(config.expected_rows)
    assert settings == manifest['config']
    assert (config.precision, config.micro_batch, config.global_batch, config.max_steps,
            config.learning_rate, config.num_train_epochs) == ('fp16', 128, 256, -1, .0002, 1.)
    assert recovery['checkpoint_step'] == 14000
    assert recovery['remaining_step_attempts'] == 1625
    distinct = runner.make_recovery_config(manifest, tmp_path/'distinct', manifest_sha='d'*64)
    assert distinct.expected_manifest_sha == 'd'*64
    assert {k:v for k,v in asdict(distinct).items() if k not in ('output_dir', 'expected_manifest_sha')} == {
        k:v for k,v in asdict(config).items() if k not in ('output_dir', 'expected_manifest_sha')}


@pytest.mark.parametrize('change', ['checkpoint_manifest', 'step', 'optimizer', 'scaler', 'rng',
                                  'batch', 'bundle', 'queue_status', 'training_source'])
def test_checkpoint_or_scientific_mismatch_is_rejected_before_launch(tmp_path, change):
    runner = api()
    failed, checkpoint, proof = failed_fixture(tmp_path)
    if change == 'checkpoint_manifest':
        path = checkpoint/'run_manifest.json'
        value = json.loads(path.read_text())
        value['precision'] = 'bf16'
        put(path, value)
    elif change == 'step':
        put(checkpoint/'trainer_state.json', dict(global_step=13999))
    elif change in ('optimizer', 'scaler', 'rng'):
        (checkpoint/{'optimizer':'optimizer.pt', 'scaler':'scaler.pt', 'rng':'rng_state.pth'}[change]).unlink()
    elif change == 'batch':
        path = failed/'trials/repeat4m/run_manifest.json'
        value = json.loads(path.read_text())
        value['config']['micro_batch'] = 64
        put(path, value)
        put(checkpoint/'run_manifest.json', value)
    elif change == 'bundle':
        proof['bundles']['repeat4m']['bundle_sha256'] = '0'*64
    elif change == 'queue_status':
        path = failed/'queue_state.json'
        value = json.loads(path.read_text())
        value['status'] = 'running'
        put(path, value)
    else:
        path = failed/'run_request.json'
        value = json.loads(path.read_text())
        value['sources']['training.py'] = '0'*64
        put(path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.verify_failed_stage2(failed, proof, 'bart')


def queue_fixture(tmp_path, monkeypatch):
    runner = api()
    monkeypatch.setattr(runner, 'GPU_LOCK_DIRECTORY', tmp_path/'gpu_lock')
    failed, checkpoint, proof = failed_fixture(tmp_path)
    stage1 = stage1_fixture(tmp_path/'stage1')
    prepared = tmp_path/'prepared'
    prepared.mkdir()
    monkeypatch.setattr(runner, 'verify_prepared', lambda path: proof)
    return runner, failed, checkpoint, stage1, prepared, proof


def test_recovery_sequence_resumes_only_repeat_then_evaluates_both(monkeypatch, tmp_path):
    from kaggle_experiments import file_sha256
    from tools import evaluate_fixed_bundle
    import kaggle_experiments
    runner, failed, checkpoint, stage1, prepared, proof = queue_fixture(tmp_path, monkeypatch)
    launches = []
    def train(bundle, config, *, resume_from_checkpoint=None):
        variant = Path(bundle).name
        launches.append((variant, 'training', resume_from_checkpoint))
        trial = Path(config.output_dir)
        _, identity = training_fixture(trial)
        manifest = json.loads((trial/'run_manifest.json').read_text())
        settings = asdict(config)
        settings.pop('output_dir')
        settings['expected_rows'] = list(config.expected_rows)
        manifest.update(config=settings, bundle_sha256=proof['bundles'][variant]['bundle_sha256'],
                        selection_sha256=proof['bundles'][variant]['selection_sha256'])
        put(trial/'run_manifest.json', manifest)
        complete = json.loads((trial/'complete.json').read_text())
        complete['amp_diagnostics'] = dict(skipped_optimizer_steps=1)
        put(trial/'complete.json', complete)
        return complete
    def evaluate(bundle, saved, output, **kwargs):
        variant = Path(bundle).name
        launches.append((variant, 'evaluation', None))
        source = stage1_fixture(Path(output).parent/'fixture_source')/'evaluations/bart'
        import shutil
        shutil.copytree(source, output)
        shutil.rmtree(source.parents[1])
        root = Path(output)
        reference = json.loads(GOLDEN.read_text())
        metrics = json.loads((root/'metrics.json').read_text())
        metrics['pass@1'] = .26 if variant == 'repeat4m' else .27
        put(root/'metrics.json', metrics)
        identity = proof['bundles'][variant]
        put(root/'config.json', dict(architecture='bart', checkpoint=str(saved), checkpoint_step=15625,
            validation_rows=20000, validation_sha256=reference['validation_sha256'],
            bundle_sha256=identity['bundle_sha256'], selection_sha256=identity['selection_sha256'],
            do_sample=False, use_cache=True, official_test_used=False, **runner.EVALUATION))
        put(root/'progress.json', dict(phase='complete'))
        (root/'predictions.jsonl').write_text(''.join(json.dumps(dict(sample_row=i))+'\n' for i in range(20000)))
        return metrics
    def invoke(command, **kwargs):
        values = dict(zip(command[3::2], command[4::2]))
        return runner.worker(Namespace(request=values['--request'], request_sha=values['--request-sha'],
            variant=values['--variant'], phase=values['--phase'],
            resume_from=values.get('--resume-from')))
    monkeypatch.setattr(kaggle_experiments, 'run_experiment', train)
    monkeypatch.setattr(evaluate_fixed_bundle, 'run', evaluate)
    monkeypatch.setattr(runner, '_invoke', invoke)
    old_sha = file_sha256(checkpoint/'run_manifest.json')
    out = tmp_path/'recovery'
    summary = runner.run_recovery(stage1, failed, prepared, out, reference_identity=GOLDEN)
    assert launches == [('repeat4m','training',str(checkpoint.resolve())), ('repeat4m','evaluation',None),
                        ('distinct4m','training',None), ('distinct4m','evaluation',None)]
    assert summary['ranking'] == ['distinct4m', 'repeat4m']
    assert summary['architecture'] == 'bart'
    assert summary['recovery']['checkpoint_step'] == 14000
    assert summary['amp_diagnostics']['repeat4m']['skipped_optimizer_steps'] == 1
    assert summary['planned_step_attempts'] == 15625
    assert 'attempt' in summary['optimizer_updates_per_epoch_interpretation']
    assert file_sha256(checkpoint/'run_manifest.json') == old_sha
    assert json.loads((out/'queue_state.json').read_text())['status'] == 'complete'
    with pytest.raises(ValueError, match='existing|fresh|already'):
        runner.run_recovery(stage1, failed, prepared, out, reference_identity=GOLDEN)
    assert len(launches) == 4


def test_partial_failure_is_retained_and_never_retried(monkeypatch, tmp_path):
    runner, failed, checkpoint, stage1, prepared, proof = queue_fixture(tmp_path, monkeypatch)
    launches = []
    def fail(command, **kwargs):
        launches.append(command)
        partial = tmp_path/'recovery/trials/repeat4m'
        partial.mkdir(parents=True)
        (partial/'partial').write_bytes(b'inspect this')
        raise RuntimeError('synthetic training failure')
    monkeypatch.setattr(runner, '_invoke', fail)
    out = tmp_path/'recovery'
    with pytest.raises(RuntimeError, match='synthetic'):
        runner.run_recovery(stage1, failed, prepared, out, reference_identity=GOLDEN)
    original_state = (out/'queue_state.json').read_bytes()
    assert json.loads(original_state)['status'] == 'failed'
    assert not (out/'stage2_complete.json').exists()
    with pytest.raises(ValueError, match='existing|fresh|already'):
        runner.run_recovery(stage1, failed, prepared, out, reference_identity=GOLDEN)
    assert len(launches) == 1
    assert (out/'queue_state.json').read_bytes() == original_state
    assert (out/'trials/repeat4m/partial').read_bytes() == b'inspect this'


def test_worker_refuses_checkpoint_changed_after_request(monkeypatch, tmp_path):
    runner, failed, checkpoint, stage1, prepared, proof = queue_fixture(tmp_path, monkeypatch)
    def invoke(command, **kwargs):
        (checkpoint/'optimizer.pt').write_bytes(b'changed after request')
        values = dict(zip(command[3::2], command[4::2]))
        runner.worker(Namespace(request=values['--request'], request_sha=values['--request-sha'],
            variant=values['--variant'], phase=values['--phase'], resume_from=values.get('--resume-from')))
    monkeypatch.setattr(runner, '_invoke', invoke)
    out = tmp_path/'recovery'
    with pytest.raises(ValueError, match='checkpoint inventory'):
        runner.run_recovery(stage1, failed, prepared, out, reference_identity=GOLDEN)
    assert not (out/'trials').exists()
    assert json.loads((out/'queue_state.json').read_text())['status'] == 'failed'


def test_changed_source_snapshot_stops_before_worker(monkeypatch, tmp_path):
    runner, failed, checkpoint, stage1, prepared, proof = queue_fixture(tmp_path, monkeypatch)
    sources = runner._sources
    changed = False
    def source_snapshot():
        result = sources()
        if changed:
            result['tools/recover_a100_stage2.py'] = '0'*64
        return result
    write_json = runner.write_json
    def write_and_change(path, value):
        nonlocal changed
        write_json(path, value)
        if Path(path).name == 'run_request.json':
            changed = True
    monkeypatch.setattr(runner, '_sources', source_snapshot)
    monkeypatch.setattr(runner, 'write_json', write_and_change)
    monkeypatch.setattr(runner, '_invoke', lambda *args, **kwargs: pytest.fail('changed source launched'))
    out = tmp_path/'recovery'
    with pytest.raises(ValueError, match='source snapshot'):
        runner.run_recovery(stage1, failed, prepared, out, reference_identity=GOLDEN)
    assert not (out/'trials').exists()


def test_shared_gpu_lock_excludes_another_fresh_output(monkeypatch, tmp_path):
    runner, failed, checkpoint, stage1, prepared, proof = queue_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, '_invoke', lambda *args, **kwargs: pytest.fail('duplicate GPU work launched'))
    with runner.queue_lock(runner.GPU_LOCK_DIRECTORY):
        with pytest.raises(RuntimeError, match='already running'):
            runner.run_recovery(stage1, failed, prepared, tmp_path/'different-output', reference_identity=GOLDEN)
    assert not (tmp_path/'different-output/run_request.json').exists()
