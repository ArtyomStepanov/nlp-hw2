"""Explicit one-shot recovery of the failed Stage2 FP16 repeat4m experiment.

Use a fresh output directory. The failed run and checkpoint are immutable inputs;
this driver never retries, changes scientific settings, or resumes its own output.
Execution must be explicitly approved before launching this command on the A100.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kaggle_experiments import KaggleConfig, file_sha256, write_json
from tools import run_a100_stage2 as stage2
from tools.run_a100_stage2 import (EVALUATION, EXPECTED_ROWS, UPDATES, VARIANTS,
    VALIDATION_SHA, ELIGIBLE, _read, _now, _finite, _require, _inventory,
    ensure_phase, queue_lock, make_training_config, verify_stage1,
    verify_prepared, verify_training, verify_evaluation)

RESUME_STEP = 14000
# Shared across fresh output directories and code snapshots on this host.
GPU_LOCK_DIRECTORY = Path('/tmp')/f'nlp_hw2_stage2_recovery_cuda0_uid{os.getuid()}'


def _settings(config):
    result = asdict(config)
    result.pop('output_dir')
    result['expected_rows'] = list(config.expected_rows)
    return result


def make_recovery_config(manifest, output_dir, *, manifest_sha=None):
    """Reconstruct every frozen setting; permit only the paired bundle's SHA."""
    values = dict(manifest['config'])
    values.pop('output_dir', None)
    values['expected_rows'] = tuple(values['expected_rows'])
    config = KaggleConfig(output_dir=output_dir, **values)
    expected = make_training_config(config.architecture, output_dir, config.expected_manifest_sha)
    _require(_settings(config), _settings(expected), 'Frozen scientific configuration')
    if manifest_sha is not None:
        config.expected_manifest_sha = manifest_sha
    return config


def _expected_manifest(original, variant, proof):
    expected = json.loads(json.dumps(original))
    expected['config'].pop('output_dir', None)
    identity = proof['bundles'][variant]
    expected['config']['expected_manifest_sha'] = identity['bundle_sha256']
    expected['bundle_sha256'] = identity['bundle_sha256']
    expected['selection_sha256'] = identity['selection_sha256']
    return expected


def verify_failed_stage2(failed_dir, prepared_proof, architecture):
    """Require the failed repeat phase and a complete, unchanged FP16 checkpoint."""
    failed = Path(failed_dir).resolve()
    state = _read(failed/'queue_state.json')
    for name, expected in dict(status='failed', stage=2, phase='training', variant='repeat4m',
                                architecture=architecture, completed=[]).items():
        _require(state.get(name), expected, f'Failed Stage2 {name}')
    if not state.get('error'):
        raise ValueError('Failed Stage2 must retain its queue error')
    old_sources = _read(failed/'run_request.json')['sources']
    new_sources = _sources()
    new_sources.pop('tools/recover_a100_stage2.py')
    _require(set(old_sources), set(new_sources), 'Failed run source inventory')
    _require({name:sha for name,sha in old_sources.items() if name != 'kaggle_experiments.py'},
             {name:sha for name,sha in new_sources.items() if name != 'kaggle_experiments.py'},
             'Unchanged training and model sources')
    trial = failed/'trials/repeat4m'
    if (failed/'stage2_complete.json').exists() or (trial/'complete.json').exists():
        raise ValueError('Recovery requires an incomplete failed Stage2 run')
    checkpoint = trial/f'checkpoint-{RESUME_STEP}'
    manifest = _read(trial/'run_manifest.json')
    _finite(manifest, 'Failed Stage2 manifest')
    config = make_recovery_config(manifest, trial)
    _require(config.architecture, architecture, 'Verified architecture')
    _require(manifest, _expected_manifest(manifest, 'repeat4m', prepared_proof), 'Failed data identity')
    for key, expected in dict(precision='fp16', world_size=1, gradient_accumulation_steps=2,
                              train_rows=4000000, validation_rows=20000).items():
        _require(manifest.get(key), expected, f'Failed training {key}')
    for key, expected in dict(d_model=1024, encoder_layers=8, decoder_layers=8,
                              attention_heads=8, ffn_dim=4096, vocab_size=8000).items():
        _require(manifest['model_config'].get(key), expected, f'Failed model {key}')
    _require(_read(checkpoint/'run_manifest.json'), manifest, 'Checkpoint manifest')
    trainer_state = _read(checkpoint/'trainer_state.json')
    _finite(trainer_state, 'Checkpoint trainer state')
    _require(trainer_state.get('global_step'), RESUME_STEP, 'Checkpoint step')
    arguments = _read(trial/'trainer_args.json')
    for key, expected in dict(num_train_epochs=1., max_steps=-1,
            per_device_train_batch_size=128, per_device_eval_batch_size=128,
            gradient_accumulation_steps=2, learning_rate=.0002, warmup_steps=.05,
            lr_scheduler_type='cosine', fp16=True, bf16=False, use_cpu=False,
            gradient_checkpointing=False, torch_compile=False, seed=42, data_seed=42,
            eval_steps=1000, save_steps=1000, logging_nan_inf_filter=False).items():
        _require(arguments.get(key), expected, f'Frozen trainer argument {key}')
    required = ('trainer_state.json', 'config.json', 'tokenizer.json', 'optimizer.pt',
                'scheduler.pt', 'scaler.pt', 'rng_state.pth', 'run_manifest.json')
    for name in required:
        if not (checkpoint/name).is_file() or (checkpoint/name).stat().st_size == 0:
            raise ValueError(f'Incomplete checkpoint: missing nonempty {name}')
    weights = list(checkpoint.glob('*.safetensors')) + list(checkpoint.glob('pytorch_model*.bin'))
    if not weights or any(path.stat().st_size == 0 for path in weights):
        raise ValueError('Incomplete checkpoint: missing model weights')
    return dict(failed_stage2_dir=str(failed), checkpoint=str(checkpoint), checkpoint_step=RESUME_STEP,
        remaining_step_attempts=UPDATES-RESUME_STEP, planned_step_attempts=UPDATES,
        failed_queue_state_sha256=file_sha256(failed/'queue_state.json'),
        failed_request_sha256=file_sha256(failed/'run_request.json'),
        frozen_manifest_sha256=file_sha256(trial/'run_manifest.json'),
        source_patch=dict(path='kaggle_experiments.py',
            before_sha256=old_sources['kaggle_experiments.py'],
            after_sha256=new_sources['kaggle_experiments.py']),
        manifest=manifest, checkpoint_files=_inventory(checkpoint))


def _sources():
    return dict(stage2._sources(), **{'tools/recover_a100_stage2.py': file_sha256(Path(__file__))})


def _verified_training(trial, config, identity, expected):
    result = verify_training(trial, config, identity)
    _require(_read(Path(trial)/'run_manifest.json'), expected, 'Output-independent training manifest')
    return result


def _invoke(command, *, log, state, state_path, lock, gpu_lock):
    """Keep both locks alive in workers even if their driver is interrupted."""
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {key:value for key,value in os.environ.items() if key not in
           {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'GROUP_RANK'}}
    env.update(MPLBACKEND='Agg', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES='0')
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(), gpu_lock.fileno()))
        try:
            state.update(child_pid=process.pid, log=str(log), updated_at=_now())
            write_json(state_path, state)
            print(f'{state["variant"]} {state["phase"]}: pid={process.pid}; log={log}', flush=True)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    state.update(child_pid=None, updated_at=_now())
    write_json(state_path, state)
    if code:
        raise RuntimeError(f'{state["variant"]} {state["phase"]} exited {code}; see {log}')


def run_recovery(stage1_dir, failed_stage2_dir, prepared_data_dir, output_dir, *, reference_identity):
    stage1, failed, prepared, output, reference = map(lambda p: Path(p).resolve(),
        (stage1_dir, failed_stage2_dir, prepared_data_dir, output_dir, reference_identity))
    for source in (stage1, failed, prepared):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Recovery output must be separate from immutable input directories')
    if output.is_relative_to(GPU_LOCK_DIRECTORY.resolve()):
        raise ValueError('Recovery output must be separate from the shared GPU lock')
    with queue_lock(output) as lock, queue_lock(GPU_LOCK_DIRECTORY) as gpu_lock:
        existing = {path.name for path in output.iterdir()} - {'queue.lock'}
        if existing:
            raise ValueError(f'Recovery requires a fresh output; existing artifacts: {sorted(existing)}')
        state = dict(status='running', stage=2, recovery=True, phase='dependency_verification',
            driver_pid=os.getpid(), child_pid=None, started_at=_now(), completed=[])
        state_path = output/'queue_state.json'
        try:
            write_json(state_path, state)
            first = verify_stage1(stage1, reference)
            proof = verify_prepared(prepared)
            architecture = first['winner']
            recovery = verify_failed_stage2(failed, proof, architecture)
            request = dict(schema=1, stage=2, recovery=recovery, architecture=architecture,
                stage1_dir=str(stage1), prepared_data_dir=str(prepared), output_dir=str(output),
                reference_identity=str(reference), stage1=first, prepared=proof,
                evaluation=EVALUATION, shared_gpu_lock=str(GPU_LOCK_DIRECTORY), sources=_sources())
            request_path = output/'run_request.json'
            write_json(request_path, request)
            request_sha = file_sha256(request_path)
            metrics, diagnostics = {}, {}
            for variant in VARIANTS:
                identity = proof['bundles'][variant]
                trial, evaluation = output/'trials'/variant, output/'evaluations'/variant
                config = make_recovery_config(recovery['manifest'], trial,
                                               manifest_sha=identity['bundle_sha256'])
                expected = _expected_manifest(recovery['manifest'], variant, proof)
                for phase, directory, verify in (
                    ('training', trial, lambda: _verified_training(trial, config, identity, expected)),
                    ('evaluation', evaluation, lambda: verify_evaluation(evaluation,
                        architecture=architecture, checkpoint=trial/'final', identity=identity,
                        reference_identity=reference))):
                    _require(_sources(), request['sources'], 'Immutable recovery source snapshot')
                    state.update(variant=variant, architecture=architecture, phase=phase,
                                 child_pid=None, updated_at=_now())
                    write_json(state_path, state)
                    command = [sys.executable, str(Path(__file__).resolve()), 'worker',
                        '--request', str(request_path), '--request-sha', request_sha,
                        '--variant', variant, '--phase', phase]
                    if (variant, phase) == ('repeat4m', 'training'):
                        command += ['--resume-from', recovery['checkpoint']]
                    result = ensure_phase(directory, output/'verified'/f'{variant}_{phase}.json',
                        request_sha=request_sha, verify=verify,
                        launch=lambda: _invoke(command, log=output/'logs'/f'{variant}_{phase}.log',
                            state=state, state_path=state_path, lock=lock, gpu_lock=gpu_lock))
                    _require(_sources(), request['sources'], 'Immutable recovery source snapshot')
                    if phase == 'evaluation':
                        metrics[variant] = result
                    elif 'amp_diagnostics' in result:
                        diagnostics[variant] = result['amp_diagnostics']
                state['completed'].append(variant)
                write_json(state_path, state)
            ranking = sorted(VARIANTS, key=lambda name: metrics[name]['pass@1'], reverse=True)
            summary = dict(schema=1, stage=2, status='complete', architecture=architecture,
                request_sha256=request_sha, models=metrics, ranking=ranking,
                validation_sha256=VALIDATION_SHA, num_evaluated=ELIGIBLE,
                optimizer_updates_per_epoch=UPDATES, planned_step_attempts=UPDATES, epochs=1,
                optimizer_updates_per_epoch_interpretation=(
                    'Scheduled Trainer step attempts; successful weight updates can be fewer on AMP overflow.'),
                amp_step_semantics=('Native fused AdamW can skip a weight update while Accelerate reports '
                    'step_was_skipped=False; the native Trainer/scheduler behavior is unchanged. '
                    'AMP diagnostics count overflows in each current process session.'),
                recovery={key:value for key,value in recovery.items() if key != 'manifest'},
                amp_diagnostics=diagnostics, finished_at=_now())
            write_json(output/'stage2_complete.json', summary)
            state.update(status='complete', phase='complete', child_pid=None, ranking=ranking, updated_at=_now())
            write_json(state_path, state)
            return summary
        except BaseException as error:
            state.update(status='failed', error=f'{type(error).__name__}: {error}',
                         child_pid=None, updated_at=_now())
            write_json(state_path, state)
            raise


def worker(args):
    _require(file_sha256(args.request), args.request_sha, 'Recovery request SHA')
    request = _read(args.request)
    _require(_sources(), request['sources'], 'Immutable recovery worker source snapshot')
    variant, phase = args.variant, args.phase
    if variant not in VARIANTS or phase not in ('training', 'evaluation'):
        raise ValueError('Unsupported recovery phase')
    recovery = request['recovery']
    resume = recovery['checkpoint'] if (variant, phase) == ('repeat4m', 'training') else None
    _require(args.resume_from, resume, 'Resume is required only for repeat4m training')
    output = Path(request['output_dir'])
    trial = output/'trials'/variant
    destination = trial if phase == 'training' else output/'evaluations'/variant
    if destination.exists():
        raise ValueError(f'Refusing to overwrite existing recovery output: {destination}')
    identity = request['prepared']['bundles'][variant]
    bundle = Path(request['prepared_data_dir'])/variant
    config = make_recovery_config(recovery['manifest'], trial, manifest_sha=identity['bundle_sha256'])
    if phase == 'training':
        if resume:
            _require(_inventory(Path(resume)), recovery['checkpoint_files'], 'Immutable checkpoint inventory')
        from kaggle_experiments import run_experiment
        result = run_experiment(str(bundle), config, resume_from_checkpoint=resume)
        _require(_read(trial/'run_manifest.json'), _expected_manifest(recovery['manifest'], variant,
                 request['prepared']), 'Output-independent training manifest')
    else:
        from tools.evaluate_fixed_bundle import run
        result = run(str(bundle), str(trial/'final'), str(destination), architecture=config.architecture,
            expected_rows=EXPECTED_ROWS, expected_manifest_sha=identity['bundle_sha256'], **EVALUATION)
    _require(_sources(), request['sources'], 'Immutable recovery worker source snapshot')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        parser.add_argument('worker')
        for name in ('request', 'request-sha'):
            parser.add_argument('--'+name, required=True)
        parser.add_argument('--variant', choices=VARIANTS, required=True)
        parser.add_argument('--phase', choices=('training', 'evaluation'), required=True)
        parser.add_argument('--resume-from')
        worker(parser.parse_args())
    else:
        for name in ('stage1-dir', 'failed-stage2-dir', 'prepared-data-dir', 'output-dir', 'reference-identity'):
            parser.add_argument('--'+name, required=True)
        print(json.dumps(run_recovery(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
