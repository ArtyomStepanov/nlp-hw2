"""Run the verified, sequential A100 repetition/diversity comparison.

Production settings are deliberately fixed. Interrupted or failed phase outputs
are retained and require inspection plus a new output directory; no automatic
resume, OOM retry, or scientific-setting adjustment is performed.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kaggle_experiments import KaggleConfig, config_manifest, file_sha256, load_bundle, write_json
from tools.run_a100_stage1 import verify_reference_results

EXPECTED_ROWS = (4_000_000, 20_000)
VALIDATION_SHA = '282001d478bffb2d13ab2ca41e8232a0c99c3623f898dd7077128c266172c406'
ELIGIBLE = 19886
UPDATES = 15625
VARIANTS = ('repeat4m', 'distinct4m')
EVALUATION = dict(batch_size=64, num_beams=4, max_new_tokens=256,
                  length_penalty=1.0, device='cuda:0')


def _read(path):
    return json.loads(Path(path).read_text())


def _now():
    return datetime.now(timezone.utc).isoformat()


def _finite(value, label):
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f'Nonfinite {label}')
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f'{label}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f'{label}[{index}]')


def _require(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label} mismatch: expected {expected!r}, got {actual!r}')


def _reference(path):
    reference = _read(path)
    _require(reference['validation_sha256'], VALIDATION_SHA, 'Golden validation identity')
    _require(len(reference['rows']), EXPECTED_ROWS[1], 'Golden reference rows')
    _require([row['sample_row'] for row in reference['rows']], list(range(EXPECTED_ROWS[1])),
             'Golden reference order')
    _require(sum(row['evaluable'] for row in reference['rows']), ELIGIBLE, 'Golden eligibility')
    return reference


def _metrics(path, architecture, reference):
    metrics = _read(path/'metrics.json')
    _finite(metrics, 'evaluation metrics')
    for name, expected in dict(architecture=architecture, validation_sha256=VALIDATION_SHA,
            num_evaluated=ELIGIBLE, num_beams=4, max_new_tokens=256, length_penalty=1.).items():
        _require(metrics.get(name), expected, f'Evaluation {name}')
    score = metrics.get('pass@1')
    if type(score) not in (int, float) or not 0 <= score <= 1:
        raise ValueError('Evaluation pass@1 must be finite and between zero and one')
    verify_reference_results(path/'evaluation/evaluation.jsonl', reference['rows'])
    return metrics


def verify_stage1(stage1_dir, reference_identity):
    """Recheck the winner against both saved evaluations and every golden reference."""
    stage1_dir, reference_identity = map(Path, (stage1_dir, reference_identity))
    reference = _reference(reference_identity)
    completed = _read(stage1_dir/'stage1_complete.json')
    _finite(completed, 'Stage1 completion')
    metrics = {name: _metrics(stage1_dir/'evaluations'/name, name, reference)
               for name in ('bart', 'copy')}
    _require(completed.get('models'), metrics, 'Stage1 saved metrics')
    winner = 'copy' if metrics['copy']['pass@1'] > metrics['bart']['pass@1'] + .005 else 'bart'
    _require(completed.get('winner'), winner, 'Stage1 winner')
    return dict(winner=winner, completion_sha256=file_sha256(stage1_dir/'stage1_complete.json'),
                reference_sha256=file_sha256(reference_identity))


def _validation_sha(data):
    digest = hashlib.sha256()
    for row in data:
        digest.update((json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n').encode())
    return digest.hexdigest()


def verify_prepared(prepared_dir, *, expected_rows=EXPECTED_ROWS,
                    base_rows=(1_000_000, 20_000), repeat_factor=4,
                    full_rows=(11_593_060, 118_302),
                    expected_original_manifest_sha='7480c7d1778bffce50db987019f2ccf87c06f8643c1ce59ad811e4fd3c77d0e1',
                    expected_validation_sha=VALIDATION_SHA):
    """Verify sealed preparation evidence and the bytes of both complete bundles.

    Count/hash overrides permit small real Arrow software tests. The CLI always
    uses the fixed production defaults, including the unchanged 20k validation.
    """
    prepared_dir = Path(prepared_dir)
    completed = _read(prepared_dir/'complete.json')
    _finite(completed, 'Prepared completion')
    for key, expected in dict(schema=1, status='complete', expected_rows=list(expected_rows),
            optimizer_updates_per_epoch=math.ceil(expected_rows[0]/256), epochs=1).items():
        _require(completed.get(key), expected, f'Prepared {key}')
    request = completed['request']
    for key, expected in dict(base_rows=list(base_rows), full_rows=list(full_rows), repeat_factor=repeat_factor,
                              seed=42, global_batch=256).items():
        _require(request.get(key), expected, f'Prepared request {key}')
    _require(expected_rows[0], base_rows[0]*repeat_factor, 'Prepared training budget')
    _require(expected_rows[1], base_rows[1], 'Prepared fixed validation count')
    _require(set(completed['bundles']), set(VARIANTS), 'Prepared strategies')
    proof, tokenizer_sha, validation_sha, token_rows_sha = {}, None, None, None
    for variant in VARIANTS:
        root = prepared_dir/variant
        sha = file_sha256(root/'bundle_manifest.json')
        _require(sha, completed['bundles'][variant]['bundle_manifest_sha256'], f'{variant} manifest SHA')
        data, _, identity = load_bundle(root, expected_rows=expected_rows, expected_manifest_sha=sha)
        manifest = _read(root/'bundle_manifest.json')
        recorded = [entry['path'] for entry in manifest['files']]
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
                  if p.is_file() and p != root/'bundle_manifest.json'}
        if len(recorded) != len(set(recorded)) or set(recorded) != actual:
            raise ValueError(f'{variant} bundle integrity file inventory mismatch')
        selection = identity['selection']
        _require(selection.get('strategy'), variant, f'{variant} strategy')
        _require(selection.get('provenance'), completed['provenance'], f'{variant} provenance')
        _require(completed['provenance']['verified_original_rows'],
                 dict(zip(('train', 'val'), base_rows)), 'Verified original row counts')
        _require(completed['provenance']['original_bundle_sha256'], expected_original_manifest_sha,
                 'Original fixed bundle manifest')
        original_identity = completed['provenance']['original_selection']['identity']
        for key, value in dict(train_rows=base_rows[0], validation_rows=base_rows[1], seed=42).items():
            _require(original_identity.get(key), value, f'Original selection {key}')
        sha_tokenizer = file_sha256(root/'tokenizer.json')
        _require(sha_tokenizer, original_identity['tokenizer_sha256'], 'Original BPE tokenizer')
        sha_validation = _validation_sha(data['val'])
        _require(sha_validation, expected_validation_sha, 'Fixed validation ordered rows')
        sha_token_rows = selection['splits']['val']['rows_sha256']
        _require(sha_token_rows, completed['provenance']['validation_rows_sha256'],
                 'Fixed validation token rows')
        if tokenizer_sha is not None:
            _require(sha_tokenizer, tokenizer_sha, 'Paired BPE tokenizers')
            _require(sha_validation, validation_sha, 'Paired validation rows')
            _require(sha_token_rows, token_rows_sha, 'Paired validation token rows')
        tokenizer_sha, validation_sha, token_rows_sha = sha_tokenizer, sha_validation, sha_token_rows
        proof[variant] = dict(bundle_sha256=sha, selection_sha256=identity['selection_sha256'],
                              tokenizer_sha256=sha_tokenizer)
        del data
    return dict(completion_sha256=file_sha256(prepared_dir/'complete.json'), bundles=proof)


def make_training_config(architecture, output_dir, manifest_sha, *, micro_batch=128):
    if architecture not in ('bart', 'copy'):
        raise ValueError('Stage2 architecture must be the verified BART/Copy winner')
    if type(micro_batch) is not int or micro_batch != 128:
        raise ValueError('The authorized A100 batch layout is micro128/accum2/global256')
    return KaggleConfig(architecture=architecture, output_dir=output_dir,
        micro_batch=128, global_batch=256, learning_rate=2e-4, num_train_epochs=1.,
        eval_steps=1000, precision='fp16', gradient_checkpointing=False, torch_compile=False,
        expected_rows=EXPECTED_ROWS, expected_manifest_sha=manifest_sha, seed=42)


def verify_training(trial, config, identity):
    trial = Path(trial)
    manifest, result = _read(trial/'run_manifest.json'), _read(trial/'complete.json')
    arguments, history = _read(trial/'trainer_args.json'), _read(trial/'log_history.json')
    _finite([manifest, result, arguments, history], 'Training artifacts')
    settings = config_manifest(config)
    saved_settings = dict(manifest['config'])
    if saved_settings.get('early_stopping_patience') is None:
        saved_settings.pop('early_stopping_patience', None)
    _require(saved_settings, settings, 'Training configuration')
    for key, expected in dict(precision='fp16', world_size=1, gradient_accumulation_steps=2,
            train_rows=4000000, validation_rows=20000, bundle_sha256=identity['bundle_sha256'],
            selection_sha256=identity['selection_sha256']).items():
        _require(manifest.get(key), expected, f'Training manifest {key}')
    model = manifest['model_config']
    for key, expected in dict(d_model=1024, encoder_layers=8, decoder_layers=8,
                              attention_heads=8, ffn_dim=4096, vocab_size=8000).items():
        _require(model.get(key), expected, f'Training model {key}')
    required = dict(num_train_epochs=1., max_steps=-1, per_device_train_batch_size=128,
        per_device_eval_batch_size=128, gradient_accumulation_steps=2, learning_rate=.0002,
        warmup_steps=.05, lr_scheduler_type='cosine', fp16=True, bf16=False, use_cpu=False,
        gradient_checkpointing=False, torch_compile=False, seed=42, data_seed=42,
        eval_steps=1000, save_steps=1000, logging_nan_inf_filter=False)
    for key, expected in required.items():
        _require(arguments.get(key), expected, f'Trainer argument {key}')
    for key, expected in dict(architecture=config.architecture, global_step=UPDATES, precision='fp16',
                              world_size=1, effective_batch=256).items():
        _require(result.get(key), expected, f'Training completion {key}')
    loss = result.get('metrics', {}).get('eval_token_loss')
    if type(loss) not in (float, int) or not math.isfinite(loss):
        raise ValueError('Training completion lacks a finite final validation loss')
    if not history or not any('loss' in entry for entry in history):
        raise ValueError('Training completion lacks finite training loss history')
    _require(_read(trial/'trainer_state.json')['global_step'], UPDATES, 'Trainer state global step')
    _require(_read(trial/'final/model_config.json'), model, 'Saved model configuration')
    saved = _read(trial/'final/config.json')
    for key, expected in dict(model_type={'bart':'bart', 'copy':'copy_bart'}[config.architecture],
            d_model=1024, encoder_layers=8, decoder_layers=8, encoder_attention_heads=8,
            decoder_attention_heads=8, encoder_ffn_dim=4096, decoder_ffn_dim=4096, vocab_size=8000).items():
        _require(saved.get(key), expected, f'Saved checkpoint {key}')
    weights = list((trial/'final').glob('*.safetensors')) + list((trial/'final').glob('pytorch_model*.bin'))
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError('Training completion lacks saved model weights')
    _require(file_sha256(trial/'source_tokenizer.json'), identity['tokenizer_sha256'],
             'Saved source tokenizer')
    return result


def verify_evaluation(output, *, architecture, checkpoint, identity, reference_identity):
    output = Path(output)
    metrics = _metrics(output, architecture, _reference(reference_identity))
    config = _read(output/'config.json')
    _finite(config, 'Evaluation configuration')
    expected = dict(EVALUATION, architecture=architecture, checkpoint=str(Path(checkpoint)),
        checkpoint_step=UPDATES, validation_rows=20000, validation_sha256=VALIDATION_SHA,
        bundle_sha256=identity['bundle_sha256'], selection_sha256=identity['selection_sha256'],
        do_sample=False, use_cache=True, official_test_used=False)
    for key, value in expected.items():
        _require(config.get(key), value, f'Evaluation configuration {key}')
    _require(_read(output/'progress.json').get('phase'), 'complete', 'Evaluation phase')
    if not (output/'predictions.jsonl').is_file():
        raise ValueError('Evaluation completion lacks saved predictions')
    with (output/'predictions.jsonl').open() as stream:
        sample_rows = [row['sample_row'] for row in map(json.loads, stream)]
    _require(sample_rows, list(range(20000)), 'Evaluation prediction count and order')
    return metrics


def _inventory(directory):
    result = []
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Artifact integrity: symlink is not an immutable output: {path}')
        if path.is_file():
            result.append(dict(path=path.relative_to(directory).as_posix(),
                               bytes=path.stat().st_size, sha256=file_sha256(path)))
    return result


def ensure_phase(directory, receipt, *, request_sha, launch, verify):
    """Run once, seal verified outputs, and recheck every byte before any skip."""
    directory, receipt = Path(directory), Path(receipt)
    if directory.exists() or receipt.exists():
        if not directory.is_dir() or not receipt.is_file():
            raise ValueError(f'Incomplete or unverified output exists; inspect and choose a new output directory: {directory}')
        seal = _read(receipt)
        _require(seal.get('status'), 'verified', 'Phase completion status')
        _require(seal.get('request_sha256'), request_sha, 'Phase request integrity')
        _require(seal.get('files'), _inventory(directory), 'Phase artifact integrity')
        result = verify()
        _require(seal.get('result'), result, 'Phase result integrity')
        return result
    launch()
    result = verify()
    write_json(receipt, dict(schema=1, status='verified', request_sha256=request_sha,
        result=result, files=_inventory(directory), verified_at=_now()))
    return result


@contextmanager
def queue_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'queue.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'A Stage2 driver or inherited worker is already running: {output}') from error
        yield stream


def _sources():
    names = ['kaggle_experiments.py', 'training.py', 'evaluation.py', '_code_runner.py', 'plotting.py',
             'tools/run_a100_stage2.py', 'tools/run_a100_stage1.py',
             'tools/evaluate_fixed_bundle.py', 'tools/prepare_a100_diversity.py']
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'models').rglob('*.py'))]
    return {name: file_sha256(ROOT/name) for name in names}


def _invoke(command, *, log, state, state_path, lock):
    log.parent.mkdir(parents=True, exist_ok=True)
    # Remove inherited distributed launch metadata: this is exactly one process/GPU.
    env = {key:value for key, value in os.environ.items() if key not in
           {'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'GROUP_RANK'}}
    env.update(MPLBACKEND='Agg', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES='0')
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
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


def run_stage2(stage1_dir, prepared_data_dir, output_dir, *, micro_batch=128,
               reference_identity=None):
    stage1, prepared, output = (Path(p).resolve() for p in (stage1_dir, prepared_data_dir, output_dir))
    reference = Path(reference_identity or ROOT/'reference_identity.json').resolve()
    for source in (stage1, prepared):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Stage2 output must be separate from immutable input directories')
    make_training_config('bart', output/'unused', '0'*64, micro_batch=micro_batch)
    with queue_lock(output) as lock:
        state = dict(status='running', stage=2, phase='dependency_verification',
            driver_pid=os.getpid(), child_pid=None, started_at=_now(), completed=[])
        state_path = output/'queue_state.json'
        try:
            write_json(state_path, state)
            first = verify_stage1(stage1, reference)
            proof = verify_prepared(prepared)
            architecture = first['winner']
            request = dict(schema=1, stage=2, stage1_dir=str(stage1), prepared_data_dir=str(prepared),
                reference_identity=str(reference), stage1=first, prepared=proof, architecture=architecture,
                micro_batch=micro_batch, global_batch=256, optimizer_updates_per_epoch=UPDATES,
                evaluation=EVALUATION, sources=_sources())
            request_path = output/'run_request.json'
            if request_path.exists():
                _require(_read(request_path), request, 'Immutable Stage2 run request')
            else:
                existing = {p.name for p in output.iterdir()} - {'queue.lock', 'queue_state.json'}
                if existing:
                    raise ValueError(f'Unverified existing Stage2 output: {sorted(existing)}')
                write_json(request_path, request)
            request_sha = file_sha256(request_path)
            metrics = {}
            for variant in VARIANTS:
                bundle = prepared/variant
                identity = proof['bundles'][variant]
                trial, evaluation = output/'trials'/variant, output/'evaluations'/variant
                config = make_training_config(architecture, trial, identity['bundle_sha256'], micro_batch=micro_batch)
                for phase, directory, verify in (
                    ('training', trial, lambda: verify_training(trial, config, identity)),
                    ('evaluation', evaluation, lambda: verify_evaluation(evaluation,
                        architecture=architecture, checkpoint=trial/'final', identity=identity,
                        reference_identity=reference))):
                    _require(_sources(), request['sources'], 'Immutable Stage2 source snapshot')
                    state.update(variant=variant, architecture=architecture, phase=phase,
                                 child_pid=None, updated_at=_now())
                    write_json(state_path, state)
                    command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--phase', phase,
                        '--bundle-root', str(bundle), '--output-dir', str(directory),
                        '--architecture', architecture, '--manifest-sha', identity['bundle_sha256'],
                        '--micro-batch', str(micro_batch)]
                    if phase == 'evaluation':
                        command += ['--checkpoint', str(trial/'final')]
                    result = ensure_phase(directory, output/'verified'/f'{variant}_{phase}.json',
                        request_sha=request_sha, verify=verify,
                        launch=lambda: _invoke(command, log=output/'logs'/f'{variant}_{phase}.log',
                            state=state, state_path=state_path, lock=lock))
                    _require(_sources(), request['sources'], 'Immutable Stage2 source snapshot')
                    if phase == 'evaluation':
                        metrics[variant] = result
                state['completed'].append(variant)
                write_json(state_path, state)
            ranking = sorted(VARIANTS, key=lambda name: metrics[name]['pass@1'], reverse=True)
            summary = dict(schema=1, stage=2, status='complete', architecture=architecture,
                request_sha256=request_sha, models=metrics, ranking=ranking,
                validation_sha256=VALIDATION_SHA, num_evaluated=ELIGIBLE,
                optimizer_updates_per_epoch=UPDATES, epochs=1)
            marker = output/'stage2_complete.json'
            if marker.exists():
                old = _read(marker)
                _require({k:v for k,v in old.items() if k != 'finished_at'}, summary,
                         'Stage2 completion integrity')
                summary = old
            else:
                summary['finished_at'] = _now()
                write_json(marker, summary)
            state.update(status='complete', phase='complete', child_pid=None, ranking=ranking, updated_at=_now())
            write_json(state_path, state)
            return summary
        except BaseException as error:
            state.update(status='failed', error=f'{type(error).__name__}: {error}',
                         child_pid=None, updated_at=_now())
            write_json(state_path, state)
            raise


def worker(args):
    config = make_training_config(args.architecture, args.output_dir, args.manifest_sha,
                                  micro_batch=args.micro_batch)
    if args.phase == 'training':
        from kaggle_experiments import run_experiment
        return run_experiment(args.bundle_root, config)
    if not args.checkpoint:
        raise ValueError('Evaluation worker requires --checkpoint')
    from tools.evaluate_fixed_bundle import run
    return run(args.bundle_root, args.checkpoint, args.output_dir, architecture=args.architecture,
        expected_rows=EXPECTED_ROWS, expected_manifest_sha=args.manifest_sha, **EVALUATION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        parser.add_argument('worker')
        parser.add_argument('--phase', choices=('training', 'evaluation'), required=True)
        for name in ('bundle-root', 'output-dir', 'architecture', 'manifest-sha'):
            parser.add_argument('--'+name, required=True)
        parser.add_argument('--micro-batch', type=int, default=128)
        parser.add_argument('--checkpoint')
        worker(parser.parse_args())
    else:
        for name in ('stage1-dir', 'prepared-data-dir', 'output-dir'):
            parser.add_argument('--'+name, required=True)
        parser.add_argument('--micro-batch', type=int, default=128)
        parser.add_argument('--reference-identity', type=Path)
        print(json.dumps(run_stage2(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
