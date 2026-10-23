"""Run one fresh BART8+8 full-data schedule, with explicit exact-state resume only."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kaggle_experiments import KaggleConfig, file_sha256, load_bundle, write_json
from tools.prepare_a100_full import FULL_ROWS, BASE_ROWS, FINGERPRINTS, ORIGINAL_MANIFEST_SHA, TOKENIZER_SHA

# The recovery driver uses this same lock, excluding overlapping GPU jobs.
GPU_LOCK_DIRECTORY = Path('/tmp')/f'nlp_hw2_stage2_recovery_cuda0_uid{os.getuid()}'


def _read(path):
    return json.loads(Path(path).read_text())


def _now():
    return datetime.now(timezone.utc).isoformat()


def _require(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label} mismatch: expected {expected!r}, got {actual!r}')


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError('Nonfinite full training result')
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another active driver holds the GPU/output lock: {path}') from error
        yield stream


def make_training_config(output_dir, manifest_sha):
    return KaggleConfig(architecture='bart', output_dir=str(Path(output_dir)),
        micro_batch=128, global_batch=256, learning_rate=2e-4, num_train_epochs=4.,
        eval_steps=5000, early_stopping_patience=5, precision='fp16',
        gradient_checkpointing=False, torch_compile=False, use_cpu=False,
        tiny_model=False, preflight=False, max_steps=-1, expected_rows=FULL_ROWS,
        expected_manifest_sha=manifest_sha, seed=42)


def verify_full_bundle(bundle_root):
    """Rehash the sealed full bundle, retaining its fixed complete-source identity."""
    import numpy as np
    root = Path(bundle_root)
    sha = file_sha256(root/'bundle_manifest.json')
    manifest = _read(root/'bundle_manifest.json')
    _require(manifest.get('derive_length'), True, 'Full bundle length derivation')
    paths = [entry['path'] for entry in manifest['files']]
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p != root/'bundle_manifest.json'}
    if len(paths) != len(set(paths)) or set(paths) != actual:
        raise ValueError('Full bundle integrity inventory mismatch')
    _require(_read(root/'data/dataset_dict.json'), {'splits': ['train', 'val']}, 'Full bundle splits')
    receipt = _read(root/'preparation.json')
    _require(receipt.get('status'), 'complete', 'Full preparation status')
    _require(receipt.get('official_test_used'), False, 'Official test exclusion')
    data, _, identity = load_bundle(root, expected_rows=FULL_ROWS, expected_manifest_sha=sha)
    selection = identity['selection']
    _require(selection.get('strategy'), 'full_original_order', 'Full selection strategy')
    _require(selection['identity']['tokenizer_sha256'], TOKENIZER_SHA, 'Original BPE tokenizer')
    proof = selection['provenance']['historical_selected_row_proof']
    _require(proof.get('original_bundle_sha256'), ORIGINAL_MANIFEST_SHA, 'Original bundle SHA')
    _require(proof.get('verified_original_rows'), dict(zip(('train', 'val'), BASE_ROWS)),
             'Historical selected-row proof')
    for name, count in zip(('train', 'val'), FULL_ROWS):
        details = selection['splits'][name]
        _require(details.get('source_rows'), count, f'Full {name} source rows')
        _require(details.get('selected_rows'), count, f'Full {name} selected rows')
        _require(details.get('source_fingerprint'), FINGERPRINTS[name], f'Full {name} fingerprint')
        _require(proof['full_cache'][name]['fingerprint'], FINGERPRINTS[name],
                 f'Historical {name} fingerprint')
        values = np.load(root/f'selection/{name}_indices.npy', mmap_mode='r', allow_pickle=False)
        if not np.issubdtype(values.dtype, np.integer) or values.shape != (count,):
            raise ValueError(f'Full {name} selection must be a one-dimensional integer array')
        for start in range(0, count, 1_000_000):
            stop = min(start+1_000_000, count)
            if not np.array_equal(values[start:stop], np.arange(start, stop)):
                raise ValueError(f'Full {name} indices must preserve every original row in order')
    del data
    return dict(bundle_sha256=sha, selection_sha256=identity['selection_sha256'],
                tokenizer_sha256=TOKENIZER_SHA)


def verify_stage2(stage2_dir):
    path = Path(stage2_dir)/'stage2_complete.json'
    value = _read(path)
    for key, expected in dict(schema=1, stage=2, status='complete', architecture='bart',
                              ranking=['distinct4m', 'repeat4m']).items():
        _require(value.get(key), expected, f'Stage2 {key}')
    scores = [value['models'][name]['pass@1'] for name in ('distinct4m', 'repeat4m')]
    if any(type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
           for score in scores) or scores[0] <= scores[1]:
        raise ValueError('Stage2 distinct4m winner is inconsistent with saved scores')
    return dict(completion_sha256=file_sha256(path), architecture='bart', ranking=value['ranking'])


def _sources():
    names = ['kaggle_experiments.py', 'training.py', 'plotting.py',
             'tools/prepare_a100_full.py', 'tools/run_a100_full.py']
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'models').glob('*.py'))]
    return {name: file_sha256(ROOT/name) for name in names}


def _resume_checkpoint(checkpoint, trial, config, identity):
    from kaggle_experiments import config_manifest
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_relative_to(trial) or not checkpoint.is_dir():
        raise ValueError('Explicit resume must use a checkpoint inside the same full-run trial')
    required = ('config.json', 'optimizer.pt', 'scheduler.pt', 'scaler.pt', 'rng_state.pth',
                'trainer_state.json', 'run_manifest.json', 'tokenizer.json')
    for name in required:
        path = checkpoint/name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Exact FP16 resume requires {name}')
    weights = list(checkpoint.glob('*.safetensors')) + list(checkpoint.glob('pytorch_model*.bin'))
    if not weights or any(path.stat().st_size == 0 for path in weights):
        raise ValueError('Exact resume requires saved model weights')
    manifest = _read(checkpoint/'run_manifest.json')
    for key, expected in dict(config=config_manifest(config), precision='fp16', world_size=1,
            gradient_accumulation_steps=2, bundle_sha256=identity['bundle_sha256'],
            selection_sha256=identity['selection_sha256'], train_rows=FULL_ROWS[0],
            validation_rows=FULL_ROWS[1]).items():
        _require(manifest.get(key), expected, f'Resume {key}')
    _require(file_sha256(checkpoint/'tokenizer.json'), TOKENIZER_SHA, 'Resume tokenizer')
    step = _read(checkpoint/'trainer_state.json').get('global_step')
    if type(step) is not int or not 0 < step < 4*math.ceil(FULL_ROWS[0]/256):
        raise ValueError('Resume requires an unfinished positive checkpoint step')
    return checkpoint, step


def run_full(bundle_root, output_dir, stage2_dir, *, resume_from=None, previous_stage4_dir=None):
    from kaggle_experiments import config_manifest, run_experiment
    bundle, output, stage2 = map(lambda p: Path(p).resolve(), (bundle_root, output_dir, stage2_dir))
    for source in (bundle, stage2):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Full-run output must be separate from immutable inputs')
    if output.exists() and not resume_from:
        raise FileExistsError(f'Full training requires a fresh output: {output}')
    if resume_from and not output.is_dir():
        raise ValueError('Explicit resume requires the original full-run output directory')
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
    with _lock(GPU_LOCK_DIRECTORY/'queue.lock'), _lock(output/'queue.lock'):
        if not resume_from and any(p.name != 'queue.lock' for p in output.iterdir()):
            raise FileExistsError(f'Full training requires a fresh output: {output}')
        state_path = output/'queue_state.json'
        state = dict(status='running', phase='dependency_verification', driver_pid=os.getpid(),
                     architecture='bart', started_at=_now())
        try:
            write_json(state_path, state)
            previous = None
            if previous_stage4_dir:
                prior = Path(previous_stage4_dir).resolve()
                with _lock(prior/'queue.lock'):
                    _require(_read(prior/'queue_state.json').get('status'), 'complete', 'Stage4 queue status')
                    marker = prior/'stage4_complete.json'
                    _require(_read(marker).get('status'), 'complete', 'Stage4 completion')
                    previous = dict(path=str(prior), completion_sha256=file_sha256(marker))
            first = verify_stage2(stage2)
            proof = verify_full_bundle(bundle)
            trial = output/'trial'
            config = make_training_config(trial, proof['bundle_sha256'])
            request = dict(schema=1, architecture='bart', initialization='from_scratch',
                bundle_root=str(bundle), stage2_dir=str(stage2), stage2=first, bundle=proof,
                config=config_manifest(config), previous_stage4=previous,
                model=dict(d_model=1024, encoder_layers=8, decoder_layers=8,
                           attention_heads=8, ffn_dim=4096, vocab_size=8000),
                planned_step_attempts=4*math.ceil(FULL_ROWS[0]/256),
                sources=_sources(), official_test_used=False)
            request_path = output/'run_request.json'
            checkpoint = None
            if resume_from:
                _require(_read(request_path), request, 'Immutable full-run request')
                if (output/'full_complete.json').exists():
                    raise ValueError('Completed full training cannot be resumed')
                checkpoint, step = _resume_checkpoint(resume_from, trial, config, proof)
                resume_receipt = output/f'resume_request-{step}.json'
                with resume_receipt.open('x') as stream:
                    json.dump(dict(checkpoint=str(checkpoint), step=step, requested_at=_now()), stream)
            else:
                write_json(request_path, request)
            state.update(phase='training', trial=str(trial), updated_at=_now(),
                         resume_from=str(checkpoint) if checkpoint else None)
            write_json(state_path, state)
            result = run_experiment(bundle, config, resume_from_checkpoint=checkpoint)
            _require(_sources(), request['sources'], 'Immutable full-training sources')
            _finite(result)
            _require(_read(trial/'complete.json'), result, 'Training completion receipt')
            for key, expected in dict(architecture='bart', precision='fp16', world_size=1,
                                      effective_batch=256).items():
                _require(result.get(key), expected, f'Training result {key}')
            if type(result.get('global_step')) is not int or result['global_step'] < 1:
                raise ValueError('Training must complete at least one step')
            loss = result.get('metrics', {}).get('eval_token_loss')
            if type(loss) not in (int, float) or not math.isfinite(loss):
                raise ValueError('Training requires a finite final validation token loss')
            summary = dict(schema=1, status='complete', architecture='bart',
                request_sha256=file_sha256(request_path), result=result, finished_at=_now())
            write_json(output/'full_complete.json', summary)
            state.update(status='complete', phase='complete', updated_at=_now())
            write_json(state_path, state)
            return summary
        except BaseException as error:
            state.update(status='failed', error=f'{type(error).__name__}: {error}', updated_at=_now())
            write_json(state_path, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle-root', 'output-dir', 'stage2-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--previous-stage4-dir', type=Path)
    print(json.dumps(run_full(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
