"""Compare five fixed decoders on first 2k validation, then confirm one on 20k.

Use a NEW output directory. Failures retain evidence and require inspection;
there is no resume, OOM retry, training, or T4 session launch.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kaggle_experiments import file_sha256, load_bundle, write_json
from tools import evaluate_fixed_bundle
from tools.run_a100_stage1 import verify_reference_results

VALIDATION_SHA = '282001d478bffb2d13ab2ca41e8232a0c99c3623f898dd7077128c266172c406'
GRID = tuple(dict(num_beams=beams, length_penalty=penalty, batch_size=batch)
             for beams, penalty, batch in ((2, 1., 64), (4, 1., 64), (8, 1., 32),
                                            (4, .8, 64), (4, 1.2, 64)))
TIMING_SCOPE = 'A100 evaluation timing only; not a T4 compliance measurement.'


def _read(path):
    return json.loads(Path(path).read_text())


def _require(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label} mismatch: expected {expected!r}, got {actual!r}')


def validation_sha256(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n').encode())
    return digest.hexdigest()


def select_winner(candidates):
    """Rank raw pass counts only after checking the common eligibility denominator."""
    denominators = {row['num_evaluated'] for row in candidates}
    if len(denominators) != 1 or next(iter(denominators)) <= 0:
        raise ValueError('Decoder comparison requires the same positive denominator')
    return min(candidates, key=lambda row: (-row['num_passed'], row['num_beams'],
                                           abs(row['length_penalty']-1.)))


def verify_evaluation(output, *, architecture, checkpoint, checkpoint_step, identity,
                      reference, decode, count, validation_sha):
    """Check saved decoding configuration and every corresponding golden reference."""
    if count not in (2000, 20000):
        raise ValueError('Only first 2000 selection or full 20000 confirmation is allowed')
    if count == 20000:
        _require(validation_sha, reference['validation_sha256'], 'Full validation hash')
    output = Path(output)
    expected = dict(decode, architecture=architecture, checkpoint=str(checkpoint),
        checkpoint_step=checkpoint_step, validation_rows=count, validation_sha256=validation_sha,
        bundle_sha256=identity['bundle_sha256'], selection_sha256=identity['selection_sha256'],
        device='cuda:0', max_new_tokens=256, do_sample=False, use_cache=True, official_test_used=False)
    config, metrics = _read(output/'config.json'), _read(output/'metrics.json')
    for key, value in expected.items():
        _require(config.get(key), value, f'Evaluation configuration {key}')
    references = reference['rows'][:count]
    eligible = sum(row['evaluable'] for row in references)
    for key, value in dict(architecture=architecture, validation_sha256=validation_sha,
            num_examples=count, num_evaluated=eligible, num_reference_errors=count-eligible,
            num_beams=decode['num_beams'], length_penalty=decode['length_penalty'],
            max_new_tokens=256).items():
        _require(metrics.get(key), value, f'Evaluation metrics {key}')
    passed = metrics.get('num_passed')
    if type(passed) is not int or not 0 <= passed <= eligible:
        raise ValueError('Invalid num_passed')
    _require(metrics.get('pass@1'), passed/eligible, 'Evaluation pass@1')
    seconds = metrics.get('generation_seconds')
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid generation_seconds')
    verify_reference_results(output/'evaluation/evaluation.jsonl', references)
    with (output/'evaluation/evaluation.jsonl').open() as stream:
        _require(sum(row['passed'] is True for row in map(json.loads, stream)), passed,
                 'Saved execution pass count')
    with (output/'predictions.jsonl').open() as stream:
        _require([row['sample_row'] for row in map(json.loads, stream)], list(range(count)),
                 'Saved prediction row order')
    _require(_read(output/'progress.json').get('phase'), 'complete', 'Evaluation completion')
    return metrics


def run_decoding(bundle_root, checkpoint, output_dir, *, architecture, reference_identity,
                 expected_train_rows=4000000):
    if architecture not in ('bart', 'copy') or expected_train_rows not in (1000000, 4000000):
        raise ValueError('Use bart|copy and exactly 1M or 4M training rows')
    bundle_root, checkpoint, output, golden = map(lambda path: Path(path).resolve(),
        (bundle_root, checkpoint, output_dir, reference_identity))
    output.mkdir(parents=True, exist_ok=False)
    with (output/'queue.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status='running', stage=4, phase='input_verification',
                     started_at=datetime.now(timezone.utc).isoformat())
        write_json(output/'queue_state.json', state)
        try:
            reference = _read(golden)
            _require(reference['validation_sha256'], VALIDATION_SHA, 'Golden validation hash')
            _require(len(reference['rows']), 20000, 'Golden reference count')
            _require([row['sample_row'] for row in reference['rows']], list(range(20000)),
                     'Golden reference order')
            _require(sum(row['evaluable'] for row in reference['rows']), 19886, 'Golden eligibility')
            manifest = _read(checkpoint.parent/'run_manifest.json')
            completed = _read(checkpoint.parent/'complete.json')
            for key, value in dict(train_rows=expected_train_rows, validation_rows=20000).items():
                _require(manifest.get(key), value, f'Training manifest {key}')
            _require(manifest['config']['architecture'], architecture, 'Training architecture')
            _require(manifest['config']['expected_rows'], [expected_train_rows, 20000], 'Training row contract')
            _require(completed['architecture'], architecture, 'Completed architecture')
            step = completed['global_step']
            if type(step) is not int or step <= 0 or not checkpoint.is_dir():
                raise ValueError('A completed saved checkpoint is required')
            data, _, identity = load_bundle(bundle_root, expected_rows=(expected_train_rows, 20000),
                expected_manifest_sha=manifest['bundle_sha256'])
            for key in ('bundle_sha256', 'selection_sha256'):
                _require(identity[key], manifest[key], f'Training bundle {key}')
            rows = list(data['val'])
            _require(len(rows), 20000, 'Full validation count')
            full_sha = validation_sha256(rows)
            _require(full_sha, VALIDATION_SHA, 'Original ordered 20k validation')
            subset_sha = validation_sha256(rows[:2000])
            request = dict(architecture=architecture, checkpoint=str(checkpoint), checkpoint_step=step,
                bundle_root=str(bundle_root), expected_train_rows=expected_train_rows,
                bundle_sha256=identity['bundle_sha256'], selection_sha256=identity['selection_sha256'],
                training_manifest_sha256=file_sha256(checkpoint.parent/'run_manifest.json'),
                reference_identity=str(golden), reference_identity_sha256=file_sha256(golden),
                subset_validation_sha256=subset_sha, full_validation_sha256=full_sha,
                subset_rows=2000, full_validation_rows=20000, decoders=GRID, max_new_tokens=256,
                selection_rule='Highest num_passed; ties: fewer beams, then length_penalty closest to 1.0.',
                batch_note='64 for beams 2/4; 32 for beam 8 (256 beam-expanded examples, matching beam 4).',
                timing_scope=TIMING_SCOPE)
            write_json(output/'run_request.json', request)
            write_json(output/'t4_benchmark_required.json', dict(status='pending',
                reason=TIMING_SCOPE, next_step='Benchmark the confirmed decoder on T4 separately.'))

            def evaluate(decode, destination, count):
                if destination.exists():
                    raise FileExistsError(f'Refusing to reuse evaluation output: {destination}')
                state.update(phase='selection' if count == 2000 else 'confirmation',
                             decoder=decode, evaluation_output=str(destination))
                write_json(output/'queue_state.json', state)
                # The helper generates using input_ids/attention_mask only. It moves the
                # model to CPU, deletes it and empties CUDA cache before execution checks.
                evaluate_fixed_bundle.run(bundle_root, checkpoint, destination, architecture=architecture,
                    expected_rows=(expected_train_rows, 20000), expected_manifest_sha=manifest['bundle_sha256'],
                    device='cuda:0', max_new_tokens=256, limit=2000 if count == 2000 else None, **decode)
                return verify_evaluation(destination, architecture=architecture, checkpoint=checkpoint,
                    checkpoint_step=step, identity=identity, reference=reference, decode=decode,
                    count=count, validation_sha=subset_sha if count == 2000 else full_sha)

            candidates = []
            for index, decode in enumerate(GRID):
                destination = output/'selection'/f'{index+1}_beam{decode["num_beams"]}_lp{decode["length_penalty"]:g}'
                metrics = evaluate(decode, destination, 2000)
                candidates.append(dict(metrics, batch_size=decode['batch_size'], output=str(destination)))
            winner = select_winner(candidates)
            decode = {key:winner[key] for key in ('num_beams', 'length_penalty', 'batch_size')}
            confirmation = evaluate(decode, output/'confirmation', 20000)
            summary = dict(status='complete', stage=4, request=request, candidates=candidates,
                winner=winner, confirmation=confirmation, timing_scope=TIMING_SCOPE,
                t4_benchmark_status='pending', finished_at=datetime.now(timezone.utc).isoformat())
            write_json(output/'stage4_complete.json', summary)
            state.update(status='complete', phase='complete')
            write_json(output/'queue_state.json', state)
            return summary
        except BaseException as error:
            state.update(status='failed', error=str(error))
            write_json(output/'queue_state.json', state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle_root', 'checkpoint', 'output_dir', 'reference_identity'):
        parser.add_argument('--'+name.replace('_', '-'), required=True)
    parser.add_argument('--architecture', choices=('bart', 'copy'), required=True)
    parser.add_argument('--expected-train-rows', type=int, choices=(1000000, 4000000), default=4000000)
    run_decoding(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
