"""Decoder selection and saved-result gates; no model or GPU is needed."""
import importlib.util
import json
from pathlib import Path

import pytest


GOLDEN = Path(__file__).resolve().parents[1]/'reports/a100_sequential_20261022/reference_identity.json'


def api():
    assert importlib.util.find_spec('tools.run_a100_decoding'), 'Decoder comparison runner is missing'
    from tools import run_a100_decoding
    return run_a100_decoding


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_selection_uses_pass_count_with_fixed_denominator_and_deterministic_ties():
    runner = api()
    candidates = [dict(num_beams=4, length_penalty=.8, num_passed=50, num_evaluated=1988),
                  dict(num_beams=4, length_penalty=1., num_passed=50, num_evaluated=1988),
                  dict(num_beams=2, length_penalty=1., num_passed=50, num_evaluated=1988)]
    assert runner.select_winner(candidates) == candidates[2]
    assert runner.select_winner(candidates[:2]) == candidates[1]
    candidates[0]['num_passed'] = 51
    assert runner.select_winner(candidates) == candidates[0]
    candidates[0]['num_evaluated'] = 1987
    with pytest.raises(ValueError, match='denominator'):
        runner.select_winner(candidates)


def result_fixture(root, *, count=2000):
    runner = api()
    reference = json.loads(GOLDEN.read_text())
    references = reference['rows'][:count]
    eligible = sum(row['evaluable'] for row in references)
    checkpoint = root.parent/'checkpoint/final'
    identity = dict(bundle_sha256='a'*64, selection_sha256='b'*64)
    decode = dict(num_beams=4, length_penalty=1., batch_size=64)
    validation_sha = reference['validation_sha256'] if count == 20000 else 'c'*64
    config = dict(**decode, architecture='bart', checkpoint=str(checkpoint), checkpoint_step=15625,
        validation_rows=count, validation_sha256=validation_sha, **identity,
        device='cuda:0', max_new_tokens=256, do_sample=False, use_cache=True, official_test_used=False)
    put(root/'config.json', config)
    put(root/'progress.json', {'phase':'complete'})
    metrics = dict(architecture='bart', validation_sha256=validation_sha, num_examples=count,
        num_evaluated=eligible, num_passed=1, num_reference_errors=count-eligible,
        num_beams=4, length_penalty=1., max_new_tokens=256, generation_seconds=123.,
        **{'pass@1':1/eligible})
    put(root/'metrics.json', metrics)
    (root/'evaluation').mkdir()
    (root/'evaluation/evaluation.jsonl').write_text(''.join(json.dumps(dict(row, passed=index == 0))+'\n'
        for index, row in enumerate(references)))
    (root/'predictions.jsonl').write_text(''.join(json.dumps(dict(sample_row=i))+'\n' for i in range(count)))
    kwargs = dict(architecture='bart', checkpoint=checkpoint, checkpoint_step=15625,
        identity=identity, reference=reference, decode=decode, count=count, validation_sha=validation_sha)
    return runner, kwargs, metrics


@pytest.mark.parametrize('change', [None, 'reference', 'reference_status', 'eligibility', 'hash', 'batch', 'passed'])
def test_saved_trial_rejects_reference_or_configuration_drift(tmp_path, change):
    output = tmp_path/'evaluation'
    runner, kwargs, metrics = result_fixture(output)
    if change in ('reference', 'reference_status'):
        path = output/'evaluation/evaluation.jsonl'
        before, after = ('"reference_result_json": "20.0"', '"reference_result_json": "21.0"') \
            if change == 'reference' else ('"reference_status": "ok"', '"reference_status": "runtime_error"')
        path.write_text(path.read_text().replace(before, after, 1))
    elif change in ('eligibility', 'passed'):
        metrics['num_evaluated' if change == 'eligibility' else 'num_passed'] -= 1
        put(output/'metrics.json', metrics)
    elif change:
        path = output/'config.json'
        config = json.loads(path.read_text())
        config['validation_sha256' if change == 'hash' else 'batch_size'] = 'd'*64 if change == 'hash' else 32
        put(path, config)
    if change:
        with pytest.raises(ValueError):
            runner.verify_evaluation(output, **kwargs)
    else:
        assert runner.verify_evaluation(output, **kwargs) == metrics


def test_full_confirmation_requires_all_20k_references_and_full_denominator(tmp_path):
    output = tmp_path/'full'
    runner, kwargs, metrics = result_fixture(output, count=20000)
    assert runner.verify_evaluation(output, **kwargs)['num_evaluated'] == 19886
    metrics['num_evaluated'] = 1988
    put(output/'metrics.json', metrics)
    with pytest.raises(ValueError, match='num_evaluated'):
        runner.verify_evaluation(output, **kwargs)


def test_runner_executes_fixed_grid_then_only_its_winner_and_refuses_reuse(monkeypatch, tmp_path):
    runner = api()
    from tools import evaluate_fixed_bundle
    checkpoint = tmp_path/'trial/final'
    checkpoint.mkdir(parents=True)
    identity = dict(bundle_sha256='a'*64, selection_sha256='b'*64)
    reference = json.loads(GOLDEN.read_text())
    put(checkpoint.parent/'run_manifest.json', dict(**identity, train_rows=4000000,
        validation_rows=20000, config=dict(architecture='bart', expected_rows=[4000000, 20000])))
    put(checkpoint.parent/'complete.json', dict(architecture='bart', global_step=15625))
    validation = [dict(input_ids=[i], attention_mask=[1], labels=[2], length=1) for i in range(20000)]
    monkeypatch.setattr(runner, 'load_bundle', lambda *args, **kwargs: ({'val':validation}, None, identity))
    # Stand-in data still exercises real row hashing and every saved-result gate.
    reference['validation_sha256'] = runner.validation_sha256(validation)
    monkeypatch.setattr(runner, 'VALIDATION_SHA', reference['validation_sha256'])
    golden = tmp_path/'golden.json'
    put(golden, reference)
    calls = []
    def evaluate(bundle, saved, output, **settings):
        assert settings['max_new_tokens'] == 256
        assert settings['expected_rows'] == (4000000, 20000)
        assert settings['expected_manifest_sha'] == 'a'*64
        count = settings['limit'] or 20000
        calls.append((settings['num_beams'], settings['length_penalty'], settings['batch_size'], count))
        eligible = sum(row['evaluable'] for row in reference['rows'][:count])
        passed = 100 if settings['num_beams'] == 2 else 90
        decode = {key:settings[key] for key in ('num_beams', 'length_penalty', 'batch_size')}
        sha = runner.validation_sha256(validation[:count])
        put(Path(output)/'config.json', dict(**decode, architecture='bart', checkpoint=str(saved),
            checkpoint_step=15625, validation_rows=count, validation_sha256=sha, **identity,
            device='cuda:0', max_new_tokens=256, do_sample=False, use_cache=True, official_test_used=False))
        put(Path(output)/'progress.json', dict(phase='complete'))
        put(Path(output)/'metrics.json', dict(architecture='bart', validation_sha256=sha,
            num_examples=count, num_evaluated=eligible, num_passed=passed, num_reference_errors=count-eligible,
            num_beams=settings['num_beams'], length_penalty=settings['length_penalty'],
            max_new_tokens=256, generation_seconds=123., **{'pass@1':passed/eligible}))
        (Path(output)/'evaluation').mkdir()
        (Path(output)/'evaluation/evaluation.jsonl').write_text(''.join(json.dumps(dict(row, passed=i < passed))+'\n'
            for i, row in enumerate(reference['rows'][:count])))
        (Path(output)/'predictions.jsonl').write_text(''.join(json.dumps(dict(sample_row=i))+'\n'
            for i in range(count)))
    monkeypatch.setattr(evaluate_fixed_bundle, 'run', evaluate)
    output = tmp_path/'comparison'
    summary = runner.run_decoding(tmp_path/'bundle', checkpoint, output,
        architecture='bart', reference_identity=golden)
    assert calls == [(2, 1., 64, 2000), (4, 1., 64, 2000), (8, 1., 32, 2000),
                     (4, .8, 64, 2000), (4, 1.2, 64, 2000), (2, 1., 64, 20000)]
    assert summary['winner']['num_beams'] == 2
    assert summary['confirmation']['num_evaluated'] == 19886
    assert json.loads((output/'t4_benchmark_required.json').read_text())['status'] == 'pending'
    with pytest.raises(FileExistsError):
        runner.run_decoding(tmp_path/'bundle', checkpoint, output,
            architecture='bart', reference_identity=golden)
    assert len(calls) == 6


def test_missing_inputs_record_failure_without_creating_evaluations(tmp_path):
    runner = api()
    output = tmp_path/'failed'
    with pytest.raises(FileNotFoundError):
        runner.run_decoding(tmp_path/'missing-bundle', tmp_path/'missing-checkpoint', output,
            architecture='copy', reference_identity=GOLDEN)
    state = json.loads((output/'queue_state.json').read_text())
    assert (state['status'], state['phase']) == ('failed', 'input_verification')
    assert not (output/'selection').exists()
    with pytest.raises(FileExistsError):
        runner.run_decoding(tmp_path/'missing-bundle', tmp_path/'missing-checkpoint', output,
            architecture='copy', reference_identity=GOLDEN)
