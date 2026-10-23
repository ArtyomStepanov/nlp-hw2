"""Saved-result auditing must catch wrong denominators, altered rows and wrong weights."""
import importlib
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    assert importlib.util.find_spec('tools.summarize_full_evaluation'), 'Comparison tool is missing'
    return importlib.import_module('tools.summarize_full_evaluation')


def put(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in value) if lines else json.dumps(value))


@pytest.fixture
def saved_results(tmp_path):
    golden = [dict(sample_row=i, evaluable=i != 5,
        reference_status='ok' if i != 5 else 'runtime_error',
        reference_result_json='10' if i != 5 else None) for i in range(6)]
    for name, successes in (('full', {0, 1, 4}), ('old', {0, 2})):
        predictions, execution = [], []
        for i in range(6):
            success = i in successes
            text = '10' if success else '0'
            if i == 5 and name == 'full':
                text = 'changed excluded prediction'
            prediction = dict(sample_row=i, question=f'question {i}', code='reference',
                generated_code=text, generated_tokens=1, ended_with_eos=True, hit_token_limit=False)
            result = dict(prediction, **{key:value for key,value in golden[i].items() if key != 'sample_row'},
                execution_status='ok', result_json='10' if success else '0',
                passed=success if i != 5 else None,
                status=('passed' if success else 'wrong_answer') if i != 5 else 'ok')
            predictions.append(prediction)
            execution.append(result)
        count = 3 if name == 'full' else 2
        metrics = dict(num_examples=6, num_evaluated=5, num_passed=count, num_bad_code=0,
            **{'pass@1': .6 if name == 'full' else .4}, bad_code_rate=0.,
            wrong_answer_rate=.4 if name == 'full' else .6, reference_error_rate=1/6,
            evaluation_coverage=5/6, status_counts={'passed':count,'wrong_answer':5-count,'ok':1},
            reference_status_counts={'ok':5,'runtime_error':1}, num_reference_errors=1,
            num_hit_token_limit=0, passed_fraction_of_all_sampled=count/6)
        destination = tmp_path/name
        put(destination/'predictions.jsonl', predictions, lines=True)
        put(destination/'evaluation/evaluation.jsonl', execution, lines=True)
        put(destination/'metrics.json', metrics)
        put(destination/'evaluation/metrics.json', {k:v for k,v in metrics.items()
            if k not in ('num_reference_errors','num_hit_token_limit','passed_fraction_of_all_sampled')})
        put(destination/'evaluation/evaluation_config.json', dict(relative_tolerance=1e-6,absolute_tolerance=1e-6))
        put(destination/'progress.json', dict(generated=6,total=6,phase='complete'))
    return tmp_path, golden


def test_recomputes_paired_outcomes_with_one_common_denominator(saved_results):
    root, golden = saved_results
    tool = module()
    full, full_rows = tool.audit_evaluation(root/'full', golden)
    old, old_rows = tool.audit_evaluation(root/'old', golden)
    assert full['num_passed'] == 3 and old['num_passed'] == 2
    paired = tool.paired_outcomes(full_rows, old_rows)
    assert paired == dict(both_pass=1, full_only=2, distinct4m_only=1, neither_pass=1,
        excluded_both=1, common_eligible=5, net_additional_full_passes=1,
        delta_percentage_points=20., prediction_changes=4, eligible_prediction_changes=3)


@pytest.mark.parametrize('change', ['golden', 'false_pass', 'denominator', 'prediction', 'nonfinite'])
def test_rejects_saved_evidence_drift(saved_results, change):
    root, golden = saved_results
    destination = root/'full'
    if change == 'golden':
        golden[3]['reference_result_json'] = '999'
    elif change in ('false_pass','prediction'):
        path = destination/('evaluation/evaluation.jsonl' if change == 'false_pass' else 'predictions.jsonl')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[2]['passed' if change == 'false_pass' else 'generated_code'] = True if change == 'false_pass' else 'tampered'
        put(path, rows, lines=True)
    else:
        path = destination/'metrics.json'
        metrics = json.loads(path.read_text())
        metrics['num_evaluated' if change == 'denominator' else 'generation_seconds'] = 6 if change == 'denominator' else float('nan')
        put(path, metrics)
    with pytest.raises(ValueError):
        module().audit_evaluation(destination, golden)


def test_rejects_different_question_despite_identical_reference_result(saved_results):
    root, golden = saved_results
    tool = module()
    _, full_rows = tool.audit_evaluation(root/'full', golden)
    _, old_rows = tool.audit_evaluation(root/'old', golden)
    full_rows[4]['question'] = 'different question'
    with pytest.raises(ValueError, match='identity'):
        tool.paired_outcomes(full_rows, old_rows)


@pytest.mark.parametrize('change', ['step', 'weights'])
def test_rejects_latest_or_wrong_weights_in_full_provenance(change):
    tool = module()
    root = Path(__file__).resolve().parents[1]
    training = root/'reports/a100_full_20261023/final_report'
    request = json.loads((training/'training/run_request.json').read_text())
    manifest = training/'training/trial/run_manifest.json'
    completion = training/'training/trial/complete.json'
    config = dict(checkpoint_step=180000, training_global_step=181144,
        training_bundle_sha256=request['bundle']['bundle_sha256'],
        training_selection_sha256=request['bundle']['selection_sha256'],
        training_manifest_sha256=tool.file_sha256(manifest),
        training_completion_sha256=tool.file_sha256(completion),
        checkpoint_weight_sha256={'model.safetensors':'2059268173416c77cd8c901c791eb86e7cbcea0ede756b2bb830ec23638b8165'})
    tool.verify_full_provenance(config, training)
    config['checkpoint_step' if change == 'step' else 'checkpoint_weight_sha256'] = 181144 if change == 'step' else {'model.safetensors':'0'*64}
    with pytest.raises(ValueError):
        tool.verify_full_provenance(config, training)


def test_download_receipt_must_cover_the_artifacts_being_compared(tmp_path):
    tool = module()
    put(tmp_path/'metrics.json', {'num_passed':3})
    put(tmp_path/'predictions.jsonl', [{'sample_row':0}], lines=True)
    receipt = dict(files={'metrics.json':dict(size=(tmp_path/'metrics.json').stat().st_size,
        sha256=tool.file_sha256(tmp_path/'metrics.json'))})
    with pytest.raises(ValueError, match='cover'):
        tool.verify_receipt(tmp_path, receipt, required_paths=[tmp_path/'predictions.jsonl'])
