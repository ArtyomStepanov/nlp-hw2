"""Verify saved full-model/4M evaluations and write their paired local comparison.

Uses only the standard library. It never generates or executes model predictions,
reads model weights, or changes its input reports.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'reports/a100_sequential_20261022'
NEW = ROOT/'reports/a100_full_eval_20261023'
TRAINING = ROOT/'reports/a100_full_20261023/final_report'
VALIDATION_SHA = '282001d478bffb2d13ab2ca41e8232a0c99c3623f898dd7077128c266172c406'
ORIGINAL_BUNDLE_SHA = '7480c7d1778bffce50db987019f2ccf87c06f8643c1ce59ad811e4fd3c77d0e1'
GOLDEN_SHA = '42fc3fa34f9e7ce9cfbc84fb2ad35dddccceb275e6d249ec7f7d2fe4cef9785b'
BEST_WEIGHT_SHA = '2059268173416c77cd8c901c791eb86e7cbcea0ede756b2bb830ec23638b8165'
IDENTITY_FIELDS = ('sample_row', 'question', 'code')
PREDICTION_FIELDS = (*IDENTITY_FIELDS, 'generated_code', 'generated_tokens', 'ended_with_eos', 'hit_token_limit')
REFERENCE_FIELDS = ('sample_row', 'evaluable', 'reference_status', 'reference_result_json')


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label} mismatch: expected {expected!r}, got {actual!r}')


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Saved metrics contain a nonfinite number')
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def equal_results(actual, expected):
    """Match evaluator equality at the recorded 1e-6 tolerances without execution."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if isinstance(actual, int) and isinstance(expected, int):
            return actual == expected
        try:
            return math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
        except OverflowError:
            return actual == expected
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(equal_results(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equal_results(actual[k], expected[k]) for k in actual)
    return type(actual) is type(expected) and actual == expected


def audit_evaluation(directory, golden):
    """Recompute every saved pass/status and metric; retain records for pairing."""
    directory = Path(directory)
    predictions = read_rows(directory/'predictions.jsonl')
    execution = read_rows(directory/'evaluation/evaluation.jsonl')
    count = len(golden)
    if count < 1 or len(predictions) != count or len(execution) != count:
        raise ValueError('Prediction/execution/golden row counts differ or are empty')
    settings = read_json(directory/'evaluation/evaluation_config.json')
    for name in ('relative_tolerance', 'absolute_tolerance'):
        require(settings[name], 1e-6, f'Evaluator {name}')
    for index, (prediction, result, reference) in enumerate(zip(predictions, execution, golden)):
        require(prediction['sample_row'], index, 'Prediction row order')
        require({key:result[key] for key in PREDICTION_FIELDS}, prediction, 'Prediction/execution identity')
        require({key:result[key] for key in REFERENCE_FIELDS}, reference, f'Golden row {index}')
        eligible = result['reference_status'] == 'ok'
        if result['evaluable'] is not eligible:
            raise ValueError(f'Invalid reference eligibility at row {index}')
        passed = eligible and result['execution_status'] == 'ok' and equal_results(
            json.loads(result['result_json']), json.loads(result['reference_result_json']))
        if result['passed'] is not (passed if eligible else None):
            raise ValueError(f'Saved pass flag disagrees with recorded results at row {index}')
        status = ('passed' if passed else 'wrong_answer') if eligible and result['execution_status'] == 'ok' else result['execution_status']
        require(result['status'], status, f'Saved status at row {index}')
    eligible = sum(row['evaluable'] for row in execution)
    if not eligible:
        raise ValueError('A positive common eligibility denominator is required')
    passed = sum(row['passed'] is True for row in execution)
    bad = sum(row['execution_status'] != 'ok' for row in execution)
    statuses = Counter(row['status'] for row in execution)
    recomputed = dict(num_examples=count, num_evaluated=eligible, num_passed=passed,
        num_bad_code=bad, **{'pass@1': passed/eligible}, bad_code_rate=bad/count,
        wrong_answer_rate=statuses['wrong_answer']/eligible,
        reference_error_rate=(count-eligible)/count, evaluation_coverage=eligible/count,
        status_counts=dict(statuses), reference_status_counts=dict(Counter(row['reference_status'] for row in execution)),
        num_reference_errors=count-eligible, num_hit_token_limit=sum(row['hit_token_limit'] for row in predictions),
        passed_fraction_of_all_sampled=passed/count)
    metrics = read_json(directory/'metrics.json')
    finite(metrics)
    for key, value in recomputed.items():
        require(metrics[key], value, f'Recomputed metric {key}')
    for key, value in read_json(directory/'evaluation/metrics.json').items():
        require(metrics[key], value, f'Evaluator sidecar metric {key}')
    progress = read_json(directory/'progress.json')
    for key, expected in dict(generated=count, total=count, phase='complete').items():
        require(progress[key], expected, f'Evaluation progress {key}')
    return dict(metrics, pass_percent=100*passed/eligible), execution


def paired_outcomes(full, old):
    if len(full) != len(old):
        raise ValueError('Paired row counts differ')
    counts = dict(both_pass=0, full_only=0, distinct4m_only=0, neither_pass=0,
        excluded_both=0, common_eligible=0, net_additional_full_passes=0,
        delta_percentage_points=0., prediction_changes=0, eligible_prediction_changes=0)
    for new, previous in zip(full, old):
        require({key:new[key] for key in IDENTITY_FIELDS},
                {key:previous[key] for key in IDENTITY_FIELDS}, 'Paired input identity')
        require({key:new[key] for key in REFERENCE_FIELDS},
                {key:previous[key] for key in REFERENCE_FIELDS}, 'Paired golden identity')
        changed = new['generated_code'] != previous['generated_code']
        counts['prediction_changes'] += changed
        if not new['evaluable']:
            counts['excluded_both'] += 1
            continue
        counts['common_eligible'] += 1
        counts['eligible_prediction_changes'] += changed
        if new['passed'] and previous['passed']:
            counts['both_pass'] += 1
        elif new['passed']:
            counts['full_only'] += 1
        elif previous['passed']:
            counts['distinct4m_only'] += 1
        else:
            counts['neither_pass'] += 1
    if not counts['common_eligible']:
        raise ValueError('No common eligible rows')
    counts['net_additional_full_passes'] = counts['full_only']-counts['distinct4m_only']
    counts['delta_percentage_points'] = 100*counts['net_additional_full_passes']/counts['common_eligible']
    return counts


def verify_full_provenance(config, training_report):
    training = Path(training_report)
    completed = read_json(training/'training/trial/complete.json')
    manifest = read_json(training/'training/trial/run_manifest.json')
    request = read_json(training/'training/run_request.json')
    proof = read_json(training/'remote_weight_checks.json')
    require(proof['status'], 'passed', 'Remote tensor proof')
    require(proof['final_exactly_matches_best'], True, 'Final/best tensor proof')
    require(proof['all_best_latest_final_tensors_finite'], True, 'Finite tensor proof')
    require(completed['best_global_step'], 180000, 'Training best step')
    require(completed['global_step'], 181144, 'Completed attempts')
    require(completed['final_weights_source'], 'best_eval_token_loss', 'Final export source')
    expected = dict(checkpoint_step=180000, training_global_step=181144,
        training_bundle_sha256=manifest['bundle_sha256'],
        training_selection_sha256=manifest['selection_sha256'],
        training_manifest_sha256=file_sha256(training/'training/trial/run_manifest.json'),
        training_completion_sha256=file_sha256(training/'training/trial/complete.json'),
        checkpoint_weight_sha256={'model.safetensors': BEST_WEIGHT_SHA})
    for key, value in expected.items():
        require(config.get(key), value, f'Full evaluation provenance {key}')
    require(proof['weight_sha256']['final/model.safetensors'], BEST_WEIGHT_SHA, 'Verified final weights')
    require(proof['weight_sha256']['checkpoint-180000/model.safetensors'], BEST_WEIGHT_SHA, 'Verified best weights')
    require(request['bundle']['bundle_sha256'], manifest['bundle_sha256'], 'Full training bundle')
    return dict(best_step=180000, training_attempts=181144, weight_sha256=BEST_WEIGHT_SHA,
        remote_weight_proof_sha256=file_sha256(training/'remote_weight_checks.json'))


def verify_receipt(base, receipt, *, required_paths=()):
    """Accept the two existing report receipt formats, without reading weights."""
    base = Path(base).resolve()
    entries = receipt['files']
    if isinstance(entries, dict):
        entries = [dict(path=name, bytes=value['size'], sha256=value['sha256']) for name,value in entries.items()]
    verified = {}
    for item in entries:
        path = (base/item['path']).resolve()
        if not path.is_relative_to(base) or path.suffix in ('.safetensors', '.bin', '.pt', '.arrow'):
            raise ValueError(f'Unsafe/non-report receipt path: {item["path"]}')
        require(path.stat().st_size, item['bytes'], f'Receipt size {item["path"]}')
        require(file_sha256(path), item['sha256'], f'Receipt SHA {item["path"]}')
        verified[item['path']] = item['sha256']
    covered = {(base/name).resolve() for name in verified}
    if not all(Path(path).resolve() in covered for path in required_paths):
        raise ValueError('Download receipt does not cover every evaluated artifact')
    return verified


def summarize(evaluation_dir, *, baseline_dir=OLD/'stage2_recovery/evaluations/distinct4m',
              golden_path=OLD/'reference_identity.json', training_report=TRAINING,
              download_receipt=None, output=NEW/'comparison_checks.json'):
    evaluation, baseline, output = map(lambda path: Path(path).resolve(), (evaluation_dir, baseline_dir, output))
    if output.exists():
        raise FileExistsError(f'Comparison report already exists: {output}')
    if output.is_relative_to(baseline) or output.is_relative_to(Path(training_report).resolve()):
        raise ValueError('Comparison output must be separate from historical training/baseline inputs')
    receipt_path = Path(download_receipt) if download_receipt else evaluation.parent/'download_receipt.json'
    required_files = ('config.json', 'metrics.json', 'predictions.jsonl', 'progress.json',
                      'evaluation/evaluation.jsonl', 'evaluation/metrics.json', 'evaluation/evaluation_config.json')
    new_receipt = verify_receipt(receipt_path.parent, read_json(receipt_path),
        required_paths=[evaluation/name for name in required_files])
    old_receipt_path = baseline.parents[1]/'verified/distinct4m_evaluation.json'
    old_receipt = verify_receipt(baseline, read_json(old_receipt_path),
        required_paths=[baseline/name for name in required_files])
    require(file_sha256(golden_path), GOLDEN_SHA, 'Original golden reference file')
    golden = read_json(golden_path)
    require(golden['validation_sha256'], VALIDATION_SHA, 'Original validation SHA')
    require(len(golden['rows']), 20000, 'Fixed validation rows')
    require(sum(row['evaluable'] for row in golden['rows']), 19886, 'Fixed eligibility denominator')
    full_config, old_config = (read_json(path/'config.json') for path in (evaluation, baseline))
    common = dict(architecture='bart', validation_rows=20000, validation_sha256=VALIDATION_SHA,
        batch_size=64, num_beams=4, length_penalty=1., max_new_tokens=256,
        do_sample=False, use_cache=True, official_test_used=False)
    for label, config in (('Full',full_config), ('Distinct4M',old_config)):
        for key, expected in common.items():
            require(config[key], expected, f'{label} configuration {key}')
    require(full_config['bundle_sha256'], ORIGINAL_BUNDLE_SHA, 'Original evaluation bundle')
    require(full_config['reference_identity_sha256'], GOLDEN_SHA, 'Full evaluation golden proof')
    mapping = full_config['validation_mapping']
    require(mapping['verified_rows'], 20000, 'All mapped validation rows')
    require(mapping['full_validation_rows'], 118302, 'Full source validation rows')
    require(mapping['validation_sha256'], VALIDATION_SHA, 'Mapped canonical validation SHA')
    historical_proof = read_json(OLD/'stage2_data/complete.json')['provenance']
    historical = historical_proof['original_selection']
    require(mapping['indices_sha256'], historical['splits']['val']['indices_sha256'], 'Original validation row indices')
    require(mapping['token_rows_sha256'], historical_proof['original_rows_sha256']['val'], 'Original validation token-row checksum')
    require(full_config['selection_sha256'], historical_proof['original_selection_sha256'], 'Original evaluation selection')
    provenance = verify_full_provenance(full_config, training_report)
    full_metrics, full_rows = audit_evaluation(evaluation, golden['rows'])
    old_metrics, old_rows = audit_evaluation(baseline, golden['rows'])
    require(full_metrics.get('reference_identity_verified'), True, 'Completed golden gate')
    require(old_metrics['num_passed'], 13657, 'Verified distinct4M baseline passes')
    for metrics in (full_metrics,old_metrics):
        require(metrics['validation_sha256'], VALIDATION_SHA, 'Metrics validation SHA')
        require(metrics['num_evaluated'], 19886, 'Common eligible denominator')
        require(metrics['num_reference_errors'], 114, 'Common excluded references')
        for key in ('architecture', 'num_beams', 'length_penalty', 'max_new_tokens'):
            require(metrics[key], common[key], f'Metrics generation contract {key}')
    paired = paired_outcomes(full_rows,old_rows)
    report = dict(schema=1,status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        scope='Saved-artifact verification only; no model generation, prediction execution, raw-cache loading, or weights downloaded.',
        validation=dict(rows=20000,common_eligible=19886,excluded_both=114,
            validation_sha256=VALIDATION_SHA,golden_reference_sha256=GOLDEN_SHA),
        checks=dict(all_golden_reference_identities_and_row_order_match=True,
            all_prediction_execution_and_between_model_input_identities_match=True,
            all_pass_flags_statuses_counts_rates_and_denominators_recomputed=True,
            evaluator_sidecars_and_completion_match=True,
            same_generation_settings=True,original_validation_mapping_and_canonical_receipts_match=True,
            actual_best180000_weight_sha_and_training181144_provenance_match=True,
            downloaded_and_historical_file_receipts_verified=True),
        full_model=full_metrics, distinct4m=old_metrics, paired=paired, source_checkpoint=provenance,
        verified_file_sha256=dict(full_download=new_receipt,distinct4m=old_receipt,
            download_receipt=file_sha256(receipt_path)),
        limitations=['One seed (42) with different training data/budgets; this comparison does not establish universal superiority.',
            'These validation rows were used for loss monitoring and best-checkpoint selection; this is not official hidden-test accuracy.',
            'Canonical token-row identity is checked against pinned evaluator mapping receipts; local reports contain decoded text rather than the original Arrow token buffers.',
            'A100 generation timing does not establish the 180-second T4 requirement.'],findings=[])
    payload = json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        stream.write(payload)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, default=NEW/'evaluation')
    parser.add_argument('--baseline-dir', type=Path, default=OLD/'stage2_recovery/evaluations/distinct4m')
    parser.add_argument('--golden-path', type=Path, default=OLD/'reference_identity.json')
    parser.add_argument('--training-report', type=Path, default=TRAINING)
    parser.add_argument('--download-receipt', type=Path)
    parser.add_argument('--output', type=Path, default=NEW/'comparison_checks.json')
    report = summarize(**vars(parser.parse_args()))
    print(json.dumps(dict(status=report['status'],full_pass_percent=report['full_model']['pass_percent'],
        distinct4m_pass_percent=report['distinct4m']['pass_percent'],paired=report['paired']),indent=2))


if __name__ == '__main__':
    main()
