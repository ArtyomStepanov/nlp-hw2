"""Sample verified validation failures and original training rows for local inspection.

This reads saved evaluator results; it does not execute code, generate new targets,
filter a training dataset, or launch training. The CLI requires the original
1,000,000/20,000 bundle, including its tokenizer and selection manifests.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys


ORIGINAL_BUNDLE_SHA256 = '7480c7d1778bffce50db987019f2ccf87c06f8643c1ce59ad811e4fd3c77d0e1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _read_rows(path):
    with path.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f'Expected JSON objects in {path}')
    return rows


def _diagnostics(code, prediction=None):
    result = dict(syntax_valid=True, has_top_level_solve_definition=None)
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as error:
        result.update(syntax_valid=False, syntax_error=dict(message=str(error),
            line=getattr(error, 'lineno', None), offset=getattr(error, 'offset', None)))
    else:
        result['has_top_level_solve_definition'] = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'solve'
            for node in tree.body)
    if prediction is not None:
        for key in ('hit_token_limit', 'ended_with_eos', 'generated_tokens'):
            result[key] = prediction.get(key)
        result['saved_execution_status'] = prediction.get('execution_status')
        result['saved_evaluation_status'] = prediction['status']
    return result


def audit_errors(bundle_root, evaluation_dir, output_dir, *, seed=42, sample_size=200,
                 expected_rows=(1000000, 20000), expected_manifest_sha=ORIGINAL_BUNDLE_SHA256):
    """Verify validation and original-bundle identity before sampling.

    expected_manifest_sha is an API-only override for small test fixtures. The CLI
    always pins the known original 1M/20k bundle, independently of the checkpoint's
    potentially different Stage 2 training-bundle identity.
    """
    if type(seed) is not int or type(sample_size) is not int or sample_size < 1:
        raise ValueError('seed must be an integer and sample_size must be positive.')
    bundle_root, evaluation_dir, output_dir = map(
        lambda path: Path(path).resolve(), (bundle_root, evaluation_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite audit output: {output_dir}')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from kaggle_experiments import file_sha256, load_bundle, write_json

    data, tokenizer, identity = load_bundle(bundle_root, expected_rows=expected_rows,
        expected_manifest_sha=expected_manifest_sha)
    paths = dict(config=evaluation_dir/'config.json', metrics=evaluation_dir/'metrics.json',
        predictions=evaluation_dir/'predictions.jsonl',
        evaluation=evaluation_dir/'evaluation/evaluation.jsonl')
    configuration = json.loads(paths['config'].read_text())
    metrics = json.loads(paths['metrics'].read_text())
    predictions, results = (_read_rows(paths[key]) for key in ('predictions', 'evaluation'))
    count = len(data['val'])
    if configuration.get('validation_rows') != count or len(predictions) != count or len(results) != count:
        raise ValueError('Audit requires predictions and evaluation for the complete validation split.')
    indices = {}
    for split in ('train', 'val'):
        values = np.load(bundle_root/f'selection/{split}_indices.npy', allow_pickle=False)
        source_rows = identity['selection']['splits'][split].get('source_rows')
        if (values.shape != (len(data[split]),) or values.dtype.kind not in 'iu' or
                len(np.unique(values)) != len(values) or np.any(values < 0) or
                (source_rows is not None and np.any(values >= source_rows))):
            raise ValueError(f'Original {split} selection must have unique valid source row IDs.')
        indices[split] = values

    validation_digest, failed = hashlib.sha256(), []
    eligible_count = passed_count = 0
    for position, (original, prediction, result) in enumerate(zip(data['val'], predictions, results)):
        validation_digest.update((_canonical(original)+'\n').encode('utf-8'))
        for row in (prediction, result):
            if type(row.get('sample_row')) is not int or row['sample_row'] != position:
                raise ValueError(f'Validation row order/sample_row mismatch at {position}.')
        for field, column in (('question', 'input_ids'), ('code', 'labels')):
            expected = tokenizer.decode(original[column], skip_special_tokens=True)
            if prediction.get(field) != expected or result.get(field) != expected:
                raise ValueError(f'Original validation {field} mismatch at row {position}.')
        if not isinstance(prediction.get('generated_code'), str):
            raise ValueError(f'Missing generated_code at row {position}.')
        if any(result.get(key) != value for key, value in prediction.items()):
            raise ValueError(f'Prediction/evaluation row mismatch at {position}.')
        if not isinstance(result.get('status'), str) or not isinstance(result.get('reference_status'), str):
            raise ValueError(f'Missing evaluator status at row {position}.')
        eligible = result['reference_status'] == 'ok'
        if type(result.get('evaluable')) is not bool or result['evaluable'] != eligible:
            raise ValueError(f'Inconsistent reference eligibility at row {position}.')
        if eligible:
            if type(result.get('passed')) is not bool:
                raise ValueError(f'Missing boolean pass result at eligible row {position}.')
            if not isinstance(result.get('reference_result_json'), str):
                raise ValueError(f'Missing reference result at eligible row {position}.')
            json.loads(result['reference_result_json'])
            if (result['status'] == 'passed') != result['passed']:
                raise ValueError(f'Inconsistent pass status at row {position}.')
            eligible_count += 1
            passed_count += result['passed']
            if result['passed'] is False:
                failed.append(position)
        elif result.get('passed') is not None:
            raise ValueError(f'Noneligible reference has a pass/fail label at row {position}.')

    validation_sha = validation_digest.hexdigest()
    for label, source in (('config', configuration), ('metrics', metrics)):
        if source.get('validation_sha256') != validation_sha:
            raise ValueError(f'{label} validation_sha256 differs from the complete original validation rows.')
    for key, expected in (('num_examples', count), ('num_evaluated', eligible_count),
                          ('num_passed', passed_count)):
        if metrics.get(key) != expected:
            raise ValueError(f'Saved metric {key} differs from the evaluator rows.')
    if len(failed) < sample_size or len(data['train']) < sample_size:
        raise ValueError(f'Need {sample_size} eligible failures and original train rows; '
                         f'found {len(failed)} and {len(data["train"])}.')

    selected_validation = sorted(random.Random(seed).sample(failed, sample_size))
    selected_training = sorted(random.Random(seed).sample(range(len(data['train'])), sample_size))

    def provenance(original, question, code):
        return dict(bundle_sha256=identity['bundle_sha256'],
            selection_sha256=identity['selection_sha256'],
            bundle_row_sha256=_sha(_canonical(original)),
            input_ids_sha256=_sha(_canonical(original['input_ids'])),
            labels_sha256=_sha(_canonical(original['labels'])),
            question_sha256=_sha(question), code_sha256=_sha(code))

    errors, train = [], []
    for position in selected_validation:
        row = dict(results[position])
        row.update(source_split='val', source_row=int(indices['val'][position]),
            provenance=provenance(data['val'][position], row['question'], row['code']),
            diagnostics=_diagnostics(row['generated_code'], row))
        row['provenance'].update(prediction_row_sha256=_sha(_canonical(predictions[position])),
            evaluation_row_sha256=_sha(_canonical(results[position])))
        errors.append(row)
    for position in selected_training:
        original = data['train'][position]
        question = tokenizer.decode(original['input_ids'], skip_special_tokens=True)
        code = tokenizer.decode(original['labels'], skip_special_tokens=True)
        train.append(dict(sample_row=position, source_split='train',
            source_row=int(indices['train'][position]), question=question, code=code,
            provenance=provenance(original, question, code),
            diagnostics=_diagnostics(code)))

    report = dict(schema=1, purpose='Inspection samples only; no new or corrected targets.',
        sampling=dict(seed=seed, sample_size=sample_size,
            algorithm='Python random.Random(seed).sample, independently per split; sorted by sample_row'),
        provenance=dict(original_bundle_root=str(bundle_root), evaluation_dir=str(evaluation_dir),
            original_bundle_sha256=identity['bundle_sha256'],
            original_selection_sha256=identity['selection_sha256'],
            tokenizer_sha256=file_sha256(bundle_root/'tokenizer.json'),
            original_source=identity['selection']['identity'].get('source'),
            evaluation_bundle_sha256=configuration.get('bundle_sha256'),
            evaluation_selection_sha256=configuration.get('selection_sha256'),
            checkpoint=configuration.get('checkpoint'), architecture=configuration.get('architecture'),
            validation_sha256=validation_sha,
            input_files={key: dict(path=str(path), sha256=file_sha256(path)) for key, path in paths.items()}),
        counts=dict(train_rows=len(data['train']), validation_rows=count,
            eligible=eligible_count, eligible_passed=passed_count, eligible_failed=len(failed),
            excluded_reference_errors=count-eligible_count, selected_errors=len(errors),
            train_sample=len(train),
            all_statuses=dict(Counter(row['status'] for row in results)),
            reference_statuses=dict(Counter(row['reference_status'] for row in results)),
            eligible_failure_statuses=dict(Counter(results[i]['status'] for i in failed)),
            sampled_failure_statuses=dict(Counter(row['status'] for row in errors)),
            sampled_syntax_errors=sum(row['diagnostics']['syntax_valid'] is False for row in errors),
            sampled_no_top_level_solve=sum(row['diagnostics']['has_top_level_solve_definition'] is False
                                          for row in errors),
            sampled_hit_token_limit=sum(row['diagnostics']['hit_token_limit'] is True for row in errors)),
        limitations=['Executable references are eligible for the existing comparison, not proven semantically correct.',
            'AST checks describe syntax and top-level definitions; saved execution results are not rerun.',
            'Token-limit flags are saved generation diagnostics, not proof that an answer was truncated.'])
    output_dir.mkdir(parents=True)
    for filename, rows in (('selected_errors.jsonl', errors), ('train_sample.jsonl', train)):
        with (output_dir/filename).open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    report['output_files'] = {name: file_sha256(output_dir/name)
        for name in ('selected_errors.jsonl', 'train_sample.jsonl')}
    write_json(output_dir/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-root', required=True)
    parser.add_argument('--evaluation-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-size', type=int, default=200)
    args = parser.parse_args()
    report = audit_errors(**vars(args))
    print(json.dumps(report['counts'], ensure_ascii=False, indent=2))
