"""Small real Arrow/evaluator fixtures for the local error audit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import datasets as ds
import numpy as np
import pytest

from evaluation import EvaluationConfig, evaluate_code
from kaggle_experiments import load_bundle
from tokenizer import train_bpe_tokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.write_text(json.dumps(value))


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def put_rows(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


@pytest.fixture
def sources(tmp_path):
    root, evaluation = tmp_path/'original', tmp_path/'evaluation'
    (root/'selection').mkdir(parents=True)
    evaluation.mkdir()
    gold = ['def solve():\n    return 1\n', 'def solve():\n    return 2\n',
            'def solve():\n    return 3\n', 'x = 4\n', 'def solve(:\n    return 5\n',
            'def solve():\n    return 6\n']
    questions = [f'What is the number {i+1}?' for i in range(6)]
    raw = ds.Dataset.from_dict(dict(question=questions, code=gold))
    tokenizer = train_bpe_tokenizer(raw, vocab_size=320, min_frequency=1)
    tokenizer.save(str(root/'tokenizer.json'))
    def encode(i):
        source = tokenizer.encode(questions[i % 6]).ids
        target = tokenizer.encode(gold[i % 6]).ids[1:]
        return dict(input_ids=source, attention_mask=[1]*len(source),
                    labels=target, length=max(len(source), len(target)))
    data = ds.DatasetDict(train=ds.Dataset.from_list([encode(i) for i in range(8)]),
                          val=ds.Dataset.from_list([encode(i) for i in range(6)]))
    data.save_to_disk(str(root/'data'))
    details = {}
    for split, indices in (('train', [15, 10, 44, 12, 13, 14, 11, 18]),
                            ('val', [101, 109, 103, 105, 107, 111])):
        path = root/f'selection/{split}_indices.npy'
        np.save(path, indices)
        details[split] = dict(selected_rows=len(indices), source_rows=200,
                              indices_sha256=digest(path))
    selection = root/'selection/data_manifest.json'
    put(selection, dict(identity=dict(train_rows=8, validation_rows=6, seed=42,
        source='/original/full/cache', tokenizer_sha256=digest(root/'tokenizer.json')),
        splits=details))
    files = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size,
                  sha256=digest(p)) for p in sorted(root.rglob('*')) if p.is_file()]
    put(root/'bundle_manifest.json', dict(schema=1, files=files,
                                          selection_sha256=digest(selection)))
    validation_sha = hashlib.sha256(''.join(json.dumps(row, sort_keys=True,
        separators=(',', ':'))+'\n' for row in data['val']).encode()).hexdigest()
    generated = [gold[0], 'def solve():\n    return 99\n', 'def solve(:\n',
                 gold[0], gold[0], 'x = 6\n']
    predictions = [dict(sample_row=i, question=tokenizer.decode(row['input_ids'],
        skip_special_tokens=True), code=tokenizer.decode(row['labels'], skip_special_tokens=True),
        generated_code=generated[i], generated_tokens=256 if i == 2 else 12,
        ended_with_eos=i != 2, hit_token_limit=i == 2) for i, row in enumerate(data['val'])]
    put_rows(evaluation/'predictions.jsonl', predictions)
    result = evaluate_code(ds.Dataset.from_list(predictions), EvaluationConfig(
        workers=1, output_dir=evaluation/'evaluation'))
    # A Stage 2 training bundle may differ; its complete validation must still match.
    put(evaluation/'config.json', dict(validation_rows=6, validation_sha256=validation_sha,
        bundle_sha256='f'*64, selection_sha256='e'*64, max_new_tokens=256,
        checkpoint='/saved/best/final', architecture='bart'))
    put(evaluation/'metrics.json', dict(result.metrics, validation_sha256=validation_sha))
    return root, evaluation


def audit(sources, output, **kwargs):
    assert importlib.util.find_spec('tools.audit_a100_errors'), 'Local audit tool is missing'
    from tools.audit_a100_errors import audit_errors
    return audit_errors(*sources, output, expected_rows=(8, 6),
                        expected_manifest_sha=kwargs.pop('expected_manifest_sha',
                            digest(sources[0]/'bundle_manifest.json')),
                        sample_size=kwargs.pop('sample_size', 2), **kwargs)


def test_sampling_is_deterministic_and_excludes_noneligible_references(sources, tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    report = audit(sources, a)
    assert audit(sources, b) == report
    for filename in ('selected_errors.jsonl', 'train_sample.jsonl', 'report.json'):
        assert (a/filename).read_bytes() == (b/filename).read_bytes()
    changed_seed = tmp_path/'changed_seed'
    audit(sources, changed_seed, seed=43)
    assert [row['sample_row'] for row in jsonl(a/'train_sample.jsonl')] != [
        row['sample_row'] for row in jsonl(changed_seed/'train_sample.jsonl')]
    errors, train = jsonl(a/'selected_errors.jsonl'), jsonl(a/'train_sample.jsonl')
    assert len(errors) == len(train) == 2
    assert {row['sample_row'] for row in errors} <= {1, 2, 5}
    assert all(row['passed'] is False and row['evaluable'] is True and
               row['reference_status'] == 'ok' for row in errors)
    assert report['counts']['validation_rows'] == 6
    assert report['counts']['eligible_failed'] == 3
    assert report['counts']['excluded_reference_errors'] == 2
    data, tokenizer, identity = load_bundle(sources[0], expected_rows=(8, 6))
    for split, selected in (('val', errors), ('train', train)):
        indices = np.load(sources[0]/f'selection/{split}_indices.npy')
        for row in selected:
            original = data[split][row['sample_row']]
            assert row['source_row'] == int(indices[row['sample_row']])
            assert row['source_split'] == split
            assert row['question'] == tokenizer.decode(original['input_ids'], skip_special_tokens=True)
            assert row['code'] == tokenizer.decode(original['labels'], skip_special_tokens=True)
            assert row['provenance']['bundle_sha256'] == identity['bundle_sha256']
            assert row['provenance']['code_sha256'] == hashlib.sha256(row['code'].encode()).hexdigest()
    all_errors = tmp_path/'all'
    audit(sources, all_errors, sample_size=3)
    rows = {row['sample_row']: row for row in jsonl(all_errors/'selected_errors.jsonl')}
    assert rows[2]['diagnostics']['syntax_valid'] is False
    assert rows[2]['diagnostics']['hit_token_limit'] is True
    assert rows[5]['diagnostics']['has_top_level_solve_definition'] is False


@pytest.mark.parametrize('change', ['question', 'code', 'order', 'evaluation_only',
    'config_hash', 'metrics_hash', 'missing_row', 'eligibility', 'reference_result'])
def test_rejects_mismatched_rows_or_complete_validation_identity(sources, tmp_path, change):
    evaluation = sources[1]
    predictions = jsonl(evaluation/'predictions.jsonl')
    results = jsonl(evaluation/'evaluation/evaluation.jsonl')
    if change in ('question', 'code'):
        # An ineligible reference cannot be selected, but must still be verified.
        predictions[4][change] = results[4][change] = 'changed original content'
    elif change == 'order':
        predictions[0], predictions[1] = predictions[1], predictions[0]
        results[0], results[1] = results[1], results[0]
    elif change == 'evaluation_only':
        results[-1]['generated_code'] = 'changed prediction'
    elif change.endswith('_hash'):
        path = evaluation/('config.json' if change == 'config_hash' else 'metrics.json')
        value = json.loads(path.read_text())
        value['validation_sha256'] = '0'*64
        put(path, value)
    elif change == 'missing_row':
        predictions.pop()
        results.pop()
    elif change == 'eligibility':
        results[3].update(evaluable=True, passed=False)
    else:
        results[1]['reference_result_json'] = None
    put_rows(evaluation/'predictions.jsonl', predictions)
    put_rows(evaluation/'evaluation/evaluation.jsonl', results)
    output = tmp_path/'rejected'
    with pytest.raises(ValueError):
        audit(sources, output)
    assert not output.exists()


def test_requires_exact_sample_size_and_fresh_output(sources, tmp_path):
    with pytest.raises(ValueError, match='eligible|sample'):
        audit(sources, tmp_path/'too_many', sample_size=4)
    occupied = tmp_path/'occupied'
    occupied.mkdir()
    (occupied/'keep').write_text('preserved')
    with pytest.raises(FileExistsError):
        audit(sources, occupied)
    assert (occupied/'keep').read_text() == 'preserved'


def test_production_default_rejects_an_unpinned_bundle(sources, tmp_path):
    from tools.audit_a100_errors import audit_errors
    output = tmp_path/'production-default'
    with pytest.raises(ValueError, match='Bundle manifest checksum'):
        audit_errors(*sources, output, expected_rows=(8, 6), sample_size=2)
    assert not output.exists()


def test_repacked_changed_train_is_rejected_despite_unchanged_validation(sources, tmp_path):
    root = sources[0]
    original_sha = digest(root/'bundle_manifest.json')
    original, _, _ = load_bundle(root, expected_rows=(8, 6))
    validation = original['val'].to_list()
    train = original['train'].to_list()
    train[0] = train[1]
    replacement = tmp_path/'repacked-data'
    ds.DatasetDict(train=ds.Dataset.from_list(train), val=original['val']).save_to_disk(str(replacement))
    shutil.rmtree(root/'data')
    replacement.rename(root/'data')
    files = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size,
                  sha256=digest(p)) for p in sorted(root.rglob('*'))
             if p.is_file() and p.name != 'bundle_manifest.json']
    put(root/'bundle_manifest.json', dict(schema=1, files=files,
        selection_sha256=digest(root/'selection/data_manifest.json')))
    # Internal checksums and complete validation still pass the unpinned loader.
    repacked, _, _ = load_bundle(root, expected_rows=(8, 6))
    assert repacked['val'].to_list() == validation
    assert repacked['train'][0] != original['train'][0]
    output = tmp_path/'repacked-audit'
    with pytest.raises(ValueError, match='Bundle manifest checksum'):
        audit(sources, output, expected_manifest_sha=original_sha)
    assert not output.exists()
