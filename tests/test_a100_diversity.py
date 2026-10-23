"""Real Arrow fixtures for the Stage 2 repetition/diversity identity gate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import datasets as ds
import numpy as np
import pytest
from tokenizers import Tokenizer, models

from kaggle_experiments import load_bundle


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal_bundle(root):
    files = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size,
                  sha256=digest(p)) for p in sorted(root.rglob('*'))
             if p.is_file() and p.name != 'bundle_manifest.json']
    (root/'bundle_manifest.json').write_text(json.dumps(dict(schema=1, files=files,
        selection_sha256=digest(root/'selection/data_manifest.json'))))


def make_sources(tmp_path):
    root, cache = tmp_path/'original', tmp_path/'full-cache'
    (root/'selection').mkdir(parents=True)
    tokenizer = Tokenizer(models.WordLevel({'<unk>': 0, '<bos>': 1, '<eos>': 2,
        '<pad>': 3, **{str(i): i+4 for i in range(32)}}))
    tokenizer.save(str(root/'tokenizer.json'))

    def rows(offset, count):
        return ds.Dataset.from_list([dict(input_ids=[1, offset+i+4, 2],
            attention_mask=[1, 1, 1], labels=[offset+i+4, 2]) for i in range(count)])

    full = ds.DatasetDict(train=rows(0, 12), val=rows(20, 6))
    full.save_to_disk(str(cache))
    indices = {'train': [7, 1, 9, 3], 'val': [4, 1]}
    selected, details = {}, {}
    for name, values in indices.items():
        path = root/f'selection/{name}_indices.npy'
        np.save(path, values)
        selected[name] = full[name].select(values).map(lambda row: {'length': 3})
        details[name] = dict(source_rows=len(full[name]), selected_rows=len(values),
            source_fingerprint='original-with-length-fingerprint',
            indices_sha256=digest(path), seed=42 if name == 'train' else 43)
    ds.DatasetDict(selected).save_to_disk(str(root/'data'))
    (root/'selection/data_manifest.json').write_text(json.dumps(dict(identity=dict(
        train_rows=4, validation_rows=2, seed=42, source='/original/full/cache',
        tokenizer_sha256=digest(root/'tokenizer.json')), splits=details)))
    seal_bundle(root)
    return root, cache


@pytest.fixture
def sources(tmp_path):
    return make_sources(tmp_path)


def api():
    assert importlib.util.find_spec('tools.prepare_a100_diversity'), 'Stage 2 preparer is missing'
    from tools.prepare_a100_diversity import prepare_diversity
    return prepare_diversity


def prepare(sources, output, **kwargs):
    return api()(*sources, output, base_rows=(4, 2), full_rows=(12, 6),
                 repeat_factor=kwargs.pop('repeat_factor', 2),
                 batch_size=kwargs.pop('batch_size', 2), **kwargs)


@pytest.mark.parametrize('factor', [2, 3])
def test_prepares_loadable_repetition_and_distinct_bundles(sources, tmp_path, factor):
    """Wrong split, lost base rows, changed validation, or missing manifests must fail."""
    output = tmp_path/'prepared'
    result = prepare(sources, output, repeat_factor=factor)
    original, _, original_identity = load_bundle(sources[0], expected_rows=(4, 2))
    full = ds.load_from_disk(str(sources[1]))
    validation_hashes = []
    for strategy in ('repeat4m', 'distinct4m'):
        root = output/strategy
        data, _, identity = load_bundle(root, expected_rows=(4*factor, 2))
        assert data['val'].to_dict() == original['val'].to_dict()
        assert digest(root/'tokenizer.json') == digest(sources[0]/'tokenizer.json')
        assert data['train'].select(range(4)).to_dict() == original['train'].to_dict()
        selected = np.load(root/'selection/train_indices.npy').tolist()
        assert selected[:4] == [7, 1, 9, 3]
        if strategy == 'repeat4m':
            assert selected == [7, 1, 9, 3]*factor
            assert data['train'].to_dict() == {
                key: values*factor for key, values in original['train'].to_dict().items()}
        else:
            assert len(set(selected)) == 4*factor
            assert all(0 <= value < 12 for value in selected)
            for position, index in enumerate(selected):
                for column in ('input_ids', 'attention_mask', 'labels'):
                    assert data['train'][position][column] == full['train'][index][column]
        assert np.load(root/'selection/val_indices.npy').tolist() == [4, 1]
        selection = identity['selection']
        validation_hashes.append(selection['provenance']['validation_rows_sha256'])
        assert selection['provenance']['original_bundle_sha256'] == original_identity['bundle_sha256']
        assert selection['provenance']['full_cache']['train']['fingerprint'] == full['train']._fingerprint
        assert selection['provenance']['verified_original_rows'] == {'train': 4, 'val': 2}
        manifest = json.loads((root/'bundle_manifest.json').read_text())
        assert all(digest(root/e['path']) == e['sha256'] and
                   (root/e['path']).stat().st_size == e['bytes'] for e in manifest['files'])
    assert len(set(validation_hashes)) == 1
    completion = json.loads((output/'complete.json').read_text())
    assert completion['status'] == 'complete'
    assert result['expected_rows'] == [4*factor, 2]
    assert result['optimizer_updates_per_epoch'] == 1


def change_cache(cache, transform):
    data = ds.load_from_disk(str(cache))
    changed = transform(data)
    temporary = cache.with_name('changed-cache')
    changed.save_to_disk(str(temporary))
    shutil.rmtree(cache)
    temporary.rename(cache)


@pytest.mark.parametrize('split,column', [('train', 'input_ids'), ('train', 'attention_mask'),
    ('train', 'labels'), ('val', 'input_ids'), ('val', 'labels')])
def test_rejects_any_selected_row_mismatch_before_output(sources, tmp_path, split, column):
    """Skipping a column, validation, or the last comparison batch is a bug."""
    index = 3 if split == 'train' else 1
    def transform(data):
        rows = data[split].to_list()
        rows[index][column][0] += 1
        data[split] = ds.Dataset.from_list(rows)
        return data
    change_cache(sources[1], transform)
    output = tmp_path/'prepared'
    with pytest.raises(ValueError, match='row mismatch'):
        prepare(sources, output)
    assert not output.exists()
    assert not output.with_name('.prepared.partial').exists()


def test_rejects_reordered_full_cache(sources, tmp_path):
    def transform(data):
        data['train'] = data['train'].select([1, 0, *range(2, 12)])
        return data
    change_cache(sources[1], transform)
    with pytest.raises(ValueError, match='row mismatch'):
        prepare(sources, tmp_path/'prepared')


@pytest.mark.parametrize('split,values,message', [
    ('train', [7, 1, 9, 7], 'duplicate'), ('val', [4, 4], 'duplicate'),
    ('train', [7, 1, 9, 12], 'range'), ('val', [4, -1], 'range'),
    ('train', [7., 1., 9., 3.], 'integer')])
def test_rejects_invalid_original_selection(sources, tmp_path, split, values, message):
    root = sources[0]
    path = root/f'selection/{split}_indices.npy'
    np.save(path, values)
    manifest = json.loads((root/'selection/data_manifest.json').read_text())
    manifest['splits'][split]['indices_sha256'] = digest(path)
    (root/'selection/data_manifest.json').write_text(json.dumps(manifest))
    seal_bundle(root)
    with pytest.raises(ValueError, match=message):
        prepare(sources, tmp_path/'prepared')
    assert not (tmp_path/'prepared').exists()


def test_rejects_tokenizer_sha_mismatch(sources, tmp_path):
    with (sources[0]/'tokenizer.json').open('a') as stream:
        stream.write(' ')
    # Even a new outer checksum cannot change the tokenizer identity contract.
    seal_bundle(sources[0])
    with pytest.raises(ValueError, match='Tokenizer checksum'):
        prepare(sources, tmp_path/'prepared')


@pytest.mark.parametrize('target', ['actual', 'manifest'])
def test_rejects_full_cache_count_mismatch(sources, tmp_path, target):
    if target == 'actual':
        def transform(data):
            data['train'] = data['train'].select(range(11))
            return data
        change_cache(sources[1], transform)
    else:
        root = sources[0]
        manifest = json.loads((root/'selection/data_manifest.json').read_text())
        manifest['splits']['train']['source_rows'] = 11
        (root/'selection/data_manifest.json').write_text(json.dumps(manifest))
        seal_bundle(root)
    with pytest.raises(ValueError, match='source.*rows|count'):
        prepare(sources, tmp_path/'prepared')


def test_completed_resume_verifies_bytes_without_rewriting(sources, tmp_path):
    output = tmp_path/'prepared'
    first = prepare(sources, output)
    snapshot = {p.relative_to(output): (p.stat().st_mtime_ns, digest(p))
                for p in output.rglob('*') if p.is_file()}
    assert prepare(sources, output) == first
    assert {p.relative_to(output): (p.stat().st_mtime_ns, digest(p))
            for p in output.rglob('*') if p.is_file()} == snapshot
    path = next((output/'distinct4m/data/train').glob('*.arrow'))
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum|integrity'):
        prepare(sources, output)


@pytest.mark.parametrize('state', ['output', 'partial', 'locked'])
def test_rejects_incomplete_or_active_output_without_overwrite(sources, tmp_path, state):
    output = tmp_path/'prepared'
    occupied = output if state == 'output' else tmp_path/f'.prepared.{"partial" if state == "partial" else "lock"}'
    if state == 'locked':
        occupied.write_text('active')
        preserved = occupied
    else:
        occupied.mkdir()
        preserved = occupied/'keep'
        preserved.write_text('unfinished')
    with pytest.raises((ValueError, FileExistsError), match='[Ii]ncomplete|[Pp]artial|[Ll]ock|exists'):
        prepare(sources, output)
    assert preserved.read_text() == ('active' if state == 'locked' else 'unfinished')


def test_selection_is_deterministic_across_batch_sizes(sources, tmp_path):
    prepare(sources, tmp_path/'first', batch_size=1)
    prepare(sources, tmp_path/'second', batch_size=3)
    for strategy in ('repeat4m', 'distinct4m'):
        for name in ('train', 'val'):
            first = tmp_path/f'first/{strategy}/selection/{name}_indices.npy'
            second = tmp_path/f'second/{strategy}/selection/{name}_indices.npy'
            assert first.read_bytes() == second.read_bytes()
        a = json.loads((tmp_path/f'first/{strategy}/selection/data_manifest.json').read_text())
        b = json.loads((tmp_path/f'second/{strategy}/selection/data_manifest.json').read_text())
        assert a['provenance']['validation_rows_sha256'] == b['provenance']['validation_rows_sha256']


def test_cli_prepares_only_tiny_requested_counts(sources, tmp_path):
    result = subprocess.run([sys.executable, 'tools/prepare_a100_diversity.py',
        '--bundle-root', str(sources[0]), '--full-cache', str(sources[1]),
        '--output-dir', str(tmp_path/'cli'), '--base-train-rows', '4',
        '--validation-rows', '2', '--full-train-rows', '12',
        '--full-validation-rows', '6', '--repeat-factor', '2', '--batch-size', '2'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['expected_rows'] == [8, 2]
