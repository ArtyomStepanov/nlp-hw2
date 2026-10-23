"""Prepare verified repetition/distinctness bundles for the sequential A100 study.

The full cache may legitimately have another fingerprint after removal of its
length column. Its selected rows, including every validation row, must therefore
match the original checksummed bundle before an output directory is created.
Only small batches of tokenized rows and integer selection arrays enter RAM.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from kaggle_experiments import file_sha256, load_bundle, write_json

TOKEN_COLUMNS = ('input_ids', 'attention_mask', 'labels')
STRATEGIES = ('repeat4m', 'distinct4m')
SEED = 42
GLOBAL_BATCH = 256


def _indices(path, *, count, source_count, unique=True):
    import numpy as np
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f'Selection must be a one-dimensional integer array: {path}')
    if values.min() < 0 or values.max() >= source_count:
        raise ValueError(f'Selection index out of range: {path}')
    if unique and len(np.unique(values)) != count:
        raise ValueError(f'Selection contains duplicate indices: {path}')
    return values


def _compare_rows(actual, expected, *, name, batch_size):
    """Compare every ordered row and hash canonical token columns, ignoring metadata."""
    if len(actual) != len(expected):
        raise ValueError(f'{name} row count mismatch')
    digest = hashlib.sha256()
    for start in range(0, len(actual), batch_size):
        stop = min(start+batch_size, len(actual))
        left, right = actual[start:stop], expected[start:stop]
        for column in TOKEN_COLUMNS:
            if left[column] != right[column]:
                for offset, (a, b) in enumerate(zip(left[column], right[column])):
                    if a != b:
                        raise ValueError(f'{name} row mismatch at {start+offset}, column {column}')
        for offset in range(stop-start):
            row = [left[column][offset] for column in TOKEN_COLUMNS]
            length = max(len(row[0]), len(row[2]))
            if 'length' in left and left['length'][offset] != length:
                raise ValueError(f'{name} length mismatch at {start+offset}')
            if 'length' in right and right['length'][offset] != length:
                raise ValueError(f'{name} length mismatch at {start+offset}')
            digest.update(json.dumps(row, separators=(',', ':')).encode('utf-8'))
            digest.update(b'\n')
    return digest.hexdigest()


def _verify_sources(bundle_root, full_cache, *, base_rows, full_rows, batch_size,
                    expected_bundle_manifest_sha):
    import datasets as ds
    original, _, identity = load_bundle(bundle_root, expected_rows=base_rows,
        expected_manifest_sha=expected_bundle_manifest_sha)
    selection = identity['selection']
    full = ds.load_from_disk(str(full_cache), keep_in_memory=False)
    selected, indices, hashes, fingerprints = {}, {}, {}, {}
    for name, count, full_count in zip(('train', 'val'), base_rows, full_rows):
        if name not in full or len(full[name]) != full_count:
            raise ValueError(f'{name} full-cache source row count mismatch; expected {full_count}')
        info = selection['splits'][name]
        if info.get('source_rows') != full_count or info.get('selected_rows') != count:
            raise ValueError(f'{name} manifest source rows or selected count mismatch')
        if not set(TOKEN_COLUMNS).issubset(full[name].column_names):
            raise ValueError(f'{name} full cache lacks tokenized columns')
        indices[name] = _indices(bundle_root/f'selection/{name}_indices.npy',
            count=count, source_count=full_count)
        selected[name] = full[name].select_columns(list(TOKEN_COLUMNS)).select(indices[name])
        hashes[name] = _compare_rows(selected[name], original[name],
            name=f'original {name}', batch_size=batch_size)
        fingerprints[name] = dict(rows=full_count, fingerprint=full[name]._fingerprint,
            original_source_fingerprint=info.get('source_fingerprint'),
            fingerprint_matches_original=full[name]._fingerprint == info.get('source_fingerprint'))
    provenance = dict(original_bundle_sha256=identity['bundle_sha256'],
        original_selection_sha256=identity['selection_sha256'],
        original_selection=selection, full_cache_path=str(full_cache), full_cache=fingerprints,
        verified_original_rows=dict(zip(('train', 'val'), base_rows)),
        original_rows_sha256=hashes, validation_rows_sha256=hashes['val'],
        row_hash_format='sha256 of ordered JSON [input_ids,attention_mask,labels] + LF')
    return full, indices, provenance


def _add_lengths(batch):
    return {'length': [max(len(source), len(target))
                       for source, target in zip(batch['input_ids'], batch['labels'])]}


def _seal_bundle(root):
    entries = [dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
                    sha256=file_sha256(path))
               for path in sorted(root.rglob('*')) if path.is_file()]
    write_json(root/'bundle_manifest.json', dict(schema=1,
        created_at=datetime.now(timezone.utc).isoformat(), files=entries,
        selection_sha256=file_sha256(root/'selection/data_manifest.json')))


def _verify_output(root, *, strategy, expected_rows, full, base_indices,
                   provenance, batch_size, expected_manifest_sha=None):
    import numpy as np
    data, _, identity = load_bundle(root, expected_rows=expected_rows,
        expected_manifest_sha=expected_manifest_sha)
    selection = identity['selection']
    if selection.get('strategy') != strategy or selection.get('provenance') != provenance:
        raise ValueError(f'{strategy} output provenance mismatch')
    # Demand a complete inventory, including dataset metadata and selection arrays.
    manifest = json.loads((root/'bundle_manifest.json').read_text())
    recorded = [entry['path'] for entry in manifest['files']]
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p != root/'bundle_manifest.json'}
    if len(recorded) != len(set(recorded)) or set(recorded) != actual:
        raise ValueError(f'{strategy} bundle integrity file inventory mismatch')
    for name, count in zip(('train', 'val'), expected_rows):
        indices = _indices(root/f'selection/{name}_indices.npy', count=count,
            source_count=len(full[name]), unique=(name == 'val' or strategy == 'distinct4m'))
        if name == 'val':
            same = np.array_equal(indices, base_indices[name])
        elif strategy == 'repeat4m':
            same = all(np.array_equal(indices[start:start+len(base_indices[name])], base_indices[name])
                       for start in range(0, count, len(base_indices[name])))
        else:
            same = np.array_equal(indices[:len(base_indices[name])], base_indices[name])
        if not same:
            raise ValueError(f'{strategy} {name} selection order mismatch')
        source = full[name].select_columns(list(TOKEN_COLUMNS)).select(indices)
        row_hash = _compare_rows(data[name], source,
            name=f'{strategy} {name}', batch_size=batch_size)
        if row_hash != selection['splits'][name]['rows_sha256']:
            raise ValueError(f'{strategy} {name} ordered row checksum mismatch')
        if name == 'val' and row_hash != provenance['validation_rows_sha256']:
            raise ValueError('Fixed validation row checksum mismatch')
    return identity['bundle_sha256']


def _completed(output, *, request, provenance, full, base_indices, batch_size):
    marker = output/'complete.json'
    if not marker.is_file():
        raise ValueError(f'Incomplete output exists; choose a new output directory: {output}')
    result = json.loads(marker.read_text())
    if result.get('status') != 'complete' or result.get('request') != request or result.get('provenance') != provenance:
        raise ValueError('Completed output does not match this request or source provenance')
    if {p.name for p in output.iterdir()} != {*STRATEGIES, 'complete.json'}:
        raise ValueError('Completed output integrity inventory mismatch')
    for strategy in STRATEGIES:
        _verify_output(output/strategy, strategy=strategy,
            expected_rows=tuple(result['expected_rows']), full=full, base_indices=base_indices,
            provenance=provenance, batch_size=batch_size,
            expected_manifest_sha=result['bundles'][strategy]['bundle_manifest_sha256'])
    return result


def prepare_diversity(bundle_root, full_cache, output_dir, *, base_rows=(1_000_000, 20_000),
                      full_rows=(11_593_060, 118_302), repeat_factor=4, batch_size=1000,
                      save_num_proc=1, expected_bundle_manifest_sha=None):
    """Create both immutable bundles, or fully verify an already completed pair.

    Count overrides support small integration checks. Production defaults are the
    exact 1M/20k selection and 11,593,060/118,302 original cache; seed is always42.
    Interrupted partial output is retained for inspection and is never resumed.
    """
    import datasets as ds
    import numpy as np
    bundle_root, full_cache, output = (Path(p).resolve() for p in (bundle_root, full_cache, output_dir))
    if len(base_rows) != 2 or len(full_rows) != 2:
        raise ValueError('Expected train/validation row count pairs')
    if any(type(value) is not int or value < 1 for value in (*base_rows, *full_rows, repeat_factor, batch_size)):
        raise ValueError('Counts, repeat factor, and batch size must be positive integers')
    if type(save_num_proc) is not int or not 1 <= save_num_proc <= 4:
        raise ValueError('save_num_proc must be between1 and4')
    total = base_rows[0]*repeat_factor
    if repeat_factor < 2 or total > full_rows[0] or base_rows[1] > full_rows[1]:
        raise ValueError('Insufficient source rows for the requested distinct selection')
    if output.is_relative_to(bundle_root) or output.is_relative_to(full_cache):
        raise ValueError('Output must be separate from the immutable input directories')
    request = dict(base_rows=list(base_rows), full_rows=list(full_rows), repeat_factor=repeat_factor,
                   seed=SEED, global_batch=GLOBAL_BATCH)
    full, base_indices, provenance = _verify_sources(bundle_root, full_cache,
        base_rows=base_rows, full_rows=full_rows, batch_size=batch_size,
        expected_bundle_manifest_sha=expected_bundle_manifest_sha)
    if output.exists():
        return _completed(output, request=request, provenance=provenance,
            full=full, base_indices=base_indices, batch_size=batch_size)
    partial, lock = (output.with_name(f'.{output.name}.{suffix}') for suffix in ('partial', 'lock'))
    if partial.exists():
        raise ValueError(f'Partial output exists; choose a new output directory: {partial}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('x') as stream:
        stream.write(json.dumps(dict(output=str(output), request=request)))
    try:
        partial.mkdir()
        work = partial/'._work'
        work.mkdir()
        # Integer metadata is small; tokenized Arrow data stays memory mapped.
        available = np.ones(full_rows[0], dtype=bool)
        available[base_indices['train']] = False
        remaining = np.flatnonzero(available)
        extra = np.random.default_rng(SEED).choice(remaining, size=total-base_rows[0], replace=False)
        del available, remaining
        train_indices = {
            'repeat4m': np.tile(base_indices['train'], repeat_factor),
            'distinct4m': np.concatenate((base_indices['train'], extra))}
        del extra
        bundles = {}
        for strategy in STRATEGIES:
            root = partial/strategy
            (root/'selection').mkdir(parents=True)
            shutil.copyfile(bundle_root/'tokenizer.json', root/'tokenizer.json')
            selected, details = {}, {}
            for name, values in (('train', train_indices[strategy]), ('val', base_indices['val'])):
                indices_path = root/f'selection/{name}_indices.npy'
                np.save(indices_path, values, allow_pickle=False)
                view = full[name].select_columns(list(TOKEN_COLUMNS)).select(values)
                selected[name] = view.map(_add_lengths, batched=True,
                    batch_size=batch_size, writer_batch_size=batch_size, num_proc=None,
                    load_from_cache_file=False, keep_in_memory=False,
                    cache_file_name=str(work/f'{strategy}-{name}.arrow'),
                    desc=f'{strategy}/{name}: add length')
                details[name] = dict(source_rows=len(full[name]), selected_rows=len(values),
                    source_fingerprint=full[name]._fingerprint,
                    indices_sha256=file_sha256(indices_path), seed=SEED if name == 'train' else SEED+1)
            ds.DatasetDict(selected).save_to_disk(str(root/'data'),
                max_shard_size='500MB', num_proc=None if save_num_proc == 1 else save_num_proc)
            # Read back all saved rows before publishing the completion marker.
            saved = ds.load_from_disk(str(root/'data'), keep_in_memory=False)
            for name, values in (('train', train_indices[strategy]), ('val', base_indices['val'])):
                details[name]['rows_sha256'] = _compare_rows(saved[name],
                    full[name].select_columns(list(TOKEN_COLUMNS)).select(values),
                    name=f'{strategy} {name}', batch_size=batch_size)
            identity = dict(provenance['original_selection']['identity'])
            identity.update(train_rows=total, validation_rows=base_rows[1], seed=SEED)
            write_json(root/'selection/data_manifest.json', dict(identity=identity,
                splits=details, strategy=strategy, provenance=provenance,
                created_at=datetime.now(timezone.utc).isoformat()))
            _seal_bundle(root)
            sha = _verify_output(root, strategy=strategy, expected_rows=(total, base_rows[1]),
                full=full, base_indices=base_indices, provenance=provenance, batch_size=batch_size)
            bundles[strategy] = dict(path=str(output/strategy), bundle_manifest_sha256=sha)
        shutil.rmtree(work)
        result = dict(schema=1, status='complete', request=request, provenance=provenance,
            expected_rows=[total, base_rows[1]], optimizer_updates_per_epoch=math.ceil(total/GLOBAL_BATCH),
            epochs=1, bundles=bundles, completed_at=datetime.now(timezone.utc).isoformat())
        write_json(partial/'complete.json', result)
        partial.rename(output)
        return result
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-root', type=Path, required=True)
    parser.add_argument('--full-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-bundle-manifest-sha')
    parser.add_argument('--base-train-rows', type=int, default=1_000_000)
    parser.add_argument('--validation-rows', type=int, default=20_000)
    parser.add_argument('--full-train-rows', type=int, default=11_593_060)
    parser.add_argument('--full-validation-rows', type=int, default=118_302)
    parser.add_argument('--repeat-factor', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--save-num-proc', type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    result = prepare_diversity(args.bundle_root, args.full_cache, args.output_dir,
        base_rows=(args.base_train_rows, args.validation_rows),
        full_rows=(args.full_train_rows, args.full_validation_rows),
        repeat_factor=args.repeat_factor, batch_size=args.batch_size,
        save_num_proc=args.save_num_proc, expected_bundle_manifest_sha=args.expected_bundle_manifest_sha)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
