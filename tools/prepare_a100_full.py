"""Seal the trusted full token cache as a new hardlinked train/validation bundle."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kaggle_experiments import file_sha256, write_json

FULL_ROWS = (11_593_060, 118_302)
BASE_ROWS = (1_000_000, 20_000)
FINGERPRINTS = dict(train='db4a9cf00f162409', val='593c37f2d44384f8')
ORIGINAL_MANIFEST_SHA = '7480c7d1778bffce50db987019f2ccf87c06f8643c1ce59ad811e4fd3c77d0e1'
TOKENIZER_SHA = 'abd75c8655b948f612ebbf065a12269613269a4755cd0a698b0bf7a5bc1e9704'
COLUMNS = ('input_ids', 'attention_mask', 'labels')


def _read(path):
    return json.loads(Path(path).read_text())


def _require(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label} mismatch: expected {expected!r}, got {actual!r}')


def _source_files(cache, name):
    split = cache/name
    state = _read(split/'state.json')
    if state.get('_indices_data_files'):
        raise ValueError(f'{name} source is an indexed view')
    paths = [split/'state.json', split/'dataset_info.json']
    for entry in state['_data_files']:
        relative = Path(entry['filename'])
        path = split/relative
        if relative.is_absolute() or not path.resolve().is_relative_to(split.resolve()):
            raise ValueError(f'Unsafe {name} Arrow source path: {relative}')
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError(f'Duplicate {name} source files')
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing or symbolic source file: {path}')
    return paths


def _verify_tokens(data, tokenizer, name):
    """Native Arrow checks cover all lengths; only representative rows enter Python."""
    import pyarrow as pa
    import pyarrow.compute as pc
    if data._indices is not None:
        raise ValueError(f'{name} source has non-null indices; flattening is forbidden')
    if not set(COLUMNS).issubset(data.column_names):
        raise ValueError(f'{name} source lacks token columns')
    lengths = {}
    for column in COLUMNS:
        array = data.data.column(column)
        dtype = array.type
        if not ((pa.types.is_list(dtype) or pa.types.is_large_list(dtype))
                and pa.types.is_integer(dtype.value_type)):
            raise ValueError(f'{name}/{column} must contain integer token lists')
        if array.null_count:
            raise ValueError(f'{name}/{column} contains null rows')
        sizes = pc.list_value_length(array)
        if pc.min(sizes).as_py() < 1 or pc.max(sizes).as_py() > 256:
            raise ValueError(f'{name}/{column} token length must be between 1 and 256')
        lengths[column] = sizes
    if not pc.all(pc.equal(lengths['input_ids'], lengths['attention_mask'])).as_py():
        raise ValueError(f'{name} attention mask length mismatch')
    eos, bos = tokenizer.token_to_id('<eos>'), tokenizer.token_to_id('<bos>')
    special = [tokenizer.token_to_id(token) for token in ('<bos>', '<eos>', '<pad>', '<unk>')]
    if any(value is None for value in special) or len(set(special)) != 4:
        raise ValueError('Tokenizer is missing distinct required special token IDs')
    positions = sorted({0, len(data)-1, *(i*(len(data)-1)//15 for i in range(16))})
    for position in positions:
        row = data[position]
        if row['labels'][-1] != eos or row['labels'][0] == bos:
            raise ValueError(f'{name} target contract requires code+EOS without BOS')
        if any(value != 1 for value in row['attention_mask']):
            raise ValueError(f'{name} representative attention mask is not unpadded')
        if any(not 0 <= token < tokenizer.get_vocab_size()
               for column in ('input_ids', 'labels') for token in row[column]):
            raise ValueError(f'{name} representative token ID is out of vocabulary')


def prepare_full_bundle(full_cache, original_bundle, stage2_data_dir, output_dir, *,
                        full_rows=FULL_ROWS, base_rows=BASE_ROWS,
                        expected_fingerprints=None,
                        expected_original_manifest_sha=ORIGINAL_MANIFEST_SHA,
                        expected_tokenizer_sha=TOKENIZER_SHA):
    """Publish once; overrides exist only for small real-Arrow integration tests.

    The historical Stage2 proof already compared all selected original rows. This
    operation checks that source identity and hashes its current files; it never
    retokenizes, selects/reorders token rows, or repeats the historical row loop.
    """
    import datasets as ds
    import numpy as np
    from tokenizers import Tokenizer
    cache, original, stage2, output = map(lambda p: Path(p).resolve(),
        (full_cache, original_bundle, stage2_data_dir, output_dir))
    partial, lock = (output.with_name(f'.{output.name}.{suffix}') for suffix in ('partial', 'lock'))
    if output.exists() or partial.exists() or lock.exists():
        raise FileExistsError(f'Fresh output required; output, partial or lock exists: {output}')
    for source in (cache, original, stage2):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Output must be separate from immutable sources')
    if (len(full_rows) != 2 or len(base_rows) != 2
            or any(type(n) is not int or n < 1 for n in (*full_rows, *base_rows))):
        raise ValueError('Positive train/validation row counts are required')
    fingerprints = FINGERPRINTS if expected_fingerprints is None else expected_fingerprints
    marker = stage2/'complete.json'
    completed = _read(marker)
    _require(completed.get('schema'), 1, 'Stage2 data schema')
    _require(completed.get('status'), 'complete', 'Stage2 data status')
    request = completed['request']
    for key, expected in dict(base_rows=list(base_rows), full_rows=list(full_rows),
                              repeat_factor=4, seed=42, global_batch=256).items():
        _require(request.get(key), expected, f'Stage2 data request {key}')
    proof = completed['provenance']
    _require(proof.get('full_cache_path'), str(cache), 'Historical full cache path')
    _require(proof.get('verified_original_rows'), dict(zip(('train', 'val'), base_rows)),
             'Historical selected-row proof counts')
    _require(file_sha256(original/'bundle_manifest.json'), expected_original_manifest_sha,
             'Original bundle manifest SHA')
    _require(proof.get('original_bundle_sha256'), expected_original_manifest_sha,
             'Historical original bundle SHA')
    old_manifest = _read(original/'bundle_manifest.json')
    selection_sha = file_sha256(original/'selection/data_manifest.json')
    _require(selection_sha, old_manifest['selection_sha256'], 'Original selection checksum')
    _require(selection_sha, proof['original_selection_sha256'], 'Historical selection checksum')
    selection = _read(original/'selection/data_manifest.json')
    _require(selection, proof['original_selection'], 'Historical original selection')
    for key, expected in dict(train_rows=base_rows[0], validation_rows=base_rows[1],
                              seed=42, tokenizer_sha256=expected_tokenizer_sha).items():
        _require(selection['identity'].get(key), expected, f'Original selection {key}')
    _require(file_sha256(original/'tokenizer.json'), expected_tokenizer_sha, 'BPE tokenizer SHA')
    tokenizer = Tokenizer.from_file(str(original/'tokenizer.json'))
    full = ds.load_from_disk(str(cache), keep_in_memory=False)
    sources = [cache/'dataset_dict.json']
    for name, count, base_count in zip(('train', 'val'), full_rows, base_rows):
        if name not in full:
            raise ValueError(f'Missing {name} source split')
        # Match the exact load_from_disk identity used by the historical helper.
        if full[name]._indices is not None:
            raise ValueError(f'{name} source has non-null indices; flattening is forbidden')
        _require(len(full[name]), count, f'{name} source rows')
        _require(proof['full_cache'][name]['rows'], count, f'Historical {name} rows')
        _require(full[name]._fingerprint, fingerprints[name], f'{name} source fingerprint')
        _require(proof['full_cache'][name]['fingerprint'], fingerprints[name],
                 f'Historical {name} fingerprint')
        for key, expected in dict(source_rows=count, selected_rows=base_count).items():
            _require(selection['splits'][name].get(key), expected, f'Original {name} {key}')
        _require(file_sha256(original/f'selection/{name}_indices.npy'),
                 selection['splits'][name]['indices_sha256'], f'Original {name} selection indices')
        row_sha = proof.get('original_rows_sha256', {}).get(name)
        if not isinstance(row_sha, str) or len(row_sha) != 64:
            raise ValueError(f'Missing historical {name} selected-row checksum')
        _verify_tokens(full[name], tokenizer, name)
        sources.extend(_source_files(cache, name))
    _require(proof['validation_rows_sha256'], proof['original_rows_sha256']['val'],
             'Historical validation row checksum')
    output.parent.mkdir(parents=True, exist_ok=True)
    if any(source.stat().st_dev != output.parent.stat().st_dev for source in [*sources, original/'tokenizer.json']):
        raise ValueError('Hardlinks require source cache, tokenizer and output on the same filesystem')
    source_files = []
    for path in sources:
        stat = path.stat()
        sha = file_sha256(path)
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Source changed during hashing: {path}')
        source_files.append(dict(path=str(path), relative_path=path.relative_to(cache).as_posix(),
            bytes=stat.st_size, sha256=sha, mtime_ns=stat.st_mtime_ns))
    with lock.open('x') as stream:
        stream.write(str(output))
    try:
        partial.mkdir()
        linked_hashes = {}
        for entry in source_files:
            if entry['relative_path'] == 'dataset_dict.json':
                continue
            source = Path(entry['path'])
            stat = source.stat()
            if (stat.st_size, stat.st_mtime_ns) != (entry['bytes'], entry['mtime_ns']):
                raise ValueError(f'Source changed before linking: {source}')
            destination = partial/'data'/entry['relative_path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            linked_hashes[destination.relative_to(partial).as_posix()] = entry['sha256']
        write_json(partial/'data/dataset_dict.json', {'splits': ['train', 'val']})
        os.link(original/'tokenizer.json', partial/'tokenizer.json')
        linked_hashes['tokenizer.json'] = expected_tokenizer_sha
        (partial/'selection').mkdir()
        splits = {}
        for name, count in zip(('train', 'val'), full_rows):
            path = partial/f'selection/{name}_indices.npy'
            np.save(path, np.arange(count, dtype=np.int64), allow_pickle=False)
            splits[name] = dict(source_rows=count, selected_rows=count,
                source_fingerprint=full[name]._fingerprint, indices_sha256=file_sha256(path), seed=42,
                order='original_full_order')
        provenance = dict(full_cache_path=str(cache), full_cache=proof['full_cache'],
            original_bundle_sha256=expected_original_manifest_sha,
            stage2_data_complete_path=str(marker), stage2_data_complete_sha256=file_sha256(marker),
            historical_selected_row_proof=proof, source_files=source_files,
            source_trust='Existing cache and historical selected-row comparison; no repeated row comparison')
        identity = dict(selection['identity'], train_rows=full_rows[0], validation_rows=full_rows[1],
                        source=str(cache), seed=42)
        write_json(partial/'selection/data_manifest.json', dict(identity=identity,
            splits=splits, strategy='full_original_order', provenance=provenance))
        write_json(partial/'preparation.json', dict(schema=1, status='complete',
            expected_rows=list(full_rows), hardlinked=True, official_test_used=False,
            prepared_at=datetime.now(timezone.utc).isoformat()))
        entries = []
        for path in sorted(partial.rglob('*')):
            if path.is_file():
                relative = path.relative_to(partial).as_posix()
                entries.append(dict(path=relative, bytes=path.stat().st_size,
                    sha256=linked_hashes.get(relative) or file_sha256(path)))
        write_json(partial/'bundle_manifest.json', dict(schema=1, derive_length=True, files=entries,
            selection_sha256=file_sha256(partial/'selection/data_manifest.json')))
        result = dict(schema=1, status='complete', bundle_root=str(output), expected_rows=list(full_rows),
            bundle_manifest_sha256=file_sha256(partial/'bundle_manifest.json'),
            selection_sha256=file_sha256(partial/'selection/data_manifest.json'))
        partial.rename(output)
        return result
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('full-cache', 'original-bundle', 'stage2-data-dir', 'output-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    print(json.dumps(prepare_full_bundle(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()
