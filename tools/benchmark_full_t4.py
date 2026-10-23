"""Verify and time the full BART finalist on all 1919 official questions on one T4."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import zipfile

sys.dont_write_bytecode = True
WEIGHT_SHA = '2059268173416c77cd8c901c791eb86e7cbcea0ede756b2bb830ec23638b8165'
WEIGHT_BYTES = 977639840
TOKENIZER_SHA = 'abd75c8655b948f612ebbf065a12269613269a4755cd0a698b0bf7a5bc1e9704'
TEST_SHA = '8bde59c8d8bb7edc4e8a503be22fc132c4a47773025d333432e78c5489e6522a'
QUESTION_SHA = '6dd86866b2ed86a09781b7f471fc2fce5c00975503c4a85f1ec959dae0b93279'
PARAMETERS = 244391936
VERSIONS = {'transformers': '5.10.2', 'tokenizers': '0.22.2', 'datasets': '4.6.0'}
ARCHITECTURE = dict(model_type='bart', architectures=['BartForConditionalGeneration'],
    vocab_size=8000, d_model=1024, encoder_layers=8, decoder_layers=8,
    encoder_attention_heads=8, decoder_attention_heads=8, encoder_ffn_dim=4096,
    decoder_ffn_dim=4096, max_position_embeddings=512, pad_token_id=0,
    bos_token_id=2, eos_token_id=3, decoder_start_token_id=2,
    tie_word_embeddings=True, forced_eos_token_id=None)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def require(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label}: expected {expected!r}, got {actual!r}')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def verify_bundle(root):
    """Check bytes and safetensors metadata without loading or allocating model weights."""
    root = Path(root).resolve()
    manifest = read(root/'manifest.json')
    for key, expected in dict(schema=1, checkpoint_step=180000, final_training_step=181144,
            model_parameters=PARAMETERS, test_rows=1919, question_order_sha256=QUESTION_SHA,
            official_test_has_targets=False).items():
        require(manifest.get(key), expected, f'Manifest {key}')
    recorded = manifest['files']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p != root/'manifest.json'}
    require(set(recorded), actual, 'Complete bundle inventory')
    for name, entry in recorded.items():
        path = root/name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Unsafe bundle file: {name}')
        require(path.stat().st_size, entry['bytes'], f'{name} bytes')
        require(digest(path), entry['sha256'], f'{name} SHA256')
    weights = root/'checkpoint/model.safetensors'
    require(weights.stat().st_size, WEIGHT_BYTES, 'Full finalist weight bytes')
    require(recorded['checkpoint/model.safetensors']['sha256'], WEIGHT_SHA, 'Full finalist weight SHA256')
    require(recorded['checkpoint/tokenizer.json']['sha256'], TOKENIZER_SHA, 'Original BPE SHA256')
    require(recorded['test.parquet']['sha256'], TEST_SHA, 'Official test SHA256')
    saved = read(root/'checkpoint/config.json')
    for key, value in ARCHITECTURE.items():
        require(saved.get(key), value, f'Full finalist architecture {key}')
    with weights.open('rb') as stream:
        header_bytes = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < header_bytes < 10_000_000:
            raise ValueError('Invalid safetensors metadata size')
        header = json.loads(stream.read(header_bytes))
    tensors = {key: value for key, value in header.items() if key != '__metadata__'}
    require(len(tensors), 344, 'Stored tensor count')
    require(tensors['final_logits_bias']['shape'], [1, 8000], 'Non-parameter logits buffer')
    require({info['dtype'] for info in tensors.values()}, {'F32'}, 'Original stored weight dtype')
    parameter_count = sum(math.prod(info['shape']) for key, info in tensors.items() if key != 'final_logits_bias')
    require(parameter_count, PARAMETERS, 'Actual stored unique parameter count')
    import pyarrow.parquet as pq
    table = pq.read_table(root/'test.parquet')
    require(table.column_names, ['question'], 'Official test columns')
    questions = table['question'].to_pylist()
    require(len(questions), 1919, 'Official question count')
    require(hashlib.sha256(json.dumps(questions, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest(),
            QUESTION_SHA, 'Official question order')
    if any(not isinstance(q, str) or not q.strip() for q in questions):
        raise ValueError('Official questions must be nonempty strings')
    return manifest, questions


def benchmark(root, output_dir, *, batch_size=128):
    root, output = Path(root).resolve(), Path(output_dir).resolve()
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer; decoder settings stay fixed')
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Write benchmark results outside the immutable bundle')
    if output.exists():
        raise FileExistsError(f'Use a fresh benchmark output: {output}')
    manifest, questions = verify_bundle(root)
    for package, version in VERSIONS.items():
        require(importlib.metadata.version(package), version, f'Installed {package}')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0, str(root))
    import torch
    import datasets as ds
    from inference import InferenceConfig, generate_code, load_inference_model
    if not torch.cuda.is_available():
        raise RuntimeError('A real NVIDIA T4 CUDA runtime is required')
    gpu = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if gpu.strip() not in ('Tesla T4', 'NVIDIA T4', 'T4') or capability != (7, 5):
        raise RuntimeError(f'Actual Tesla T4 required; got {gpu!r}, compute capability {capability}')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    output.mkdir(parents=True, exist_ok=False)
    request = dict(schema=1, checkpoint_step=180000, final_training_step=181144,
        model_parameters=PARAMETERS, weight_sha256=WEIGHT_SHA,
        bundle_manifest_sha256=digest(root/'manifest.json'), official_test_rows=1919,
        question_order_sha256=QUESTION_SHA, gpu=gpu, device='cuda:0', gpus_used=1,
        visible_gpus=torch.cuda.device_count(), compute_capability=list(capability),
        torch=str(torch.__version__), versions={key: importlib.metadata.version(key) for key in VERSIONS},
        batch_size=batch_size, num_beams=4, length_penalty=1.0, max_new_tokens=256,
        max_source_length=256, mixed_precision='fp16', sort_by_length=True,
        truncate_source=False, do_sample=False, use_cache=True, early_stopping=False,
        min_new_tokens=0, no_generated_code_execution=True,
        timer_scope='Tokenization, length sorting, transfers, generation, decoding and prediction assembly; '
                    'checkpoint loading, dependency installation and file writing excluded. No generation warmup.',
        started_at=datetime.now(timezone.utc).isoformat())
    write(output/'request.json', request)
    try:
        print(f'VERIFIED: full BART best180000, {PARAMETERS:,} parameters, {gpu}, all1919 official questions', flush=True)
        load_started = time.perf_counter()
        model, tokenizer = load_inference_model(root/'checkpoint', device='cuda:0')
        torch.cuda.synchronize(0)
        load_seconds = time.perf_counter()-load_started
        require(model.count_parameters(), PARAMETERS, 'Loaded actual parameter count')
        generation = model.network.generation_config
        for key, expected in dict(length_penalty=1.0, early_stopping=False, do_sample=False,
                num_return_sequences=1, forced_bos_token_id=None, forced_eos_token_id=None).items():
            actual = getattr(generation, key)
            if actual is None:
                actual = generation._get_default_generation_params().get(key)
            require(actual, expected, f'Loaded generation {key}')
        if generation.min_new_tokens not in (None, 0) or generation.min_length not in (None, 0):
            raise ValueError('Forced minimum generation length is forbidden in this benchmark')
        for key in ('bad_words_ids', 'force_words_ids', 'suppress_tokens', 'begin_suppress_tokens'):
            if getattr(generation, key, None) is not None:
                raise ValueError(f'Unexpected generation constraint: {key}')
        dataset = ds.Dataset.from_parquet(str(root/'test.parquet'))
        config = InferenceConfig(tokenizer=tokenizer, device='cuda:0', mixed_precision='fp16',
            batch_size=batch_size, max_source_length=256, max_new_tokens=256,
            num_beams=4, sort_by_length=True, truncate_source=False, output_dir=None)
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        started = time.perf_counter()
        result = generate_code(dataset, model, config)
        torch.cuda.synchronize(0)
        wall_seconds = time.perf_counter()-started
        peak = torch.cuda.max_memory_allocated(0)/2**30
        require(list(result.predictions['question']), questions, 'Output original question order')
        codes = result.codes
        require(len(codes), 1919, 'Generated answer count')
        if any(not isinstance(code, str) for code in codes):
            raise ValueError('Every answer must be a string')
        require(sum(batch['num_examples'] for batch in result.metrics['batch_stats']), 1919,
                'Timed batch coverage')
        metrics = dict(result.metrics, load_seconds=load_seconds,
            wall_generation_seconds=wall_seconds, peak_allocated_gib=peak,
            checkpoint_step=180000, model_parameters=PARAMETERS,
            weight_sha256=WEIGHT_SHA, question_order_sha256=QUESTION_SHA,
            gpu=gpu, batch_size=batch_size, num_beams=4, length_penalty=1.0,
            max_new_tokens=256, official_limit_seconds=180,
            within_180_seconds=result.metrics['total_seconds'] <= 180 and wall_seconds <= 180,
            empty_answer_count=sum(not code.strip() for code in codes),
            timer_scope=request['timer_scope'], finished_at=datetime.now(timezone.utc).isoformat())
        write(output/'metrics.json', metrics)
        write(output/'batch_stats.json', metrics['batch_stats'])
        # File output and JSONL validation happen after timing ends.
        result.predictions.to_json(str(output/'predictions.jsonl'), force_ascii=False)
        submission = output/'a100_full_best180000_beam4_t4.jsonl'
        submission.write_text(''.join(json.dumps({'code': code}, ensure_ascii=False)+'\n' for code in codes), encoding='utf-8')
        records = [json.loads(line) for line in submission.read_text(encoding='utf-8').splitlines()]
        require(len(records), 1919, 'Saved submission row count')
        if any(set(row) != {'code'} for row in records):
            raise ValueError('Submission must contain only the code field')
        require([row['code'] for row in records], codes, 'Saved submission order')
        write(output/'complete.json', dict(status='complete', within_180_seconds=metrics['within_180_seconds'],
            submission_sha256=digest(submission), submission_bytes=submission.stat().st_size,
            bundle_manifest_sha256=digest(root/'manifest.json')))
        files = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(output.iterdir()) if p.is_file()}
        write(output/'results_manifest.json', dict(schema=1, files=files))
        archive = output/'t4_results.zip'
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            for path in sorted(output.iterdir()):
                if path.is_file() and path != archive:
                    zipped.write(path, path.name)
        print(json.dumps(dict(status='complete', total_seconds=metrics['total_seconds'],
            wall_generation_seconds=wall_seconds, load_seconds=load_seconds,
            within_180_seconds=metrics['within_180_seconds'], peak_allocated_gib=peak,
            results_zip=str(archive), results_zip_sha256=digest(archive)), indent=2), flush=True)
        return metrics
    except BaseException as error:
        write(output/'failed.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
            failed_at=datetime.now(timezone.utc).isoformat()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        manifest, questions = verify_bundle(args.bundle_root)
        print(json.dumps(dict(status='verified', rows=len(questions), model_parameters=manifest['model_parameters'],
                              weight_sha256=WEIGHT_SHA), indent=2))
    else:
        if args.output_dir is None:
            parser.error('--output-dir is required for an actual T4 benchmark')
        benchmark(args.bundle_root, args.output_dir, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
