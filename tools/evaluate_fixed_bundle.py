"""Generate and score a checkpoint on its verified fixed validation bundle."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time


ORIGINAL_MANIFEST_SHA = '7480c7d1778bffce50db987019f2ccf87c06f8643c1ce59ad811e4fd3c77d0e1'
ORIGINAL_VALIDATION_SHA = '282001d478bffb2d13ab2ca41e8232a0c99c3623f898dd7077128c266172c406'


def _require(actual, expected, label):
    if actual != expected:
        raise ValueError(f'{label} mismatch: expected {expected!r}, got {actual!r}')


def _rows_sha(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n').encode())
    return digest.hexdigest()


def _verify_training_bundle(bundle_root, checkpoint, original, identity, manifest, completed, *,
                            training_bundle_root, expected_training_manifest_sha, expected_training_rows):
    """Prove the exported best model and all fixed validation rows against its real training bundle."""
    import numpy as np
    from kaggle_experiments import file_sha256, load_bundle
    from tools.prepare_a100_diversity import _compare_rows

    training_root = Path(training_bundle_root).resolve()
    full, _, training_identity = load_bundle(training_root, expected_rows=expected_training_rows,
        expected_manifest_sha=expected_training_manifest_sha)
    selection = training_identity['selection']
    _require(selection.get('strategy'), 'full_original_order', 'Training selection strategy')
    proof = selection['provenance']['historical_selected_row_proof']
    for key, expected in dict(original_bundle_sha256=identity['bundle_sha256'],
            original_selection_sha256=identity['selection_sha256'],
            original_selection=identity['selection']).items():
        _require(proof.get(key), expected, f'Historical source proof {key}')
    _require(selection['provenance'].get('original_bundle_sha256'), identity['bundle_sha256'],
             'Training source original bundle')
    tokenizer_sha = identity['selection']['identity']['tokenizer_sha256']
    _require(selection['identity']['tokenizer_sha256'], tokenizer_sha, 'Shared BPE tokenizer')
    for key, expected in dict(bundle_sha256=training_identity['bundle_sha256'],
            selection_sha256=training_identity['selection_sha256'], train_rows=len(full['train']),
            validation_rows=len(full['val'])).items():
        _require(manifest.get(key), expected, f'Checkpoint training {key}')
    _require(manifest['config'].get('expected_manifest_sha'), training_identity['bundle_sha256'],
             'Checkpoint pinned training bundle')
    _require(manifest['config'].get('expected_rows'), list(expected_training_rows),
             'Checkpoint training row counts')
    count = len(full['val'])
    for key in ('source_rows', 'selected_rows'):
        _require(selection['splits']['val'].get(key), count, f'Full validation {key}')
    full_indices = np.load(training_root/'selection/val_indices.npy', mmap_mode='r', allow_pickle=False)
    if full_indices.dtype.kind not in 'iu' or not np.array_equal(full_indices, np.arange(count)):
        raise ValueError('Full validation indices must preserve every original source row in order.')
    indices_path = Path(bundle_root)/'selection/val_indices.npy'
    indices = np.load(indices_path, mmap_mode='r', allow_pickle=False)
    if (indices.dtype.kind not in 'iu' or indices.shape != (len(original['val']),)
            or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= count)):
        raise ValueError('Fixed validation indices must be unique valid original source row IDs.')
    _require(identity['selection']['splits']['val'].get('source_rows'), count,
             'Fixed validation original source size')
    token_rows_sha = _compare_rows(full['val'].select(indices), original['val'],
        name='fixed validation vs full validation', batch_size=1000)

    trial = checkpoint.parent.resolve()
    _require(checkpoint.resolve(), trial/'final', 'Exported final checkpoint path')
    step, best_step = completed.get('global_step'), completed.get('best_global_step')
    if type(step) is not int or type(best_step) is not int or not 0 < best_step <= step:
        raise ValueError('Completed training must record a valid best checkpoint step.')
    for value in (completed.get('best_metric'), completed.get('metrics', {}).get('eval_token_loss')):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Completed training must record finite validation metrics.')
    _require(completed.get('final_weights_source'), 'best_eval_token_loss', 'Final weights source')
    best = Path(completed['best_model_checkpoint']).resolve()
    _require(best, trial/f'checkpoint-{best_step}', 'Saved best checkpoint path')
    state = json.loads((trial/'trainer_state.json').read_text())
    for key, expected in dict(global_step=step, best_global_step=best_step,
                              best_metric=completed['best_metric']).items():
        _require(state.get(key), expected, f'Completed trainer state {key}')
    _require(json.loads((best/'trainer_state.json').read_text()).get('global_step'), best_step,
             'Best checkpoint trainer step')
    _require(json.loads((best/'run_manifest.json').read_text()), manifest, 'Best checkpoint manifest')
    _require(json.loads((checkpoint/'model_config.json').read_text()), manifest['model_config'],
             'Exported model configuration')
    for path in (trial/'source_tokenizer.json', checkpoint/'tokenizer.json', best/'tokenizer.json'):
        _require(file_sha256(path), tokenizer_sha, f'Checkpoint tokenizer {path.name}')
    def weights(root):
        paths = sorted([*root.glob('*.safetensors'), *root.glob('pytorch_model*.bin'),
                        *root.glob('*.safetensors.index.json'), *root.glob('pytorch_model*.bin.index.json')])
        if not paths or any(not p.is_file() or not p.stat().st_size for p in paths):
            raise ValueError('Checkpoint must contain complete nonempty saved model weights.')
        return {p.name: file_sha256(p) for p in paths}
    weight_sha = weights(checkpoint)
    _require(weight_sha, weights(best), 'Exported final vs best weights')
    return dict(training_bundle_root=str(training_root),
        training_bundle_sha256=training_identity['bundle_sha256'],
        training_selection_sha256=training_identity['selection_sha256'],
        training_manifest_sha256=file_sha256(trial/'run_manifest.json'),
        training_completion_sha256=file_sha256(trial/'complete.json'),
        training_global_step=step, checkpoint_step=best_step, checkpoint_weight_sha256=weight_sha,
        validation_mapping=dict(verified_rows=len(original['val']), full_validation_rows=count,
            indices_sha256=file_sha256(indices_path), token_rows_sha256=token_rows_sha,
            validation_sha256=_rows_sha(original['val'])))


def run(bundle_root, checkpoint, output, *, architecture, device='cuda:0',
        batch_size=64, max_new_tokens=256, num_beams=4, limit=None,
        expected_manifest_sha=ORIGINAL_MANIFEST_SHA,
        expected_rows=(1000000, 20000), length_penalty=1.0,
        training_bundle_root=None, expected_training_manifest_sha=None,
        expected_training_rows=(11593060, 118302), reference_identity=None):
    bundle_root, checkpoint, output = map(Path, (bundle_root, checkpoint, output))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import datasets as ds
    import torch
    from transformers import AutoModelForSeq2SeqLM
    import models  # Registers the saved copy_bart architecture.
    from models.t5_attention import enable_t5_sdpa
    from kaggle_experiments import file_sha256, load_bundle, write_json
    from evaluation import EvaluationConfig, evaluate_code

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite evaluation: {output}')
    output.mkdir(parents=True, exist_ok=True)
    data, tokenizer, identity = load_bundle(bundle_root, expected_rows=expected_rows,
        expected_manifest_sha=expected_manifest_sha)
    manifest = json.loads((checkpoint.parent/'run_manifest.json').read_text())
    completed = json.loads((checkpoint.parent/'complete.json').read_text())
    cross_bundle = training_bundle_root is not None or expected_training_manifest_sha is not None
    provenance = {}
    if cross_bundle:
        if not all((training_bundle_root, expected_training_manifest_sha, reference_identity, expected_manifest_sha)):
            raise ValueError('Cross-bundle evaluation requires explicit pinned training/original bundles and reference identity.')
        provenance = _verify_training_bundle(bundle_root, checkpoint, data, identity, manifest, completed,
            training_bundle_root=training_bundle_root, expected_training_manifest_sha=expected_training_manifest_sha,
            expected_training_rows=expected_training_rows)
    else:
        if reference_identity is not None:
            raise ValueError('Explicit reference identity requires the training bundle proof inputs.')
        assert manifest['bundle_sha256'] == identity['bundle_sha256']
        assert manifest['selection_sha256'] == identity['selection_sha256']
    assert manifest['config']['architecture'] == architecture
    assert completed['architecture'] == architecture
    assert completed['global_step'] > 0
    golden = None
    if cross_bundle:
        golden = json.loads(Path(reference_identity).read_text())
        validation_sha = _rows_sha(data['val'])
        if expected_manifest_sha == ORIGINAL_MANIFEST_SHA:
            _require(validation_sha, ORIGINAL_VALIDATION_SHA, 'Canonical fixed validation identity')
        _require(golden.get('validation_sha256'), validation_sha, 'Golden validation identity')
        _require(len(golden['rows']), len(data['val']), 'Golden reference row count')
        _require([row.get('sample_row') for row in golden['rows']], list(range(len(data['val']))),
                 'Golden reference row order')
        provenance.update(reference_identity=str(Path(reference_identity).resolve()),
            reference_identity_sha256=file_sha256(reference_identity))
    validation = data['val']
    if limit is not None:
        validation = validation.select(range(min(limit, len(validation))))
    rows = list(validation)
    row_digest = hashlib.sha256()
    for row in rows:
        row_digest.update((json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n').encode())
    configuration = dict(architecture=architecture, checkpoint=str(checkpoint),
        checkpoint_step=completed['global_step'], validation_rows=len(rows),
        validation_sha256=row_digest.hexdigest(), bundle_sha256=identity['bundle_sha256'],
        selection_sha256=identity['selection_sha256'], device=device, batch_size=batch_size,
        num_beams=num_beams, max_new_tokens=max_new_tokens, length_penalty=length_penalty, do_sample=False,
        use_cache=True, reference_source='Full saved validation labels, decoded without truncation',
        evaluation_split='validation; used during training for loss monitoring',
        official_test_used=False, started_at=datetime.now(timezone.utc).isoformat())
    configuration.update(provenance)
    write_json(output/'config.json', configuration)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    network = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True).eval()
    assert network.config.model_type == {'t5':'t5','copy':'copy_bart','bart':'bart'}[architecture]
    for field, token in [('pad_token_id','<pad>'),('bos_token_id','<bos>'),('eos_token_id','<eos>')]:
        actual = getattr(network.config, field)
        assert actual == tokenizer.token_to_id(token), (field, actual)
    assert network.config.decoder_start_token_id == tokenizer.token_to_id('<bos>')
    assert network.config.vocab_size == tokenizer.get_vocab_size()
    if architecture == 't5':
        enable_t5_sdpa(network)
    network.config.use_cache = True
    network.to(device)
    cuda = torch.device(device).type == 'cuda'
    synchronize = (lambda: torch.cuda.synchronize(device)) if cuda else (lambda: None)
    context = torch.autocast('cuda', dtype=torch.float16) if cuda else nullcontext()
    pad, eos = tokenizer.token_to_id('<pad>'), tokenizer.token_to_id('<eos>')
    order = sorted(range(len(rows)), key=lambda index: len(rows[index]['input_ids']))
    predictions = [None] * len(rows)
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode(), context:
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset+batch_size]
            width = max(len(rows[index]['input_ids']) for index in indices)
            input_ids = torch.tensor([rows[index]['input_ids'] + [pad]*(width-len(rows[index]['input_ids']))
                for index in indices], device=device)
            attention_mask = torch.tensor([rows[index]['attention_mask'] + [0]*(width-len(rows[index]['input_ids']))
                for index in indices], device=device)
            generated = network.generate(input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, num_beams=num_beams, num_return_sequences=1,
                do_sample=False, use_cache=True, length_penalty=length_penalty)
            assert generated.shape[0] == len(indices)
            for index, sequence in zip(indices, generated.cpu().tolist()):
                # First token is the decoder start, not a model-generated answer token.
                sequence = sequence[1:]
                ended = eos in sequence
                sequence = sequence[:sequence.index(eos)+1] if ended else sequence
                predictions[index] = dict(sample_row=index,
                    question=tokenizer.decode(rows[index]['input_ids'], skip_special_tokens=True),
                    code=tokenizer.decode(rows[index]['labels'], skip_special_tokens=True),
                    generated_code=tokenizer.decode(sequence, skip_special_tokens=True),
                    generated_tokens=len(sequence), ended_with_eos=ended,
                    hit_token_limit=not ended and len(sequence) >= max_new_tokens)
            done = min(offset+batch_size, len(order))
            elapsed = time.perf_counter()-started
            write_json(output/'progress.json', dict(generated=done, total=len(rows),
                phase='generation', elapsed_seconds=elapsed))
            if offset == 0 or done % (batch_size*10) == 0 or done == len(rows):
                print(f'{architecture}: generated {done}/{len(rows)} in {elapsed:.1f}s', flush=True)
    synchronize()
    generation_seconds = time.perf_counter()-started
    dataset = ds.Dataset.from_list(predictions)
    dataset.to_json(str(output/'predictions.jsonl'), force_ascii=False)
    network.to('cpu')
    del network
    if cuda:
        torch.cuda.empty_cache()
    write_json(output/'progress.json', dict(generated=len(rows), total=len(rows), phase='execution'))
    print(f'{architecture}: execution checks, {len(rows)} references and predictions', flush=True)
    result = evaluate_code(dataset, EvaluationConfig(workers=4, timeout_seconds=2,
        relative_tolerance=1e-6, absolute_tolerance=1e-6, output_dir=output/'evaluation'))
    assert list(result.predictions['sample_row']) == list(range(len(rows)))
    if golden is not None:
        from tools.run_a100_stage1 import verify_reference_results
        verify_reference_results(output/'evaluation/evaluation.jsonl', golden['rows'][:len(rows)])
        _require(result.metrics['num_evaluated'], sum(row['evaluable'] for row in golden['rows'][:len(rows)]),
                 'Golden eligible reference count')
    metrics = dict(result.metrics, architecture=architecture, num_beams=num_beams,
        max_new_tokens=max_new_tokens, length_penalty=length_penalty, generation_seconds=generation_seconds,
        validation_sha256=configuration['validation_sha256'], evaluation_split='validation',
        num_reference_errors=len(rows)-result.metrics['num_evaluated'],
        num_hit_token_limit=sum(row['hit_token_limit'] for row in predictions),
        passed_fraction_of_all_sampled=result.metrics['num_passed']/len(rows),
        finished_at=datetime.now(timezone.utc).isoformat())
    if golden is not None:
        metrics['reference_identity_verified'] = True
    write_json(output/'metrics.json', metrics)
    write_json(output/'progress.json', dict(generated=len(rows), total=len(rows), phase='complete'))
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle_root','checkpoint','output','architecture'):
        parser.add_argument('--'+name.replace('_','-'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--num-beams', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--length-penalty', type=float, default=1.0)
    parser.add_argument('--training-bundle-root')
    parser.add_argument('--expected-training-manifest-sha')
    parser.add_argument('--reference-identity')
    args = parser.parse_args()
    run(**vars(args))
