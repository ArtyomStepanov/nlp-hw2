"""One-off pass@1 run on the unchanged 20k validation set; no training."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


def run(bundle_root, checkpoint, output, *, architecture, device='cuda:0',
        batch_size=64, max_new_tokens=256, num_beams=4, limit=None,
        expected_manifest_sha='7480c7d1778bffce50db987019f2ccf87c06f8643c1ce59ad811e4fd3c77d0e1',
        expected_rows=(1000000, 20000)):
    bundle_root, checkpoint, output = map(Path, (bundle_root, checkpoint, output))
    sys.path.insert(0, str(bundle_root))
    import datasets as ds
    import torch
    from transformers import AutoModelForSeq2SeqLM
    import models  # Registers the saved copy_bart architecture.
    from models.t5_attention import enable_t5_sdpa
    from kaggle_experiments import load_bundle, write_json
    from evaluation import EvaluationConfig, evaluate_code

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite evaluation: {output}')
    output.mkdir(parents=True, exist_ok=True)
    data, tokenizer, identity = load_bundle(bundle_root, expected_rows=expected_rows,
        expected_manifest_sha=expected_manifest_sha)
    manifest = json.loads((checkpoint.parent/'run_manifest.json').read_text())
    completed = json.loads((checkpoint.parent/'complete.json').read_text())
    assert manifest['bundle_sha256'] == identity['bundle_sha256']
    assert manifest['selection_sha256'] == identity['selection_sha256']
    assert manifest['config']['architecture'] == architecture
    assert completed['architecture'] == architecture
    assert completed['metrics']['epoch'] == 1.0
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
        num_beams=num_beams, max_new_tokens=max_new_tokens, do_sample=False,
        use_cache=True, reference_source='Full saved validation labels, decoded without truncation',
        evaluation_split='validation; used during training for loss monitoring',
        official_test_used=False, started_at=datetime.now(timezone.utc).isoformat())
    write_json(output/'config.json', configuration)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    network = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True).eval()
    assert network.config.model_type == ('t5' if architecture == 't5' else 'copy_bart')
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
                do_sample=False, use_cache=True)
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
    metrics = dict(result.metrics, architecture=architecture, num_beams=num_beams,
        max_new_tokens=max_new_tokens, generation_seconds=generation_seconds,
        validation_sha256=configuration['validation_sha256'], evaluation_split='validation',
        num_reference_errors=len(rows)-result.metrics['num_evaluated'],
        num_hit_token_limit=sum(row['hit_token_limit'] for row in predictions),
        passed_fraction_of_all_sampled=result.metrics['num_passed']/len(rows),
        finished_at=datetime.now(timezone.utc).isoformat())
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
    args = parser.parse_args()
    run(**vars(args))
