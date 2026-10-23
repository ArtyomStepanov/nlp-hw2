"""Run from repo root: PYTHONPATH=. .venv/bin/python reports/.../validate_t5_h100.py.

Disposable random weights: verify CUDA/BF16 parity and compiled full-size
training memory before launching the actual sweep. Does not modify datasets.
"""
from copy import deepcopy
import gc
import json
from pathlib import Path
import time

import datasets as ds
import torch
from tokenizers import Tokenizer
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast, T5ForConditionalGeneration, set_seed

from models import T5Transformer, TransformerConfig
from train_sweep import build_trials, build_trial_model, prepare_data, prepared_root, write_json
from training import _token_normalized_loss


def main():
    torch.set_num_threads(4)
    set_seed(42)
    output = Path('artifacts/sweeps/hpo_006_t5')
    config = json.loads(Path('configs/hpo_006_t5/sweep.json').read_text())
    manifest = prepare_data(config)
    small = T5Transformer(TransformerConfig(vocab_size=64, d_model=64,
        encoder_layers=2, decoder_layers=2, attention_heads=4, ffn_dim=128, dropout=0)).network.cuda().eval()
    eager = T5ForConditionalGeneration(deepcopy(small.config)).cuda().eval()
    eager.load_state_dict(small.state_dict())
    ids = torch.randint(4, 64, (4, 17), device='cuda')
    mask = torch.ones_like(ids)
    mask[0, -5:] = 0
    labels = torch.randint(4, 64, (4, 11), device='cuda')
    labels[0, -4:] = -100
    results = []
    for network in (small, eager):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = network(input_ids=ids, attention_mask=mask, labels=labels).logits
            loss = _token_normalized_loss(type('Output', (), {'logits': logits}), labels)
        loss.backward()
        results.append((logits.detach().float(), loss.item(),
            {n: p.grad.detach().clone() for n, p in network.named_parameters()}))
    actual, reference = results
    torch.testing.assert_close(actual[0], reference[0], atol=.06, rtol=.04)
    grad_l2 = sum((actual[2][n] - g).square().sum().item() for n, g in reference[2].items()) ** .5
    ref_l2 = sum(g.square().sum().item() for g in reference[2].values()) ** .5
    assert grad_l2 / ref_l2 < .04
    report = dict(cuda_bf16=dict(max_abs_logit_difference=(actual[0]-reference[0]).abs().max().item(),
        sdpa_loss=actual[1], eager_loss=reference[1], relative_gradient_l2=grad_l2/ref_l2), data_manifest=manifest)
    print(json.dumps(report['cuda_bf16']), flush=True)
    del small, eager, results, actual, reference, network, logits, loss
    gc.collect()
    torch.cuda.empty_cache()

    tok = Tokenizer.from_file(config['tokenizer'])
    trial = build_trials(output, specs=config['trials'])[0]
    wrapper = build_trial_model(trial, tok)
    wrapper.network.config.use_cache = False
    model = wrapper.network.cuda().train()
    report['parameters'] = wrapper.count_parameters()
    model = torch.compile(model, dynamic=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    # Maximum permitted padded lengths, two microbatches = effective256.
    worst = dict(input_ids=torch.randint(4, len(tok.get_vocab()), (128, 256), device='cuda'),
        attention_mask=torch.ones(128, 256, dtype=torch.long, device='cuda'),
        labels=torch.randint(4, len(tok.get_vocab()), (128, 256), device='cuda'))

    def step(batch):
        optimizer.zero_grad(set_to_none=True)
        decoder = wrapper.network.prepare_decoder_input_ids_from_labels(batch['labels'])
        total = 0.
        for _ in range(2):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                    decoder_input_ids=decoder, use_cache=False)
                loss = _token_normalized_loss(outputs, batch['labels']) / 2
            loss.backward()
            total += loss.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(norm) and torch.isfinite(loss)
        optimizer.step()
        torch.cuda.synchronize()
        return total

    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    report['worst_case_loss'] = step(worst)
    report['worst_case_seconds_including_compile'] = time.perf_counter()-begin
    report['worst_case_peak_allocated_gib'] = torch.cuda.max_memory_allocated()/2**30
    print(json.dumps({k: v for k, v in report.items() if k.startswith('worst') or k == 'parameters'}), flush=True)
    del worst
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token='<bos>', eos_token='<eos>',
        pad_token='<pad>', unk_token='<unk>', model_input_names=['input_ids', 'attention_mask'])
    data = ds.load_from_disk(str(prepared_root(config)/'data'))['train']
    rows = [{k: row[k] for k in ('input_ids', 'attention_mask', 'labels')} for row in data.select(range(128))]
    batch = DataCollatorForSeq2Seq(fast, pad_to_multiple_of=8)(rows)
    batch = {k: v.cuda() for k, v in batch.items()}
    torch.cuda.reset_peak_memory_stats()
    step(batch)  # compile actual shapes before timing
    times = []
    for _ in range(3):
        begin = time.perf_counter()
        loss = step(batch)
        times.append(time.perf_counter()-begin)
    report['real_batch'] = dict(source_length=batch['input_ids'].shape[1], target_length=batch['labels'].shape[1],
        step_seconds=times, examples_per_second=256/(sum(times)/len(times)),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, loss=loss)
    write_json(output/'validation_h100.json', report)
    print(json.dumps(report['real_batch']), flush=True)


if __name__ == '__main__':
    main()
