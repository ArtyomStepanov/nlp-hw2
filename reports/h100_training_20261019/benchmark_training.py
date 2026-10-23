"""Temporary H100 speed probe on real train-only examples; no quality claim."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.nn import functional as F
from transformers.trainer_pt_utils import get_length_grouped_indices
from models import BaselineTransformer, TransformerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--order", choices=["random", "grouped"], default="random")
    parser.add_argument("--optim", choices=["fused", "foreach"], default="fused")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with gzip.open(Path(__file__).with_name("benchmark_data.json.gz"), "rt") as f:
        rows = json.load(f)
    if args.stress:
        rows = [dict(input_ids=[2] + [10] * 254 + [3], attention_mask=[1] * 256,
                     labels=[10] * 255 + [3]) for _ in range(args.batch * 4)]
    lengths = [max(len(r["input_ids"]), len(r["labels"])) for r in rows]
    gen = torch.Generator().manual_seed(42)
    order = (get_length_grouped_indices(lengths, args.batch, generator=gen)
             if args.order == "grouped" else torch.randperm(len(rows), generator=gen).tolist())
    batches = [order[i:i + args.batch] for i in range(0, len(rows), args.batch)]
    config = TransformerConfig(vocab_size=8000, d_model=1024, encoder_layers=10,
                               decoder_layers=10, attention_heads=8, ffn_dim=4096)
    model = BaselineTransformer(config).network.cuda().train()
    model.config.use_cache = False
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01,
                                 fused=args.optim == "fused", foreach=args.optim == "foreach")
    runner = torch.compile(model, dynamic=True) if args.compile else model

    def collate(indices):
        source_len = math.ceil(max(len(rows[i]["input_ids"]) for i in indices) / 8) * 8
        target_len = math.ceil(max(len(rows[i]["labels"]) for i in indices) / 8) * 8
        inputs = torch.zeros(len(indices), source_len, dtype=torch.long)
        labels = torch.full((len(indices), target_len), -100, dtype=torch.long)
        for j, i in enumerate(indices):
            inputs[j, :len(rows[i]["input_ids"])]=torch.tensor(rows[i]["input_ids"])
            labels[j, :len(rows[i]["labels"])]=torch.tensor(rows[i]["labels"])
        inputs, labels = inputs.cuda(), labels.cuda()
        decoder = model.prepare_decoder_input_ids_from_labels(labels)
        return inputs, labels, decoder

    def step(indices):
        inputs, labels, decoder = collate(indices)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = runner(input_ids=inputs, attention_mask=inputs.ne(0),
                         decoder_input_ids=decoder, use_cache=False)
            loss = F.cross_entropy(out.logits.float().reshape(-1, 8000), labels.reshape(-1), ignore_index=-100)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.detach(), grad.detach(), inputs.numel()+labels.numel()

    print(json.dumps({"start": vars(args) | {"result": str(args.result)}, "parameters": params,
                      "attention": model.config._attn_implementation, "rows": len(rows)}), flush=True)
    warm_start=time.perf_counter()
    for i in range(4):
        step(batches[i % len(batches)])
    torch.cuda.synchronize()
    warm_seconds=time.perf_counter()-warm_start
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    padded_tokens=0
    max_grad=torch.tensor(0., device="cuda")
    finite=torch.tensor(True, device="cuda")
    for n, indices in enumerate(batches):
        loss, grad, padded=step(indices)
        finite &= torch.isfinite(loss) & torch.isfinite(grad)
        max_grad=torch.maximum(max_grad, grad)
        padded_tokens += padded
        if (n+1) % 32 == 0:
            print(f"batch {n+1}/{len(batches)}", flush=True)
    torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    actual_tokens=sum(len(r["input_ids"])+len(r["labels"]) for r in rows)
    result={"batch": args.batch, "order": args.order, "optim": args.optim, "compile": args.compile,
            "stress": args.stress, "rows": len(rows), "parameters": params,
            "warmup_seconds": warm_seconds, "seconds": seconds,
            "examples_per_second": len(rows)/seconds, "real_tokens_per_second": actual_tokens/seconds,
            "padding_fraction": 1-actual_tokens/padded_tokens,
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "finite": bool(finite.item()), "last_loss": loss.item(), "max_grad": max_grad.item(),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    args.result.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    assert result["finite"]


if __name__ == "__main__":
    main()
