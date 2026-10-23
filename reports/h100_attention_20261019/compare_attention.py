"""Isolated H100 experiment; does not modify the production model or trainer."""
import argparse
import gc
import gzip
import json
import math
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_HUB_KERNELS", "NO")
import torch
import torch.nn.functional as F
from transformers import BartForConditionalGeneration
from transformers.trainer_pt_utils import get_length_grouped_indices

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/attention_benchmark"
BACKENDS = {
    "sdpa": "sdpa",
    "fa2": "kernels-community/flash-attn2@c3f3c93a4a39e3d2cf49c13eac49229bc726ea13",
    "fa3": "kernels-community/flash-attn3@e29f138fc363b396e5d2706c8a5f6fa7d36f41e0",
}


def register_cross_attention_fix(backend):
    """HF's generic unpadding assumes Q and K share a padding mask; cross-attention does not."""
    from kernels import get_kernel
    from transformers.integrations.flash_attention import flash_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    repo, revision = BACKENDS[backend].split("@")
    kernel = get_kernel(repo, revision=revision)

    def forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        if not (module.is_decoder and not module.is_causal and attention_mask is not None):
            return flash_attention_forward(module, query, key, value, attention_mask,
                                           dropout=dropout, scaling=scaling, **kwargs)
        assert dropout == 0, "This experiment uses attention_dropout=0"
        q, k, v = (x.transpose(1, 2) for x in (query, key, value))
        b, qlen, heads, dim = q.shape
        klen = k.shape[1]
        indices = attention_mask.flatten().nonzero().flatten()
        lengths = attention_mask.sum(-1, dtype=torch.int32)
        cu_k = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
        cu_q = torch.arange(b+1, device=q.device, dtype=torch.int32) * qlen
        # All decoder queries stay in place; only encoder K/V padding is removed.
        out = kernel.flash_attn_varlen_func(
            q.reshape(-1, heads, dim), k.reshape(-1, heads, dim)[indices],
            v.reshape(-1, heads, dim)[indices], cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=qlen, max_seqlen_k=klen, softmax_scale=scaling, causal=False)
        if isinstance(out, tuple):
            out = out[0]
        return out.reshape(b, qlen, heads, dim), None

    ALL_ATTENTION_FUNCTIONS.register(BACKENDS[backend], forward)


def load_model(backend):
    model = BartForConditionalGeneration.from_pretrained(
        OUT / "checkpoint", attn_implementation=BACKENDS[backend], dtype=torch.float32
    ).cuda()
    model.config.use_cache = False
    assert model.config._attn_implementation == BACKENDS[backend]
    if backend != "sdpa":
        register_cross_attention_fix(backend)
    return model


def collate(rows, indices, model):
    s = math.ceil(max(len(rows[i]["input_ids"]) for i in indices) / 8) * 8
    t = math.ceil(max(len(rows[i]["labels"]) for i in indices) / 8) * 8
    inputs = torch.zeros((len(indices), s), dtype=torch.long)
    labels = torch.full((len(indices), t), -100, dtype=torch.long)
    for j, i in enumerate(indices):
        inputs[j, :len(rows[i]["input_ids"])] = torch.tensor(rows[i]["input_ids"])
        labels[j, :len(rows[i]["labels"])]=torch.tensor(rows[i]["labels"])
    inputs, labels = inputs.cuda(), labels.cuda()
    return dict(input_ids=inputs, attention_mask=inputs.ne(0),
                decoder_input_ids=model.prepare_decoder_input_ids_from_labels(labels)), labels


def ce(logits, labels):
    return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(),
                           ignore_index=-100, reduction="sum")


def correctness(rows):
    # Unequal source/target lengths and mixed right padding exercise self/cross attention.
    indices = [min(range(len(rows)), key=lambda i: len(rows[i]["input_ids"])),
               max(range(len(rows)), key=lambda i: len(rows[i]["input_ids"])), 42]
    refs = {}
    results = []
    for name, backend, dtype in [("fp32", "sdpa", torch.float32),
                                  ("sdpa", "sdpa", torch.bfloat16),
                                  ("fa2", "fa2", torch.bfloat16),
                                  ("fa3", "fa3", torch.bfloat16)]:
        torch.manual_seed(42)
        model = load_model(backend).eval()  # no dropout, but gradients enabled
        batch, labels = collate(rows, indices, model)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype != torch.float32):
            logits = model(**batch, use_cache=False).logits
            loss = ce(logits, labels) / labels.ne(-100).sum()
        loss.backward()
        valid_logits = logits[labels.ne(-100)].detach().float().cpu()
        grads = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
        record = {"backend": name, "loss": loss.item(), "finite": all(
            torch.isfinite(g).all().item() for g in grads.values()), "gradient_tensors": len(grads)}
        for ref_name, ref in refs.items():
            diff2 = norm2 = dot = thisnorm2 = 0.0
            for n, g in grads.items():
                r = ref["grads"][n]
                diff2 += (g - r).square().sum().item()
                norm2 += r.square().sum().item()
                thisnorm2 += g.square().sum().item()
                dot += (g * r).sum().item()
            record["vs_" + ref_name] = {
                "loss_abs_diff": abs(loss.item() - ref["loss"]),
                "logit_rmse": (valid_logits - ref["logits"]).square().mean().sqrt().item(),
                "gradient_relative_l2": math.sqrt(diff2 / norm2),
                "gradient_cosine": dot / math.sqrt(norm2 * thisnorm2),
            }
        if name in ("fp32", "sdpa"):
            refs[name] = dict(loss=loss.item(), logits=valid_logits, grads=grads)
        results.append(record)
        (OUT / "correctness_partial.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(record), flush=True)
        assert record["finite"]
        if name in ("fa2", "fa3"):
            # Accept errors comparable to BF16 SDPA's deviation from FP32, not bitwise equality.
            base = results[1]["vs_fp32"]
            assert record["vs_fp32"]["gradient_relative_l2"] <= 2 * base["gradient_relative_l2"] + .005
            assert record["vs_sdpa"]["loss_abs_diff"] < .005
            assert record["vs_sdpa"]["gradient_cosine"] > .99
        del model, batch, labels, logits, loss, grads
        gc.collect()
        torch.cuda.empty_cache()
    return results


def benchmark(rows, args):
    if args.stress:
        rows = [dict(input_ids=[2]+[10]*254+[3], labels=[10]*255+[3]) for _ in range(1024)]
    torch.manual_seed(42)
    model = load_model(args.backend).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01, fused=True)
    runner = torch.compile(model, dynamic=True) if args.compile else model
    lengths = [max(len(r["input_ids"]), len(r["labels"])) for r in rows]
    order = get_length_grouped_indices(lengths, 256, generator=torch.Generator().manual_seed(42))
    # Complete effective batches only, fixed for all implementations.
    groups = [order[i:i+256] for i in range(0, len(order)-255, 256)]

    def step(indices):
        denominator = sum(len(rows[i]["labels"]) for i in indices)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device="cuda")
        for start in range(0, 256, args.batch):
            batch, labels = collate(rows, indices[start:start+args.batch], model)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = runner(**batch, use_cache=False)
                loss = ce(output.logits, labels) / denominator
            loss.backward()
            loss_sum += loss.detach()
            del output, loss, batch, labels
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss_sum, grad.detach()

    warm = time.perf_counter()
    for i in range(6):
        step(groups[i % len(groups)])
    torch.cuda.synchronize()
    warm = time.perf_counter()-warm
    print(json.dumps({"backend": args.backend, "compile": args.compile, "warmup_seconds": warm}), flush=True)
    torch.cuda.reset_peak_memory_stats()
    runs = []
    finite = torch.ones((), device="cuda", dtype=torch.bool)
    for repeat in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for indices in groups:
            loss, grad = step(indices)
            finite &= torch.isfinite(loss) & torch.isfinite(grad)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        run = dict(seconds=seconds, examples_per_second=len(groups)*256/seconds,
                   last_loss=loss.item(), last_gradient_norm=grad.item())
        runs.append(run)
        print(json.dumps({"repeat": repeat, **run}), flush=True)
    assert finite.item()
    return dict(backend=args.backend, implementation=BACKENDS[args.backend], compile=args.compile,
                checkpoint_step=7500, microbatch=args.batch, accumulation=256//args.batch, global_batch=256,
                stress=args.stress,
                examples_per_repeat=len(groups)*256, warmup_seconds=warm,
                median_examples_per_second=statistics.median(r["examples_per_second"] for r in runs),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                finite=True, runs=runs, torch=torch.__version__, gpu=torch.cuda.get_device_name(0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=BACKENDS, default="sdpa")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--correctness", action="store_true")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--batch", type=int, choices=(128, 256), default=128)
    p.add_argument("--stress", action="store_true")
    p.add_argument("--result", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = not args.correctness
    torch.backends.cudnn.allow_tf32 = not args.correctness
    with gzip.open(ROOT / "reports/h100_training_20261019/benchmark_data.json.gz", "rt") as f:
        rows = json.load(f)
    result = correctness(rows) if args.correctness else benchmark(rows, args)
    args.result.write_text(json.dumps(result, indent=2))
    print("COMPLETE", args.result, flush=True)


if __name__ == "__main__":
    main()
