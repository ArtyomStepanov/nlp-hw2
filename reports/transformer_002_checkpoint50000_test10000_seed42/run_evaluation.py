"""Reproduce the 002 evaluation on its saved 98/1/1 held-out split.

Run from the project root on the two-GPU machine. Existing outputs are never
overwritten. This script evaluates weights; it does not resume training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
EXPERIMENT = ROOT / "artifacts/experiments/transformer_002_8x8_d768"
CHECKPOINT = EXPERIMENT / "checkpoint-50000"
DATA = ROOT / "artifacts/datasets/tinysms_98_1_1_seed42/test"
for key, value in {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ[key] = value


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def prepare(output):
    import datasets
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    heldout = datasets.load_from_disk(str(DATA))
    assert len(heldout) == 118380 and {"question", "code"} <= set(heldout.column_names)
    sample = heldout.add_column("test_row_index", list(range(len(heldout))))
    sample = sample.shuffle(seed=42, keep_in_memory=True).select(range(10000))
    sample = sample.add_column("sample_row", list(range(len(sample))))
    sample.save_to_disk(str(output / "sample"))
    digest = hashlib.sha256()
    for row in sample:
        digest.update((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode())
    state = json.loads((EXPERIMENT / "trainer_state.json").read_text())
    assert Path(state["best_model_checkpoint"]).name == CHECKPOINT.name
    # Verify that the final exported model contains the actual best weights.
    exported = torch.load(EXPERIMENT / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    checked = 0
    with safe_open(CHECKPOINT / "model.safetensors", framework="pt", device="cpu") as weights:
        for key in weights.keys():
            assert torch.equal(exported["state_dict"]["network." + key], weights.get_tensor(key)), key
            checked += 1
    manifest = {
        "checkpoint": str(CHECKPOINT), "dataset": str(DATA),
        "heldout_size": len(heldout), "heldout_fingerprint": heldout._fingerprint,
        "split_sizes": {"train": 0.98, "val": 0.01, "test": 0.01},
        "split_seed": 42, "sample_seed": 42, "sample_size": len(sample),
        "sample_sha256": digest.hexdigest(), "sample_fingerprint": sample._fingerprint,
        "selection": "saved test.add_column(test_row_index).shuffle(seed=42).select(range(10000))",
        "filtering": "none; source truncation recorded; references are not truncated",
        "official_test_used": False, "exported_model_equals_best_checkpoint": True,
        "checkpoint_tensors_compared": checked,
        "best_eval_token_loss": state["best_metric"],
        "training_stop_step": state["global_step"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "timing_note": "Two modes run concurrently on separate A100s; not a T4 speed benchmark.",
    }
    write(output / "sample_manifest.json", manifest)
    log(f"Prepared {len(sample)} held-out rows; verified {checked} saved tensors")


def worker(output, mode):
    import datasets
    import torch
    from evaluation import EvaluationConfig, evaluate_code
    from inference import InferenceConfig, generate_code, load_inference_model
    torch.set_num_threads(2)
    device = "cuda:0" if mode == "greedy" else "cuda:1"
    beams = 1 if mode == "greedy" else 4
    batch = 128 if mode == "greedy" else 64
    target = output / mode
    target.mkdir(exist_ok=False)
    run = {"status": "running", "mode": mode, "device": device, "num_beams": beams,
           "batch_size": batch, "precision": "fp16", "max_new_tokens": 256,
           "max_source_length": 256, "checkpoint": str(CHECKPOINT),
           "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write(target / "run.json", run)
    started = time.perf_counter()
    try:
        sample = datasets.load_from_disk(str(output / "sample"))
        model, tokenizer = load_inference_model(CHECKPOINT, device=device)
        log(f"{mode}: generating {len(sample)} answers on {device}, batch={batch}")
        generated = generate_code(sample, model, InferenceConfig(
            tokenizer=tokenizer, batch_size=batch, num_beams=beams,
            max_source_length=256, max_new_tokens=256, mixed_precision="fp16",
            device=device, output_dir=target / "inference", sort_by_length=True))
        assert list(generated.predictions["sample_row"]) == list(range(len(sample)))
        log(f"{mode}: generation finished in {generated.metrics['total_seconds']:.1f}s; evaluating solve()")
        result = evaluate_code(generated, EvaluationConfig(
            workers=8, timeout_seconds=2.0, relative_tolerance=1e-6,
            absolute_tolerance=1e-6, output_dir=target / "evaluation"))
        assert list(result.predictions["sample_row"]) == list(range(len(sample)))
        result.metrics.update({"num_reference_errors": len(sample) - result.metrics["num_evaluated"],
                               "passed_fraction_of_all_sampled": result.metrics["num_passed"] / len(sample),
                               "num_source_truncated": sum(generated.predictions["source_truncated"]),
                               "num_hit_token_limit": sum(generated.predictions["hit_token_limit"]),
                               "official_test_used": False, "checkpoint": str(CHECKPOINT)})
        write(target / "metrics.json", result.metrics)
        run.update(status="complete", run_seconds=time.perf_counter() - started,
                   completed_at_utc=datetime.now(timezone.utc).isoformat())
        write(target / "run.json", run)
        log(f"{mode}: pass@1={result.metrics['pass@1']:.6f}, passed={result.metrics['num_passed']}/{result.metrics['num_evaluated']}")
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write(target / "run.json", run)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["greedy", "beam4"])
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode:
        return worker(output, args.mode)
    prepare(output)
    children = []
    for mode in ["greedy", "beam4"]:
        stream = (output / f"{mode}.log").open("w")
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                    "--output", str(output), "--mode", mode],
                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        children.append((mode, process, stream))
        log(f"Started {mode}: pid={process.pid}")
    codes = {}
    for mode, process, stream in children:
        codes[mode] = process.wait()
        stream.close()
        log(f"{mode}: exit={codes[mode]}")
    write(output / "exitcodes.json", codes)
    if any(codes.values()):
        raise RuntimeError(f"Evaluation failed: {codes}")
    log("Both evaluations complete")


if __name__ == "__main__":
    main()
