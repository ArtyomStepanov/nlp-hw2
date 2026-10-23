"""Evaluate the final best checkpoint after training has completed.

Reuse the exact saved 10k sample and generation settings from the 50k run.
Modes run sequentially in separate processes on the available A100 GPU 0.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
PREVIOUS = ROOT / "artifacts/evaluations/transformer_002_checkpoint50000_test10000_seed42_20261013"
for key, value in {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ[key] = value


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def prepare(output):
    import datasets
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = ROOT / "artifacts/experiments/transformer_002_8x8_d768_lr1e4/checkpoint-360000"
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    assert state["global_step"] == 360000
    final_state = json.loads((checkpoint.parent / "trainer_state.json").read_text())
    assert final_state["global_step"] == 362284 and final_state["epoch"] == 4.0
    assert Path(final_state["best_model_checkpoint"]) == checkpoint
    assert not (output / "sample_manifest.json").exists()
    previous = json.loads((PREVIOUS / "sample_manifest.json").read_text())
    shutil.copytree(PREVIOUS / "sample", output / "sample")
    sample = datasets.load_from_disk(str(output / "sample"))
    digest = hashlib.sha256()
    for row in sample:
        digest.update((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode())
    assert len(sample) == 10000
    assert list(sample["sample_row"]) == list(range(10000))
    assert digest.hexdigest() == previous["sample_sha256"]
    manifest = {key: previous[key] for key in (
        "dataset", "heldout_size", "heldout_fingerprint", "split_sizes", "split_seed",
        "sample_seed", "sample_size", "sample_sha256", "selection", "filtering",
        "official_test_used")}
    manifest.update(checkpoint=str(checkpoint), checkpoint_step=360000,
                    sample_copied_from=str(PREVIOUS / "sample"),
                    validation_token_loss=next(r["eval_token_loss"] for r in reversed(state["log_history"])
                                               if r.get("step") == 360000 and "eval_token_loss" in r),
                    started_at_utc=datetime.now(timezone.utc).isoformat(),
                    timing_note="Sequential inference on one A100 after training; not a Colab/T4 benchmark.")
    manifest["checkpoint_weights_sha256"] = hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
    manifest["tokenizer_sha256"] = hashlib.sha256((checkpoint / "tokenizer.json").read_bytes()).hexdigest()
    manifest["training_finished_step"] = final_state["global_step"]
    write(output / "sample_manifest.json", manifest)
    source = output / "source"
    source.mkdir()
    hashes = {}
    for relative in ["inference.py", "evaluation.py", "_code_runner.py", "utils.py",
                     *[str(p.relative_to(ROOT)) for p in (ROOT / "models").glob("*.py")]]:
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        hashes[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
    reference_hashes = json.loads((ROOT / "artifacts/evaluations/transformer_002_checkpoint90000_test10000_seed42_20261013/source/sha256.json").read_text())
    assert hashes == reference_hashes, "Inference/evaluator source changed since 90k evaluation"
    write(source / "sha256.json", hashes)
    log("Verified the exact same 10,000 held-out rows as the 50k evaluation")


def worker(output, mode):
    import datasets
    import torch
    from evaluation import EvaluationConfig, evaluate_code
    from inference import InferenceConfig, generate_code, load_inference_model
    torch.set_num_threads(2)
    device, beams, batch = "cuda:0", (1 if mode == "greedy" else 4), (128 if mode == "greedy" else 64)
    # Bound this process's allocator to ~20 GiB on the 80 GiB A100.
    torch.cuda.set_per_process_memory_fraction(0.25, device=0)
    checkpoint = ROOT / "artifacts/experiments/transformer_002_8x8_d768_lr1e4/checkpoint-360000"
    target = output / mode
    target.mkdir(exist_ok=False)
    run = {"status": "running", "mode": mode, "device": device, "num_beams": beams,
           "batch_size": batch, "precision": "fp16", "max_new_tokens": 256,
           "max_source_length": 256, "checkpoint": str(checkpoint),
           "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write(target / "run.json", run)
    started = time.perf_counter()
    try:
        sample = datasets.load_from_disk(str(output / "sample"))
        model, tokenizer = load_inference_model(checkpoint, device=device)
        original_generate = model.generate
        completed = 0

        def generate_with_progress(input_ids, **kwargs):
            nonlocal completed
            sequences = original_generate(input_ids, **kwargs)
            completed += len(input_ids)
            write(target / "progress.json", {"generated": completed, "total": len(sample),
                                             "updated_at_utc": datetime.now(timezone.utc).isoformat()})
            return sequences

        model.generate = generate_with_progress
        log(f"{mode}: generating {len(sample)} answers on {device}, batch={batch}")
        generated = generate_code(sample, model, InferenceConfig(
            tokenizer=tokenizer, batch_size=batch, num_beams=beams,
            max_source_length=256, max_new_tokens=256, mixed_precision="fp16",
            device=device, output_dir=target / "inference", sort_by_length=True))
        assert list(generated.predictions["sample_row"]) == list(range(len(sample)))
        # Release GPU allocations while the CPU checker executes candidates.
        model.to("cpu")
        torch.cuda.empty_cache()
        log(f"{mode}: generation complete; evaluating solve()")
        result = evaluate_code(generated, EvaluationConfig(
            workers=8, timeout_seconds=2.0, relative_tolerance=1e-6,
            absolute_tolerance=1e-6, output_dir=target / "evaluation"))
        assert list(result.predictions["sample_row"]) == list(range(len(sample)))
        result.metrics.update({"num_reference_errors": len(sample) - result.metrics["num_evaluated"],
                               "passed_fraction_of_all_sampled": result.metrics["num_passed"] / len(sample),
                               "num_source_truncated": sum(generated.predictions["source_truncated"]),
                               "num_hit_token_limit": sum(generated.predictions["hit_token_limit"]),
                               "official_test_used": False, "checkpoint": str(checkpoint)})
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
    codes = {}
    for mode in ["greedy", "beam4"]:
        with (output / f"{mode}.log").open("w") as stream:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                                        "--output", str(output), "--mode", mode],
                                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            log(f"Started {mode}: pid={process.pid}")
            codes[mode] = process.wait()
        write(output / "exitcodes.json", codes)
        if codes[mode]:
            raise RuntimeError(f"Evaluation failed: {codes}")
    log("Both evaluations complete")


if __name__ == "__main__":
    main()
