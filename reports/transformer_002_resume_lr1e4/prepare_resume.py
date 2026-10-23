"""Prepare the approved LR/3 continuation; run from the remote project root.

The source experiment is read-only. This creates a new checkpoint with identical
weights, Adam moments, RNG and step counters, scaling only optimizer/scheduler
learning rates and updating the continuation's metadata.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "artifacts/experiments/transformer_002_8x8_d768"
OUTPUT = ROOT / "artifacts/experiments/transformer_002_8x8_d768_lr1e4"
CHECKPOINT = SOURCE / "checkpoint-50000"
DESTINATION = OUTPUT / CHECKPOINT.name
SCALE = 1 / 3


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def clone_cache(source, target):
    # Arrow files are immutable; hard links preserve them if the old directory
    # is later removed. Each experiment has independent metadata and cache dirs.
    if Path(source).suffix == ".arrow":
        os.link(source, target)
        return target
    return shutil.copy2(source, target)


def main():
    torch.set_num_threads(2)
    assert not OUTPUT.exists(), "Refuse to overwrite an existing continuation"
    state = read(CHECKPOINT / "trainer_state.json")
    assert state["global_step"] == 50000 and state["max_steps"] == 362284
    assert state["stateful_callbacks"]["EarlyStoppingCallback"]["attributes"]["early_stopping_patience_counter"] == 0
    original_hashes = {p.name: sha256(p) for p in CHECKPOINT.iterdir() if p.is_file()}
    original_optimizer = torch.load(CHECKPOINT / "optimizer.pt", map_location="cpu", weights_only=True, mmap=True)
    assert {int(item["step"]) for item in original_optimizer["state"].values()} == {50000}
    original_scheduler = torch.load(CHECKPOINT / "scheduler.pt", map_location="cpu", weights_only=True)
    assert original_scheduler["last_epoch"] == 50000
    assert math.isclose(original_scheduler["base_lrs"][0], 3e-4)
    assert shutil.disk_usage(ROOT).free > 8 * 1024**3

    shutil.copytree(CHECKPOINT, DESTINATION)
    optimizer = {**original_optimizer, "param_groups": [dict(group) for group in original_optimizer["param_groups"]]}
    for group in optimizer["param_groups"]:
        group["lr"] *= SCALE
        group["initial_lr"] *= SCALE
    torch.save(optimizer, DESTINATION / "optimizer.pt")
    scheduler = {**original_scheduler,
                 "base_lrs": [v * SCALE for v in original_scheduler["base_lrs"]],
                 "_last_lr": [v * SCALE for v in original_scheduler["_last_lr"]]}
    torch.save(scheduler, DESTINATION / "scheduler.pt")
    state["best_model_checkpoint"] = str(DESTINATION)
    write(DESTINATION / "trainer_state.json", state)
    layout = read(CHECKPOINT / "training_layout.json")
    layout["learning_rate"] *= SCALE
    write(DESTINATION / "training_layout.json", layout)
    write(OUTPUT / "training_layout.json", layout)
    args = torch.load(CHECKPOINT / "training_args.bin", map_location="cpu", weights_only=False)
    args.learning_rate *= SCALE
    args.output_dir = str(OUTPUT)
    torch.save(args, DESTINATION / "training_args.bin")

    config = read(SOURCE / "training_config.json")
    config.update(learning_rate=config["learning_rate"] * SCALE,
                  output_dir=str(OUTPUT), resume_from_checkpoint=str(DESTINATION))
    write(OUTPUT / "training_config.json", config)
    for name in ("model_config.json", "tokenizer.json", "preprocessing.json", "environment.json"):
        shutil.copy2(SOURCE / name, OUTPUT / name)
    shutil.copytree(SOURCE / "preprocessed", OUTPUT / "preprocessed", copy_function=clone_cache)

    # Check actual serialized moments and counters, not just configuration text.
    restored = torch.load(DESTINATION / "optimizer.pt", map_location="cpu", weights_only=True, mmap=True)
    compared = 0
    assert restored["state"].keys() == original_optimizer["state"].keys()
    for key, expected in original_optimizer["state"].items():
        for name, value in expected.items():
            actual = restored["state"][key][name]
            if torch.is_tensor(value):
                assert torch.equal(actual, value), (key, name)
                compared += 1
            else:
                assert actual == value, (key, name)
    for old, new in zip(original_optimizer["param_groups"], restored["param_groups"]):
        assert {k:v for k,v in old.items() if k not in ("lr", "initial_lr")} == {k:v for k,v in new.items() if k not in ("lr", "initial_lr")}
        assert math.isclose(new["lr"], old["lr"] * SCALE, rel_tol=1e-12)
        assert math.isclose(new["initial_lr"], 1e-4, rel_tol=1e-12)
    unchanged = ["model.safetensors", "config.json", "tokenizer.json", "rng_state_0.pth", "rng_state_1.pth"]
    for name in unchanged:
        assert sha256(DESTINATION / name) == original_hashes[name]
    assert {p.name:sha256(p) for p in CHECKPOINT.iterdir() if p.is_file()} == original_hashes

    # Exercise the existing training entry point's resume guards before launch.
    from models import BaselineTransformer, TransformerConfig
    from tokenizers import Tokenizer
    from training import TrainingConfig, _validate_resume, _validate_resume_layout
    from dataclasses import fields
    cfg = TrainingConfig(tokenizer=Tokenizer.from_file(str(OUTPUT / "tokenizer.json")),
                         **{k:v for k,v in config.items() if k in {f.name for f in fields(TrainingConfig)} and k != "tokenizer"})
    model = BaselineTransformer(TransformerConfig(**read(OUTPUT / "model_config.json")))
    assert _validate_resume(DESTINATION, model.network, cfg.tokenizer) == 50000
    _validate_resume_layout(DESTINATION, layout, cfg)
    manifest = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(CHECKPOINT), "resume_checkpoint": str(DESTINATION),
        "output_dir": str(OUTPUT), "lr_scale": SCALE,
        "base_lr": config["learning_rate"], "resumed_lr": restored["param_groups"][0]["lr"],
        "global_step": 50000, "max_steps": 362284, "optimizer_tensors_compared": compared,
        "weights_rng_and_adam_state_unchanged": True, "source_checkpoint_sha256": original_hashes,
        "early_stopping_patience": 5, "patience_counter": 0, "eval_steps": config["eval_steps"],
        "data_reuse": "Immutable Arrow files hard-linked; metadata independently copied",
        "free_disk_gib_after_prepare": shutil.disk_usage(ROOT).free / 1024**3,
    }
    write(OUTPUT / "continuation_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
