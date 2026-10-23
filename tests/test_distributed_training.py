"""Real two-process gloo coverage for the same Trainer path used by torchrun."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import datasets as ds
import pytest
import torch
from tokenizers import Tokenizer

from models import BaselineTransformer, TransformerConfig
from tokenizer import train_bpe_tokenizer
from training import TrainingConfig, _validate_resume_layout


def _run_worker(root: Path, output: str, world_size: int, *extra: str):
    repository = Path(__file__).resolve().parents[1]
    command = [sys.executable]
    if world_size > 1:
        # Explicit loopback avoids hostname/IPv6 discovery on macOS test hosts.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command += ["-m", "torch.distributed.run", "--master-addr=127.0.0.1", f"--master-port={port}", f"--nproc-per-node={world_size}"]
    command += [str(Path(__file__).resolve()), "--worker", str(root), output, *extra]
    env = dict(os.environ, ACCELERATE_USE_CPU="true", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    env["PYTHONPATH"] = str(repository) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(command, cwd=repository, env=env, text=True, capture_output=True, timeout=180)


def test_two_rank_training_matches_global_token_average_and_deduplicates_eval(tmp_path):
    data = ds.Dataset.from_dict({
        "question": [f"solve problem {i}" for i in range(8)],
        "code": ["x\n", "x = 1\n" * 2, "return 2\n", "def f():\n    return x + 1\n"] * 2,
    })
    train_bpe_tokenizer(data, vocab_size=300, min_frequency=1).save(str(tmp_path / "tokenizer.json"))
    ds.DatasetDict({"train": data, "val": data.select(range(5))}).save_to_disk(str(tmp_path / "data"))
    for output, world_size in (("single", 1), ("ddp", 2)):
        completed = _run_worker(tmp_path, output, world_size)
        assert completed.returncode == 0, completed.stdout + completed.stderr

    single = BaselineTransformer.load_checkpoint(tmp_path / "single" / "model.pt")
    distributed = BaselineTransformer.load_checkpoint(tmp_path / "ddp" / "model.pt")
    for name, value in single.state_dict().items():
        torch.testing.assert_close(distributed.state_dict()[name], value, atol=3e-6, rtol=3e-4)
    single_grads = torch.load(tmp_path / "single" / "gradients.pt", weights_only=True)
    distributed_grads = torch.load(tmp_path / "ddp" / "gradients.pt", weights_only=True)
    for name in single_grads:
        torch.testing.assert_close(distributed_grads[name], single_grads[name], atol=2e-6, rtol=2e-4)
    metrics = [json.loads((tmp_path / name / "metrics.json").read_text()) for name in ("single", "ddp")]
    assert metrics[0]["eval_target_tokens"] == metrics[1]["eval_target_tokens"]
    assert metrics[0]["eval_token_loss"] == pytest.approx(metrics[1]["eval_token_loss"], rel=1e-6)
    tokenizer = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    expected_tokens = sum(len(tokenizer.encode(row["code"]).ids) - 1 for row in data.select(range(5)))
    assert metrics[1]["eval_target_tokens"] == expected_tokens
    assert (tmp_path / "ddp-writes-0.txt").is_file()
    assert not (tmp_path / "ddp-writes-1.txt").exists()
    layout = json.loads((tmp_path / "ddp" / "checkpoint-2" / "training_layout.json").read_text())
    assert layout["world_size"] == 2
    assert layout["effective_batch_size"] == 4
    assert len(list((tmp_path / "ddp" / "preprocessed").glob("*/stats.json"))) == 1

    # Resume mid-epoch with an unchanged effective batch. Data skip and the LR
    # schedule must match the uninterrupted two-update baseline.
    resumed = _run_worker(tmp_path, "resumed", 2, str(tmp_path / "single" / "checkpoint-1"))
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    resumed_model = BaselineTransformer.load_checkpoint(tmp_path / "resumed" / "model.pt")
    for name, value in single.state_dict().items():
        torch.testing.assert_close(resumed_model.state_dict()[name], value, atol=3e-6, rtol=3e-4)
    state = json.loads((tmp_path / "resumed" / "trainer_state.json").read_text())
    assert state["global_step"] == 2


def test_resume_rejects_batch_or_schedule_change_during_topology_migration(tmp_path):
    config = TrainingConfig(train_bpe_tokenizer(ds.Dataset.from_dict({"question": ["x"], "code": ["y"]}), vocab_size=270))
    original = {"world_size": 1, "effective_batch_size": 64, "train_rows": 147871, "steps_per_epoch": 2311, "max_steps": 4622, "seed": 42, "dataset_fingerprint": "same", "learning_rate": 3e-4, "lr_scheduler_type": "linear", "warmup_ratio": 0.05}
    (tmp_path / "training_layout.json").write_text(json.dumps(original))
    with pytest.raises(ValueError, match="effective_batch_size"):
        _validate_resume_layout(tmp_path, original | {"world_size": 2, "effective_batch_size": 128}, config)
    with pytest.raises(ValueError, match="scheduler"):
        _validate_resume_layout(tmp_path, original | {"world_size": 2, "max_steps": 2311}, config)
    with pytest.raises(ValueError, match="learning_rate"):
        _validate_resume_layout(tmp_path, original | {"learning_rate": 1e-3}, config)
    with pytest.warns(UserWarning, match="RNG/dropout"):
        _validate_resume_layout(tmp_path, original | {"world_size": 2}, config)


def test_legacy_single_gpu_checkpoint_keeps_global_batch_and_schedule(tmp_path):
    tokenizer = train_bpe_tokenizer(ds.Dataset.from_dict({"question": ["x"], "code": ["y"]}), vocab_size=270)
    config = TrainingConfig(tokenizer, train_batch_size=32)
    checkpoint = tmp_path / "checkpoint-2311"
    checkpoint.mkdir()
    (checkpoint / "rng_state.pth").touch()
    (checkpoint / "trainer_state.json").write_text(json.dumps({"max_steps": 4622}))
    (tmp_path / "training_config.json").write_text(json.dumps(config.to_dict() | {"train_batch_size": 64}))
    (tmp_path / "preprocessing.json").write_text(json.dumps({"train": {"prepared_rows": 147871}}))
    layout = {"world_size": 2, "effective_batch_size": 64, "train_rows": 147871, "steps_per_epoch": 2311, "max_steps": 4622, "seed": 42, "learning_rate": 3e-4, "lr_scheduler_type": "linear", "warmup_ratio": 0.05}
    with pytest.warns(UserWarning, match="RNG/dropout"):
        _validate_resume_layout(checkpoint, layout, config)
    (checkpoint / "rng_state.pth").unlink()
    with pytest.raises(ValueError, match="world_size"):
        _validate_resume_layout(checkpoint, layout, config)


def _worker(root: Path, output: str, resume: str | None = None):
    from transformers import TrainerCallback
    import training

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.set_num_threads(1)
    torch.manual_seed(123)
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    model = BaselineTransformer(TransformerConfig.from_tokenizer(
        tokenizer, d_model=16, encoder_layers=1, decoder_layers=1,
        attention_heads=2, ffn_dim=32, max_position_embeddings=64, dropout=0,
    ))
    config = TrainingConfig(
        tokenizer, output_dir=root / output, max_steps=2, learning_rate=1e-3,
        train_batch_size=2 // world_size, eval_batch_size=2, gradient_accumulation_steps=2,
        max_grad_norm=1000, warmup_ratio=0, mixed_precision="no", use_cpu=True,
        max_source_length=48, max_target_length=48, eval_strategy="steps", eval_steps=1,
        logging_steps=1, num_proc=2,
    )
    original_prepare = training._prepare_dataset
    original_write = training._write_json
    original_save = model.save_checkpoint

    def record_write():
        with (root / f"{output}-writes-{rank}.txt").open("a") as handle:
            handle.write("write\n")

    def prepare(*args, **kwargs):
        assert rank == 0, "Only rank 0 may preprocess the dataset"
        return original_prepare(*args, **kwargs)

    def write(*args, **kwargs):
        record_write()
        return original_write(*args, **kwargs)

    def save(*args, **kwargs):
        record_write()
        return original_save(*args, **kwargs)

    class CaptureGradients(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
            if state.is_world_process_zero:
                torch.save({name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.grad is not None}, Path(args.output_dir) / "gradients.pt")

    original_trainer = training._TokenLossTrainer

    class CaptureTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(CaptureGradients())

    training._prepare_dataset = prepare
    training._write_json = write
    training._TokenLossTrainer = CaptureTrainer
    model.save_checkpoint = save
    try:
        result = training.train_model(ds.load_from_disk(str(root / "data")), model, config, resume_from_checkpoint=resume)
        assert result.best_checkpoint is not None
        assert result.trainer.accelerator.num_processes == world_size
        if world_size > 1:
            assert isinstance(result.trainer.model_wrapped, torch.nn.parallel.DistributedDataParallel)
        # Each rank must contain the selected best weights after training.
        restored = BaselineTransformer.load_checkpoint(root / output / "model.pt")
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value.cpu(), atol=0, rtol=0)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__" and sys.argv[1] == "--worker":
    _worker(Path(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
