r"""Launch training from saved artifacts, with one process per GPU.

Example (batch size is per GPU):
    torchrun --standalone --nproc_per_node=2 train_distributed.py \
        --dataset artifacts/dataset --tokenizer artifacts/tokenizer.json \
        --model-config artifacts/experiments/baseline/model_config.json \
        --training-config artifacts/experiments/baseline/training_config.json \
        --output-dir artifacts/experiments/ddp --train-batch-size 32
"""

import argparse
from dataclasses import fields
import json
from pathlib import Path

import datasets as ds
import torch
from tokenizers import Tokenizer
from transformers import set_seed

from models import BaselineTransformer, TransformerConfig
from training import TrainingConfig, train_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="DatasetDict saved with save_to_disk (train + val/validation).")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True, help="Our TransformerConfig JSON (model_config.json).")
    parser.add_argument("--training-config", type=Path, required=True, help="TrainingConfig JSON, including saved run metadata.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-batch-size", type=int, help="Per-process batch; global = batch * accumulation * world_size.")
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--mixed-precision", choices=("auto", "no", "fp16", "bf16"))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--num-proc", type=int, help="Preprocessing workers on rank 0 only.")
    parser.add_argument("--use-cpu", action="store_true", default=None, help="Use CPU/gloo (also useful for smoke tests).")
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume-from-checkpoint", help="Full Trainer checkpoint directory, or 'last'. Global batch and data must match.")
    checkpoint.add_argument("--warm-start-from", type=Path, help="Load weights from model.pt or Trainer checkpoint into a new run, with a fresh optimizer/scheduler.")
    args = parser.parse_args(argv)

    values = json.loads(args.training_config.expanduser().read_text(encoding="utf-8"))
    names = {item.name for item in fields(TrainingConfig)} - {"tokenizer"}
    metadata = {"tokenizer_vocab_size", "resume_from_checkpoint"}
    unknown = set(values) - names - metadata
    if unknown:
        parser.error(f"Unknown TrainingConfig fields: {', '.join(sorted(unknown))}")
    values = {key: value for key, value in values.items() if key in names}
    for name in names:
        override = getattr(args, name, None)
        if override is not None:
            values[name] = override
    config = TrainingConfig(tokenizer=Tokenizer.from_file(str(args.tokenizer.expanduser())), **values)
    model_config = TransformerConfig(**json.loads(args.model_config.expanduser().read_text(encoding="utf-8")))
    # Every torchrun worker constructs the same initial model on CPU; Trainer
    # moves it onto LOCAL_RANK's device and wraps it with DDP.
    set_seed(config.seed)
    model = BaselineTransformer(model_config)
    if args.warm_start_from is not None:
        from inference import load_inference_model

        model, saved_tokenizer = load_inference_model(args.warm_start_from.expanduser(), device="cpu")
        if model.config != model_config:
            raise ValueError("Warm-start model config differs from --model-config.")
        saved = json.loads(saved_tokenizer.to_str())
        current = json.loads(config.tokenizer.to_str())
        for name in ("model", "normalizer", "pre_tokenizer", "decoder", "post_processor", "added_tokens"):
            if saved.get(name) != current.get(name):
                raise ValueError("Warm-start requires the checkpoint's original BPE tokenizer.")
    data = ds.load_from_disk(str(args.dataset.expanduser()))
    if not isinstance(data, ds.DatasetDict):
        parser.error("--dataset must point to a saved DatasetDict.")
    resume = True if args.resume_from_checkpoint == "last" else args.resume_from_checkpoint
    try:
        result = train_model(data, model, config, resume_from_checkpoint=resume)
        if result.trainer.is_world_process_zero():
            print(json.dumps({"output_dir": str(result.output_dir), "best_checkpoint": result.best_checkpoint, "metrics": result.metrics}, ensure_ascii=False))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
