"""Bounded, isolated replay of checkpoint-50000 with per-step diagnostics.

Run from the project root with torchrun (two ranks). Never writes to the
original experiment or saves replay weights. The original scheduler horizon,
optimizer state, sampler and rank RNG states are restored.
"""
import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path.cwd()))

import datasets
import torch
from accelerate import PartialState
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast, Seq2SeqTrainingArguments, TrainerCallback, set_seed
from models import BaselineTransformer, TransformerConfig
from training import _TokenLossTrainer, _TokenLossMetric, _eval_statistics, _token_normalized_loss

EXPERIMENT = Path("artifacts/experiments/transformer_002_8x8_d768").resolve()
CHECKPOINT = EXPERIMENT / "checkpoint-50000"


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


class ProbeTrainer(_TokenLossTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        labels = inputs["labels"]
        self.probe_row = {
            "step": self.state.global_step + 1,
            "rank": self.args.process_index,
            "input_sha256": hashlib.sha256(inputs["input_ids"].cpu().numpy().tobytes() + labels.cpu().numpy().tobytes()).hexdigest(),
            "source_tokens": int(inputs["attention_mask"].sum()),
            "target_tokens": int(labels.ne(-100).sum()),
            "global_target_tokens": int(num_items_in_batch),
            "source_padded_length": inputs["input_ids"].shape[1],
            "target_padded_length": labels.shape[1],
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.probe_row.update(loss=float(loss), model_training=model.training)
        return loss

    def _clip_grad_norm(self, model):
        norm = super()._clip_grad_norm(model)
        value = float(norm)
        self.probe_row["grad_norm_before_clip"] = value
        if value > 3 or not torch.isfinite(norm):
            factor = max(1., value / self.args.max_grad_norm)
            top = sorted(((name, float(p.grad.norm()) * factor) for name, p in model.named_parameters()
                          if p.grad is not None), key=lambda pair: pair[1], reverse=True)[:8]
            self.probe_row["largest_gradients_before_clip"] = top
        return norm


class ProbeCallback(TrainerCallback):
    def __init__(self, trainer, output, steps, lr_scale):
        self.trainer, self.output = trainer, output
        self.stop_step, self.lr_scale = 50000 + steps, lr_scale
        self.stream = (output / f"steps-rank{trainer.args.process_index}.jsonl").open("x", buffering=1)

    def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
        assert state.global_step == 50000
        if self.lr_scale != 1:
            for group in optimizer.param_groups:
                group["lr"] *= self.lr_scale
                group["initial_lr"] *= self.lr_scale
            lr_scheduler.base_lrs = [v * self.lr_scale for v in lr_scheduler.base_lrs]
            lr_scheduler._last_lr = [v * self.lr_scale for v in lr_scheduler._last_lr]
        if state.is_world_process_zero:
            write(self.output / "resumed_state.json", {
                "step": state.global_step, "max_steps": state.max_steps,
                "lr": [g["lr"] for g in optimizer.param_groups],
                "optimizer_steps": sorted({int(v["step"]) for v in optimizer.state.values()}),
                "scheduler": lr_scheduler.state_dict(),
            })

    def on_step_end(self, args, state, control, **kwargs):
        self.stream.write(json.dumps(self.trainer.probe_row) + "\n")
        if state.global_step >= self.stop_step:
            control.should_training_stop = True
        return control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr-scale", type=float, default=1.)
    opts = parser.parse_args()
    distributed = PartialState()
    if distributed.is_main_process:
        opts.output.mkdir(parents=True, exist_ok=False)
    distributed.wait_for_everyone()
    torch.set_num_threads(2)
    set_seed(42)
    raw = json.loads((EXPERIMENT / "trainer_args.json").read_text())
    names = {f.name for f in fields(Seq2SeqTrainingArguments) if f.init}
    args = {k: v for k, v in raw.items() if k in names}
    args.update(output_dir=str(opts.output), eval_strategy="no", save_strategy="no",
                load_best_model_at_end=False, restore_callback_states_from_checkpoint=False,
                disable_tqdm=True, report_to="none", accelerator_config={"use_configured_state": True})
    args = Seq2SeqTrainingArguments(**args)
    cfg = TransformerConfig(**json.loads((EXPERIMENT / "model_config.json").read_text()))
    model = BaselineTransformer(cfg).network
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(CHECKPOINT))
    cached = list((EXPERIMENT / "preprocessed").glob("*/dataset/dataset_dict.json"))
    assert len(cached) == 1
    data = datasets.load_from_disk(str(cached[0].parent))
    trainer = ProbeTrainer(model=model, args=args, train_dataset=data["train"],
                           eval_dataset=data["val"].select(range(2048)),
                           processing_class=tokenizer,
                           data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8, label_pad_token_id=-100),
                           compute_loss_func=_token_normalized_loss,
                           compute_metrics=_TokenLossMetric(), preprocess_logits_for_metrics=_eval_statistics)
    trainer.model_accepts_loss_kwargs = False
    trainer._loss_shifts_labels = False
    callback = ProbeCallback(trainer, opts.output, opts.steps, opts.lr_scale)
    trainer.add_callback(callback)
    if distributed.is_main_process:
        write(opts.output / "probe_config.json", {"checkpoint": str(CHECKPOINT), "steps": opts.steps,
              "lr_scale": opts.lr_scale, "world_size": distributed.num_processes,
              "args": args.to_dict(), "data_fingerprint": data["train"]._fingerprint,
              "note": "No training weights saved; final validation is a fixed 2048-row subset."})
    trainer.train(resume_from_checkpoint=str(CHECKPOINT))
    metrics = trainer.evaluate()
    if distributed.is_main_process:
        write(opts.output / "final_validation.json", metrics)
        write(opts.output / "log_history.json", trainer.state.log_history)
    callback.stream.close()
    distributed.wait_for_everyone()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
