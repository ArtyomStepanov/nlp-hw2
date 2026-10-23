import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import datasets as ds
import pytest
import torch
from torch.nn import functional as F
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast

from models import BaselineTransformer, T5Transformer, CopyTransformer, TransformerConfig
from tokenizer import train_bpe_tokenizer
from training import (
    TrainingConfig,
    _eval_statistics,
    _prepare_dataset,
    _TokenLossMetric,
    _token_normalized_loss,
    train_model,
)


@pytest.fixture
def tiny_data():
    return ds.Dataset.from_dict({
        "question": ["one", "two plus two", "three times two", "four"],
        "code": [
            "def solve():\n    return 1\n",
            "def solve():\n    return 2 + 2\n",
            "def solve():\n    result = 3 * 2\n    return result\n",
            "def solve():\n    return 4\n",
        ],
    })


@pytest.fixture
def tiny_tokenizer(tiny_data):
    return train_bpe_tokenizer(tiny_data, vocab_size=320, min_frequency=1)


def test_accumulated_loss_has_same_gradients_as_full_token_average():
    # An average of microbatch means would over-weight the shorter example.
    labels = torch.tensor([[0, -100, -100], [1, 2, 0]])
    logits = torch.tensor([
        [[2.0, 0.0, -1.0], [0.0, 1.0, 2.0], [1.0, 2.0, 0.0]],
        [[0.0, 2.0, 1.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]],
    ], requires_grad=True)
    expected = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), ignore_index=-100)
    expected.backward()
    expected_grad = logits.grad.clone()
    logits.grad = None

    accumulated = sum(
        _token_normalized_loss(SimpleNamespace(logits=logits[i:i + 1]), labels[i:i + 1], num_items_in_batch=4)
        for i in range(2)
    )
    accumulated.backward()
    torch.testing.assert_close(accumulated, expected)
    torch.testing.assert_close(logits.grad, expected_grad)


def test_validation_metric_ignores_padding_and_resets_between_evaluations():
    # Uniform logits have CE=log(4); there are five actual target tokens.
    logits = torch.zeros(3, 4, 4)
    labels = torch.tensor([[0, -100, -100, -100], [1, 2, 3, -100], [2, -100, -100, -100]])
    stats = _eval_statistics(logits, labels)
    metric = _TokenLossMetric()
    assert metric(SimpleNamespace(predictions=stats[:1])) == {}
    result = metric(SimpleNamespace(predictions=stats[1:]), compute_result=True)
    assert result["token_loss"] == pytest.approx(math.log(4))
    assert result["target_tokens"] == 5
    assert metric(SimpleNamespace(predictions=stats), compute_result=True) == result


def test_preprocessing_filters_whole_examples_and_keeps_target_eos(tiny_data, tiny_tokenizer):
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tiny_tokenizer,
        bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"],
    )
    data = ds.Dataset.from_dict({
        "question": ["", None, *tiny_data["question"][:2], "too long"],
        "code": ["", None, *tiny_data["code"][:2], "x = 1\n" * 100],
    })
    config = TrainingConfig(tiny_tokenizer, max_source_length=48, max_target_length=48, tokenization_batch_size=2)
    prepared, stats = _prepare_dataset(data, fast, config, "train", max_samples=None)
    assert stats == {"original_rows": 5, "selected_rows": 5, "dropped_empty": 2, "dropped_long": 1, "prepared_rows": 2}
    assert [fast.decode(row["labels"], skip_special_tokens=True) for row in prepared] == tiny_data["code"][:2]
    assert all(row["labels"][0] != fast.bos_token_id and row["labels"][-1] == fast.eos_token_id for row in prepared)
    batch = DataCollatorForSeq2Seq(fast, label_pad_token_id=-100, pad_to_multiple_of=8)(list(prepared))
    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["labels"].shape[1] % 8 == 0
    assert batch["labels"].eq(-100).any()


@pytest.mark.parametrize('model_class', [BaselineTransformer, T5Transformer, CopyTransformer])
def test_training_updates_weights_saves_best_and_resumes(tiny_data, tiny_tokenizer, tmp_path, monkeypatch, model_class):
    # Keep this integration check small and independent of accelerator hardware.
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    torch.manual_seed(42)
    model = model_class(TransformerConfig.from_tokenizer(
        tiny_tokenizer, d_model=16, encoder_layers=1, decoder_layers=1,
        attention_heads=2, ffn_dim=32, max_position_embeddings=64, dropout=0,
    ))
    before = model.network.get_input_embeddings().weight.detach().clone()
    data = ds.DatasetDict({"train": tiny_data, "val": tiny_data.select([0, 1, 2]), "test": ds.Dataset.from_dict({"question": ["official test"]})})
    config = TrainingConfig(
        tokenizer=tiny_tokenizer, output_dir=tmp_path / "run", max_steps=2,
        learning_rate=1e-3, train_batch_size=1, eval_batch_size=1,
        gradient_accumulation_steps=2, warmup_ratio=0, mixed_precision="no",
        max_source_length=48, max_target_length=48,
        eval_strategy="steps", eval_steps=1, logging_steps=1,
    )
    result = train_model(data, model, config)
    assert result.model is model
    assert result.trainer.state.global_step == 2
    assert not model.training and not model.network.training
    assert not torch.equal(before, model.network.get_input_embeddings().weight.detach().cpu())
    assert math.isfinite(result.metrics["eval_token_loss"])
    assert result.preprocessing_stats["train"]["prepared_rows"] == 4
    checkpoint = result.best_checkpoint
    assert checkpoint is not None
    for name in ("optimizer.pt", "scheduler.pt", "trainer_state.json", "tokenizer.json"):
        assert (Path(checkpoint) / name).is_file()

    restored = model_class.load_checkpoint(result.output_dir / "model.pt")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value.cpu(), rtol=0, atol=0)

    # Seq2seq predicts every label, including the first code token. Trainer's
    # default inference classifies BART as a causal LM in Transformers 5.10.
    batches = [result.trainer.data_collator([row]) for row in result.trainer.eval_dataset]
    _, target_count = result.trainer.get_batch_samples(iter(batches), len(batches), torch.device("cpu"))
    expected_count = sum(len(row["labels"]) for row in result.trainer.eval_dataset)
    assert target_count.item() == expected_count

    # Verify the metric against CE over every non-padding token, independently
    # of Trainer's batches, then change the evaluation batch size.
    loss_sum, count = 0.0, 0
    device = next(model.parameters()).device
    with torch.inference_mode():
        for row in result.trainer.eval_dataset:
            inputs = torch.tensor([row["input_ids"]], device=device)
            labels = torch.tensor([row["labels"]], device=device)
            outputs = model(input_ids=inputs, labels=labels)
            loss_sum += F.cross_entropy(outputs.logits.float().reshape(-1, model.config.vocab_size), labels.reshape(-1), reduction="sum").item()
            count += labels.numel()
    assert result.metrics["eval_token_loss"] == pytest.approx(loss_sum / count, rel=1e-6)
    result.trainer.args.per_device_eval_batch_size = 3
    assert result.trainer.evaluate()["eval_token_loss"] == pytest.approx(loss_sum / count, rel=1e-6)

    original_config = (result.output_dir / "training_config.json").read_bytes()
    with pytest.raises(ValueError, match="checkpoint"):
        train_model(data, restored, config, resume_from_checkpoint=result.output_dir)
    assert (result.output_dir / "training_config.json").read_bytes() == original_config

    resumed = train_model(data, restored, replace(config, max_steps=3), resume_from_checkpoint=True)
    assert resumed.trainer.state.global_step == 3
    assert (result.output_dir / "training_config.json").read_bytes() == original_config
    resume_config = json.loads((result.output_dir / "resumes" / "step-2" / "training_config.json").read_text())
    assert resume_config["max_steps"] == 3


def test_training_applies_warmup_then_cosine_to_optimizer_steps(tiny_data, tiny_tokenizer, tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    model = BaselineTransformer(TransformerConfig.from_tokenizer(
        tiny_tokenizer, d_model=16, encoder_layers=1, decoder_layers=1,
        attention_heads=2, ffn_dim=32, max_position_embeddings=64, dropout=0,
    ))
    config = TrainingConfig(
        tokenizer=tiny_tokenizer, output_dir=tmp_path / "cosine",
        max_steps=6, learning_rate=1e-3, lr_scheduler_type="cosine",
        warmup_ratio=0.25, train_batch_size=1, gradient_accumulation_steps=2,
        logging_steps=1, max_source_length=48, max_target_length=48,
        mixed_precision="no",
    )
    result = train_model(tiny_data, model, config)
    recorded = [entry["learning_rate"] for entry in result.trainer.state.log_history if "learning_rate" in entry]
    # ceil(6 * .25) = 2 warmup updates; the remaining four follow a half cosine.
    # Trainer logs the LR used for each update, before stepping the scheduler.
    assert recorded == pytest.approx([0, 0.0005, 0.001, 0.000853553390593, 0.0005, 0.000146446609407])
    final_lrs = result.trainer.lr_scheduler.get_last_lr()
    assert final_lrs and all(lr == pytest.approx(0) for lr in final_lrs)
    assert json.loads((result.output_dir / "training_config.json").read_text())["lr_scheduler_type"] == "cosine"


def test_early_stopping_keeps_patience_across_resume(tiny_data, tiny_tokenizer, tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    model_config = TransformerConfig.from_tokenizer(
        tiny_tokenizer, d_model=16, encoder_layers=1, decoder_layers=1,
        attention_heads=2, ffn_dim=32, max_position_embeddings=64, dropout=0,
    )
    config = TrainingConfig(
        tiny_tokenizer, output_dir=tmp_path / "plateau", use_cpu=True,
        max_steps=10, learning_rate=1e-30, weight_decay=0, warmup_ratio=0,
        train_batch_size=2, eval_batch_size=2, mixed_precision="no",
        max_source_length=48, max_target_length=48,
        eval_strategy="steps", eval_steps=2, logging_steps=1,
        early_stopping_patience=2, save_total_limit=3,
    )
    data = ds.DatasetDict({"train": tiny_data, "val": tiny_data})
    result = train_model(data, BaselineTransformer(model_config), config)
    # With frozen numerical weights, the first check is best; two further
    # checks without improvement must stop at step 6, not the planned step 10.
    assert result.trainer.state.global_step == 6
    assert Path(result.best_checkpoint).name == "checkpoint-2"
    resumed = train_model(
        data, BaselineTransformer(model_config),
        replace(config, output_dir=tmp_path / "resumed-plateau"),
        resume_from_checkpoint=tmp_path / "plateau" / "checkpoint-4",
    )
    assert resumed.trainer.state.global_step == 6
    assert resumed.trainer.state.best_metric == pytest.approx(result.trainer.state.best_metric)


def test_loss_history_is_saved_before_training_finishes(tiny_data, tiny_tokenizer, tmp_path, monkeypatch):
    import training

    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    output_dir = tmp_path / "interrupted"

    # Interrupt at step 2 so the completed first update has been logged.
    def interrupt_second_update(self, args, state, control, **kwargs):
        if state.global_step == 2:
            raise RuntimeError("simulated interruption after logged update")
    monkeypatch.setattr(training._SaveTrainingLayout, "on_step_end", interrupt_second_update)
    model = BaselineTransformer(TransformerConfig.from_tokenizer(
        tiny_tokenizer, d_model=16, encoder_layers=1, decoder_layers=1,
        attention_heads=2, ffn_dim=32, max_position_embeddings=64, dropout=0,
    ))
    config = TrainingConfig(
        tiny_tokenizer, output_dir=output_dir, use_cpu=True, max_steps=4,
        train_batch_size=2, logging_steps=1, mixed_precision="no",
        max_source_length=48, max_target_length=48,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_model(tiny_data, model, config)
    history = json.loads((output_dir / "log_history.json").read_text())
    assert len(history) == 1 and history[0]["step"] == 1
    assert math.isfinite(history[0]["loss"])
