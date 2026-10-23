import json
from dataclasses import replace

import datasets as ds
import pytest
import torch
from accelerate import skip_first_batches
from accelerate.data_loader import prepare_data_loader
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, set_seed

from models import BaselineTransformer, TransformerConfig
from tokenizer import train_bpe_tokenizer
from training import TrainingConfig, _TokenLossTrainer, _prepare_dataset, train_model


def resources():
    data = ds.Dataset.from_dict({
        "question": ["one", "two " * 12, "three", "four " * 3, "five", "six " * 8],
        "code": ["x=1\n", "x=2\n", "x=3\n" * 15, "x=4\n" * 5, "x=5\n", "x=6\n"],
    })
    tokenizer = train_bpe_tokenizer(data, vocab_size=280, min_frequency=1)
    return data, tokenizer


def test_grouping_preserves_examples_and_tracks_both_lengths():
    data, tokenizer = resources()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<bos>",
        eos_token="<eos>", pad_token="<pad>", unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"])
    config = TrainingConfig(tokenizer, group_by_length=True)
    grouped, stats = _prepare_dataset(data, fast, config, "train", None)
    original, old_stats = _prepare_dataset(data, fast, replace(config, group_by_length=False), "train", None)
    assert grouped.remove_columns("length").to_dict() == original.to_dict()
    assert stats == old_stats
    assert grouped["length"] == [max(len(r["input_ids"]), len(r["labels"])) for r in original]
    assert any(len(r["labels"]) > len(r["input_ids"]) for r in original)


def test_grouped_training_and_resume_preserve_sampler_and_optimizer(tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    data, tokenizer = resources()
    config = TrainingConfig(tokenizer, output_dir=tmp_path / "run", use_cpu=True,
        mixed_precision="no", group_by_length=True, optim="adamw_torch_fused",
        train_batch_size=2, max_steps=2, eval_strategy="steps", eval_steps=1,
        logging_steps=1, warmup_ratio=0, max_source_length=128, max_target_length=128)
    model = BaselineTransformer(TransformerConfig.from_tokenizer(tokenizer,
        d_model=16, encoder_layers=1, decoder_layers=1, attention_heads=2,
        ffn_dim=32, max_position_embeddings=128, dropout=0))
    before = model.network.model.shared.weight.detach().clone()
    result = train_model(data, model, config)
    sampler = result.trainer._get_train_sampler()
    indices = list(sampler)
    assert sorted(indices) == list(range(len(result.trainer.train_dataset)))
    assert sampler.lengths == result.trainer.train_dataset["length"]
    # Trainer removes non-forward columns before constructing its DataLoader;
    # actual training must still use target-aware lengths, not input lengths.
    loader_sampler = result.trainer.get_train_dataloader().batch_sampler.sampler
    assert loader_sampler.lengths == result.trainer.train_dataset["length"]
    assert not torch.equal(before, model.network.model.shared.weight.detach().cpu())
    assert result.trainer.optimizer.param_groups[0]["fused"]
    checkpoint = result.output_dir / "checkpoint-2"
    saved = json.loads((checkpoint / "training_layout.json").read_text())
    assert saved["group_by_length"] is True and saved["optim"] == "adamw_torch_fused"
    for changed in (replace(config, group_by_length=False), replace(config, optim="adamw_torch")):
        with pytest.raises(ValueError, match="sampler|optimizer|group_by_length|optim"):
            train_model(data, model, changed, resume_from_checkpoint=checkpoint)
    continued = train_model(data, model, replace(config, max_steps=3), resume_from_checkpoint=checkpoint)
    assert continued.trainer.state.global_step == 3


@pytest.mark.parametrize("epoch", [1, 2])
@pytest.mark.parametrize("world_size", [1, 2])
def test_grouped_resume_preserves_later_epoch_order(tmp_path, monkeypatch, epoch, world_size):
    # Exercise the actual Accelerate loader and Trainer's resume skip, after
    # earlier epochs have consumed RNG. Every input ID identifies one row.
    _, tokenizer = resources()
    data = ds.Dataset.from_dict({
        "input_ids": [[i] for i in range(120)],
        "attention_mask": [[1]] * 120, "labels": [[10]] * 120, "length": [1] * 120,
    })
    model = BaselineTransformer(TransformerConfig.from_tokenizer(tokenizer,
        d_model=16, encoder_layers=1, decoder_layers=1, attention_heads=2, ffn_dim=32))
    args = Seq2SeqTrainingArguments(output_dir=str(tmp_path), use_cpu=True,
        per_device_train_batch_size=4, train_sampling_strategy="group_by_length",
        seed=42, data_seed=42, report_to="none")

    def loader():
        set_seed(42)
        trainer = _TokenLossTrainer(model=model.network, args=args,
            train_dataset=data, data_collator=default_data_collator)
        raw = DataLoader(data.remove_columns("length"), batch_size=4,
            sampler=trainer._get_train_sampler(), collate_fn=default_data_collator)
        return trainer, prepare_data_loader(raw, num_processes=world_size, process_index=0)

    def order(batches):
        return [i for batch in batches for i in batch["input_ids"].flatten().tolist()]

    _, uninterrupted = loader()
    previous = []
    for completed_epoch in range(epoch):
        uninterrupted.set_epoch(completed_epoch)
        previous = order(uninterrupted)
    uninterrupted.set_epoch(epoch)
    expected = order(uninterrupted)
    trainer, restarted = loader()

    def resume_epoch(self, model, epoch, train_dataloader):
        # Match Transformers 5.10: skip first, then set_epoch. In DDP this
        # produces SkipBatchSampler -> BatchSamplerShard -> BatchSampler.
        skipped = skip_first_batches(train_dataloader, 5)
        skipped.set_epoch(epoch)
        return order(skipped)

    monkeypatch.setattr(Seq2SeqTrainer, "_run_epoch", resume_epoch)
    suffix = trainer._run_epoch(model=model.network, epoch=epoch, train_dataloader=restarted)
    assert suffix == expected[20:]
    assert len(set(expected[:20] + suffix)) == 120 // world_size
    assert expected != previous
