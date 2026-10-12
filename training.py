"""Обучение моделей из ноутбука через Hugging Face Seq2SeqTrainer."""

import importlib.util
from importlib.metadata import version
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal

import datasets as ds
import torch
from packaging.version import Version
from torch.nn import functional as F
from tokenizers import Tokenizer
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from models import BaseSeq2SeqModel
from utils import get_device


__all__ = ["TrainingConfig", "TrainingResult", "train_model"]


@dataclass
class TrainingConfig:
    tokenizer: Tokenizer = field(repr=False)
    output_dir: str | Path = "artifacts/experiments/baseline"
    source_column: str = "question"
    target_column: str = "code"
    num_train_epochs: float = 3.0
    max_steps: int = -1
    learning_rate: float = 3e-4
    train_batch_size: int = 16
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: Literal["linear", "cosine"] = "linear"
    max_grad_norm: float = 1.0
    max_source_length: int = 256
    max_target_length: int = 256
    mixed_precision: Literal["auto", "no", "fp16", "bf16"] = "auto"
    eval_strategy: Literal["epoch", "steps"] = "epoch"
    eval_steps: int = 1000
    logging_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42
    gradient_checkpointing: bool = False
    early_stopping_patience: int | None = None
    dataloader_num_workers: int = 0
    tokenization_batch_size: int = 1000
    num_proc: int | None = None
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    pad_to_multiple_of: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer, Tokenizer):
            raise TypeError("Передайте наш BPE-токенизатор в TrainingConfig(tokenizer=...).")
        for name in (
            "train_batch_size", "eval_batch_size", "gradient_accumulation_steps",
            "max_source_length", "max_target_length", "eval_steps", "logging_steps",
            "save_total_limit", "tokenization_batch_size", "pad_to_multiple_of",
        ):
            self._positive_int(name, getattr(self, name))
        for name in ("num_proc", "max_train_samples", "max_eval_samples", "early_stopping_patience"):
            if getattr(self, name) is not None:
                self._positive_int(name, getattr(self, name))
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or (self.max_steps != -1 and self.max_steps < 1):
            raise ValueError("max_steps должен быть -1 или положительным целым числом.")
        if self.num_train_epochs <= 0 or self.learning_rate <= 0 or self.max_grad_norm <= 0:
            raise ValueError("Эпохи, learning_rate и max_grad_norm должны быть положительными.")
        if self.weight_decay < 0 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("weight_decay должен быть >= 0, warmup_ratio — в диапазоне [0, 1).")
        if self.lr_scheduler_type not in {"linear", "cosine"}:
            raise ValueError("lr_scheduler_type: 'linear' или 'cosine'.")
        if self.mixed_precision not in {"auto", "no", "fp16", "bf16"}:
            raise ValueError("mixed_precision: 'auto', 'no', 'fp16' или 'bf16'.")
        if self.eval_strategy not in {"epoch", "steps"}:
            raise ValueError("eval_strategy: 'epoch' или 'steps'.")
        if not isinstance(self.dataloader_num_workers, int) or self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers должен быть неотрицательным целым числом.")
        for name in ("source_column", "target_column"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} должен быть непустой строкой.")

    @staticmethod
    def _positive_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} должен быть положительным целым числом.")

    def to_dict(self) -> dict:
        values = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "tokenizer"}
        values["output_dir"] = str(self.output_dir)
        values["tokenizer_vocab_size"] = self.tokenizer.get_vocab_size()
        return values


@dataclass
class TrainingResult:
    model: BaseSeq2SeqModel
    trainer: Seq2SeqTrainer
    metrics: dict
    preprocessing_stats: dict
    output_dir: Path
    best_checkpoint: str | None


def _encode_batch(batch: dict, tokenizer: PreTrainedTokenizerFast, source_column: str, target_column: str) -> dict:
    sources, targets = [], []
    for source, target in zip(batch[source_column], batch[target_column]):
        for column, text in ((source_column, source), (target_column, target)):
            if text is not None and not isinstance(text, str):
                raise TypeError(f"Столбец {column!r} должен содержать строки или None.")
        if source is None or target is None or not source.strip() or not target.strip():
            continue
        sources.append(source)
        targets.append(target)
    if not sources:
        return {name: [] for name in ("input_ids", "attention_mask", "labels", "source_length", "target_length")}
    encoded = tokenizer(sources, truncation=False, padding=False, return_token_type_ids=False, verbose=False)
    target_ids = tokenizer(targets, truncation=False, padding=False, return_token_type_ids=False, verbose=False)["input_ids"]
    labels = [ids[1:] for ids in target_ids]
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "source_length": [len(ids) for ids in encoded["input_ids"]],
        "target_length": [len(ids) for ids in labels],
    }


def _within_limits(source_lengths: list[int], target_lengths: list[int], max_source_length: int, max_target_length: int) -> list[bool]:
    return [source <= max_source_length and target <= max_target_length for source, target in zip(source_lengths, target_lengths)]


def _prepare_dataset(dataset: ds.Dataset, tokenizer: PreTrainedTokenizerFast, config: TrainingConfig, name: str, max_samples: int | None) -> tuple[ds.Dataset, dict]:
    if not isinstance(dataset, ds.Dataset) or not len(dataset):
        raise ValueError(f"{name} должен быть непустым datasets.Dataset.")
    missing = {config.source_column, config.target_column} - set(dataset.column_names)
    if missing:
        raise ValueError(f"В {name} отсутствуют столбцы: {', '.join(sorted(missing))}.")
    original_rows = len(dataset)
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=config.seed).select(range(max_samples))
    selected_rows = len(dataset)
    encoded = dataset.map(
        _encode_batch,
        batched=True,
        batch_size=config.tokenization_batch_size,
        num_proc=config.num_proc,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer, "source_column": config.source_column, "target_column": config.target_column},
        desc=f"Токенизация {name}",
    )
    valid_rows = len(encoded)
    prepared = encoded.filter(
        _within_limits,
        input_columns=["source_length", "target_length"],
        batched=True,
        num_proc=config.num_proc,
        fn_kwargs={"max_source_length": config.max_source_length, "max_target_length": config.max_target_length},
        desc=f"Фильтрация длин {name}",
    )
    if not len(prepared):
        raise ValueError(f"После удаления пустых и слишком длинных примеров {name} пуст.")
    stats = {
        "original_rows": original_rows,
        "selected_rows": selected_rows,
        "dropped_empty": selected_rows - valid_rows,
        "dropped_long": valid_rows - len(prepared),
        "prepared_rows": len(prepared),
    }
    return prepared.remove_columns(["source_length", "target_length"]), stats


def _token_normalized_loss(outputs, labels: torch.Tensor, num_items_in_batch=None) -> torch.Tensor:
    """Нормировать CE по числу целевых токенов, включая весь accumulation batch."""
    logits = outputs.logits
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100, reduction="sum")
    count = labels.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
    denominator = torch.as_tensor(count, dtype=torch.float32, device=loss.device).clamp_min(1)
    return loss / denominator


def _eval_statistics(logits, labels: torch.Tensor) -> torch.Tensor:
    """Сжать logits до суммы CE и количества целевых токенов каждого примера."""
    if isinstance(logits, tuple):
        logits = logits[0]
    losses = F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
    return torch.stack((losses.sum(dim=1), labels.ne(-100).sum(dim=1)), dim=1)


class _TokenLossMetric:
    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.token_count = 0

    def __call__(self, prediction, compute_result: bool = False) -> dict:
        # MPS не поддерживает float64; переносим только два числа на пример.
        stats = torch.as_tensor(prediction.predictions).detach().cpu()
        self.loss_sum += stats[:, 0].sum(dtype=torch.float64).item()
        self.token_count += int(stats[:, 1].sum().item())
        if not compute_result:
            return {}
        result = {"token_loss": self.loss_sum / max(1, self.token_count), "target_tokens": self.token_count}
        self.loss_sum = 0.0
        self.token_count = 0
        return result


def _validate_resume(path: Path, network: PreTrainedModel, tokenizer: Tokenizer) -> int:
    weight_files = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json")
    required = ("trainer_state.json", "config.json", "tokenizer.json", "optimizer.pt", "scheduler.pt")
    if not path.is_dir() or any(not (path / name).is_file() for name in required) or not any((path / name).is_file() for name in weight_files):
        raise ValueError("Для продолжения нужна полная папка Trainer checkpoint-N с весами и состоянием обучения.")
    saved_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    for name in ("vocab_size", "d_model", "encoder_layers", "decoder_layers", "encoder_attention_heads", "decoder_attention_heads", "encoder_ffn_dim", "decoder_ffn_dim", "max_position_embeddings", "pad_token_id", "bos_token_id", "eos_token_id", "decoder_start_token_id"):
        if saved_config.get(name) != getattr(network.config, name, None):
            raise ValueError(f"Конфигурация checkpoint отличается от модели: {name}.")
    saved_tokenizer = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
    current_tokenizer = json.loads(tokenizer.to_str())
    for name in ("model", "normalizer", "pre_tokenizer", "decoder", "post_processor", "added_tokens"):
        if saved_tokenizer.get(name) != current_tokenizer.get(name):
            raise ValueError("Для продолжения нужен тот же BPE-токенизатор, что и в checkpoint.")
    step = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))["global_step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("В checkpoint некорректный global_step.")
    return step


def _precision_flags(config: TrainingConfig) -> tuple[bool, bool]:
    device = get_device()
    precision = "fp16" if config.mixed_precision == "auto" and device.type == "cuda" else config.mixed_precision
    if precision in {"fp16", "bf16"} and device.type != "cuda":
        raise ValueError("В этом интерфейсе fp16/bf16 доступны на CUDA; для CPU/MPS используйте 'auto' или 'no'.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Устройство не поддерживает bf16. Для NVIDIA T4 используйте fp16.")
    return precision == "fp16", precision == "bf16"


def train_model(
    dataset: ds.Dataset | ds.DatasetDict,
    model: BaseSeq2SeqModel,
    config: TrainingConfig,
    *,
    eval_dataset: ds.Dataset | None = None,
    resume_from_checkpoint: str | Path | bool | None = None,
) -> TrainingResult:
    """Обучить модель: train_model(dataset, model, config).

    Для DatasetDict берутся train и val (или validation); test игнорируется.
    Для отдельного Dataset валидацию можно передать через eval_dataset.
    BPE передаётся внутри TrainingConfig. Длинные примеры отфильтровываются,
    код не обрезается. Лучшие веса выбираются по eval_token_loss — CE по всем
    целевым токенам валидации, без зависимости от группировки по батчам.

    resume_from_checkpoint=True продолжает последний checkpoint в output_dir;
    путь должен указывать на папку Trainer checkpoint-N, а не на model.pt.
    Модель и BPE при продолжении должны совпадать с первоначальным запуском.
    """
    if importlib.util.find_spec("accelerate") is None:
        raise ImportError("Для обучения установите Accelerate: %pip install -U accelerate")
    if Version(version("transformers")) < Version("5.10"):
        raise ImportError("Для этого интерфейса нужен transformers>=5.10: %pip install -U transformers")
    if not isinstance(config, TrainingConfig) or not isinstance(model, BaseSeq2SeqModel):
        raise TypeError("Передайте нашу модель и TrainingConfig.")
    network = getattr(model, "network", None)
    if not isinstance(network, PreTrainedModel) or not network.config.is_encoder_decoder:
        raise TypeError("Пока поддерживаются наши модели с encoder–decoder Transformers в model.network.")
    if isinstance(dataset, ds.DatasetDict):
        if "train" not in dataset:
            raise ValueError("DatasetDict должен содержать train.")
        if eval_dataset is None:
            eval_dataset = dataset.get("val", dataset.get("validation"))
        dataset = dataset["train"]
    if config.early_stopping_patience is not None and eval_dataset is None:
        raise ValueError("Для early stopping нужна валидация.")
    for length in (config.max_source_length, config.max_target_length):
        padded_length = ((length + config.pad_to_multiple_of - 1) // config.pad_to_multiple_of) * config.pad_to_multiple_of
        if padded_length > model.config.max_position_embeddings:
            raise ValueError("Лимиты длины с учётом padding превышают max_position_embeddings модели.")
    if config.tokenizer.get_vocab_size() > model.config.vocab_size:
        raise ValueError("Словарь токенизатора больше словаря модели.")
    for name, token in (("pad_token_id", "<pad>"), ("bos_token_id", "<bos>"), ("eos_token_id", "<eos>")):
        if config.tokenizer.token_to_id(token) != getattr(model.config, name):
            raise ValueError(f"ID {token} в модели и токенизаторе отличаются.")
    fp16, bf16 = _precision_flags(config)

    output_dir = Path(config.output_dir).expanduser().resolve()
    if resume_from_checkpoint is True:
        resume_from_checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None
        if resume_from_checkpoint is None:
            raise ValueError("В output_dir нет checkpoint для продолжения.")
    elif resume_from_checkpoint:
        resume_from_checkpoint = str(Path(resume_from_checkpoint).expanduser().resolve())
    else:
        resume_from_checkpoint = None
    resume_step = None
    if resume_from_checkpoint is not None:
        resume_step = _validate_resume(Path(resume_from_checkpoint), network, config.tokenizer)
    if output_dir.exists() and any(output_dir.iterdir()) and resume_from_checkpoint is None:
        raise FileExistsError("output_dir уже содержит файлы. Выберите папку нового эксперимента или resume_from_checkpoint.")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=config.tokenizer,
        bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>",
        model_max_length=model.config.max_position_embeddings,
        clean_up_tokenization_spaces=False,
        model_input_names=["input_ids", "attention_mask"],
    )
    if tokenizer.encode("", add_special_tokens=True) != [model.config.bos_token_id, model.config.eos_token_id]:
        raise ValueError("Нужен токенизатор, добавляющий BOS/EOS, как train_bpe_tokenizer.")
    train_data, train_stats = _prepare_dataset(dataset, tokenizer, config, "train", config.max_train_samples)
    stats = {"train": train_stats}
    eval_data = None
    if eval_dataset is not None:
        eval_data, stats["val"] = _prepare_dataset(eval_dataset, tokenizer, config, "val", config.max_eval_samples)
    has_eval = eval_data is not None
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=float(config.warmup_ratio),
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        optim="adamw_torch",
        fp16=fp16, bf16=bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        eval_strategy=config.eval_strategy if has_eval else "no",
        eval_steps=config.eval_steps,
        save_strategy=config.eval_strategy,
        save_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_token_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        logging_steps=config.logging_steps,
        logging_first_step=True,
        report_to="none",
        prediction_loss_only=not has_eval,
        batch_eval_metrics=has_eval,
        predict_with_generate=False,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=config.seed,
        data_seed=config.seed,
        label_names=["labels"],
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)] if config.early_stopping_patience is not None else []
    trainer = Seq2SeqTrainer(
        model=network,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=network, pad_to_multiple_of=config.pad_to_multiple_of, label_pad_token_id=-100),
        compute_loss_func=_token_normalized_loss,
        compute_metrics=_TokenLossMetric() if has_eval else None,
        preprocess_logits_for_metrics=_eval_statistics if has_eval else None,
        callbacks=callbacks,
    )
    # BART возвращает среднюю CE и игнорирует num_items_in_batch; свою CE
    # нормируем по токенам всего accumulation batch через compute_loss_func.
    trainer.model_accepts_loss_kwargs = False
    # Transformers 5.10 выводит causal loss_type из ForConditionalGeneration.
    # В seq2seq сдвигаются decoder inputs, а CE включает ВСЕ labels, включая
    # первый токен кода; исключать его из num_items_in_batch нельзя.
    trainer._loss_shifts_labels = False
    output_dir.mkdir(parents=True, exist_ok=True)
    if trainer.is_world_process_zero():
        metadata_dir = output_dir
        if resume_step is not None:
            metadata_dir = output_dir / "resumes" / f"step-{resume_step}"
            suffix = 1
            while metadata_dir.exists():
                metadata_dir = output_dir / "resumes" / f"step-{resume_step}-{suffix}"
                suffix += 1
            metadata_dir.mkdir(parents=True)
        run_config = config.to_dict()
        run_config["resume_from_checkpoint"] = resume_from_checkpoint
        _write_json(metadata_dir / "training_config.json", run_config)
        _write_json(metadata_dir / "model_config.json", model.config.to_dict())
        _write_json(metadata_dir / "preprocessing.json", stats)
        _write_json(metadata_dir / "environment.json", {"device": str(args.device), "torch": str(torch.__version__), "transformers": version("transformers"), "accelerate": version("accelerate"), "datasets": version("datasets"), "tokenizers": version("tokenizers")})
        (metadata_dir / "trainer_args.json").write_text(args.to_json_string(), encoding="utf-8")
        config.tokenizer.save(str(metadata_dir / "tokenizer.json"))
    print(f"Обучение: {len(train_data):,} строк; валидация: {len(eval_data):,}" if has_eval else f"Обучение: {len(train_data):,} строк; без валидации")
    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = dict(train_output.metrics)
    if has_eval:
        metrics.update(trainer.evaluate())
    model.eval()
    if trainer.is_world_process_zero():
        model.save_checkpoint(output_dir / "model.pt")
        trainer.save_state()
        _write_json(output_dir / "metrics.json", metrics)
        _write_json(output_dir / "log_history.json", trainer.state.log_history)
    return TrainingResult(model, trainer, metrics, stats, output_dir, trainer.state.best_model_checkpoint)


def _write_json(path: Path, values) -> None:
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
