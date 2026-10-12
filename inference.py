"""Батчевая генерация кода из ноутбука с сохранением порядка и времени."""

from contextlib import nullcontext
from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from time import perf_counter
from typing import Literal

import datasets as ds
import torch
from tokenizers import Tokenizer

from models import BaseSeq2SeqModel, BaselineTransformer
from utils import get_device


@dataclass
class InferenceConfig:
    tokenizer: Tokenizer = field(repr=False)
    batch_size: int = 32
    source_column: str = "question"
    max_source_length: int | None = None
    max_new_tokens: int = 256
    num_beams: int = 1
    sort_by_length: bool = True
    truncate_source: bool = True
    mixed_precision: Literal["auto", "no", "fp16", "bf16"] = "auto"
    device: str | torch.device | None = None
    output_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer, Tokenizer):
            raise TypeError("Передайте наш BPE в InferenceConfig(tokenizer=...).")
        for name in ("batch_size", "max_new_tokens", "num_beams", "max_source_length"):
            value = getattr(self, name)
            if name == "max_source_length" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < (2 if name == "max_source_length" else 1):
                raise ValueError(f"{name} должен быть положительным целым числом (длина входа >= 2).")
        if self.mixed_precision not in {"auto", "no", "fp16", "bf16"}:
            raise ValueError("mixed_precision: 'auto', 'no', 'fp16' или 'bf16'.")
        if not isinstance(self.source_column, str) or not self.source_column:
            raise ValueError("source_column должен быть непустой строкой.")
        if not isinstance(self.sort_by_length, bool) or not isinstance(self.truncate_source, bool):
            raise TypeError("sort_by_length и truncate_source должны быть bool.")

    def to_dict(self) -> dict:
        values = {item.name: getattr(self, item.name) for item in fields(self) if item.name != "tokenizer"}
        for name in ("device", "output_dir"):
            if values[name] is not None:
                values[name] = str(values[name])
        values["tokenizer_vocab_size"] = self.tokenizer.get_vocab_size()
        return values


@dataclass
class InferenceResult:
    predictions: ds.Dataset
    metrics: dict
    output_dir: Path | None = None

    @property
    def codes(self) -> list[str]:
        """Один ответ на каждый вход, в исходном порядке."""
        return list(self.predictions["generated_code"])


def load_inference_model(
    path: str | Path, *, tokenizer_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> tuple[BaselineTransformer, Tokenizer]:
    """Загрузить model.pt и tokenizer.json, которые сохраняет train_model.

    path — папка завершённого обучения или непосредственно model.pt.
    Trainer checkpoint-N имеет другой формат и сюда не передаётся.
    """
    path = Path(path).expanduser()
    weights_path = path / "model.pt" if path.is_dir() else path
    tokenizer_path = Path(tokenizer_path).expanduser() if tokenizer_path is not None else weights_path.parent / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = BaselineTransformer.load_checkpoint(weights_path, map_location="cpu")
    _check_tokenizer(model, tokenizer)
    model.to(get_device() if device is None else torch.device(device))
    model.eval()
    return model, tokenizer


def _check_tokenizer(model: BaseSeq2SeqModel, tokenizer: Tokenizer) -> None:
    if tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError("Размер словаря токенизатора отличается от модели.")
    for name, token in (("pad_token_id", "<pad>"), ("bos_token_id", "<bos>"), ("eos_token_id", "<eos>")):
        if tokenizer.token_to_id(token) != getattr(model.config, name):
            raise ValueError(f"ID {token} отличается от конфигурации модели.")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def generate_code(dataset: ds.Dataset, model: BaseSeq2SeqModel, config: InferenceConfig) -> InferenceResult:
    """Сгенерировать ровно один кандидат для каждого вопроса.

    Передайте конкретную часть Dataset: parts['val'] или data['test'].
    Входы группируются по длине, выходы возвращаются в исходном порядке.
    Длинные вопросы обрезаются с сохранением EOS и отмечаются в результатах;
    truncate_source=False вместо этого выдаёт ошибку. Код не постобрабатывается.
    total_seconds включает токенизацию, передачу на устройство, генерацию и
    декодирование; запись файлов и загрузка checkpoint в него не входят.
    """
    if not isinstance(dataset, ds.Dataset) or not len(dataset):
        raise ValueError("Передайте непустой datasets.Dataset с конкретной частью данных.")
    if config.source_column not in dataset.column_names:
        raise ValueError(f"Нет столбца {config.source_column!r}.")
    sources = list(dataset[config.source_column])
    if any(not isinstance(text, str) or not text.strip() for text in sources):
        raise ValueError("Все вопросы должны быть непустыми строками; строки не удаляются автоматически.")
    _check_tokenizer(model, config.tokenizer)
    limit = config.max_source_length or model.config.max_position_embeddings
    if limit > model.config.max_position_embeddings:
        raise ValueError("max_source_length превышает max_position_embeddings.")
    if config.max_new_tokens >= model.config.max_position_embeddings:
        raise ValueError("max_new_tokens должен быть меньше max_position_embeddings.")
    output_dir = Path(config.output_dir).expanduser() if config.output_dir is not None else None
    if output_dir is not None and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_dir должен быть пустым, чтобы не перезаписать эксперимент.")
    device = get_device() if config.device is None else torch.device(config.device)
    precision = ("fp16" if device.type == "cuda" else "no") if config.mixed_precision == "auto" else config.mixed_precision
    if precision in {"fp16", "bf16"} and device.type != "cuda":
        raise ValueError("fp16/bf16 доступны на CUDA; для CPU/MPS используйте 'auto' или 'no'.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Устройство не поддерживает bf16; для NVIDIA T4 используйте fp16.")

    _synchronize(device)
    started = perf_counter()
    # Копия не меняет включённые padding/truncation у токенизатора в ноутбуке.
    tokenizer = Tokenizer.from_str(config.tokenizer.to_str())
    tokenizer.no_padding()
    tokenizer.no_truncation()
    encoded = [item.ids for item in tokenizer.encode_batch(sources)]
    bos, eos, pad = model.config.bos_token_id, model.config.eos_token_id, model.config.pad_token_id
    if any(not ids or ids[0] != bos or ids[-1] != eos for ids in encoded):
        raise ValueError("Токенизатор должен добавлять BOS и EOS, как train_bpe_tokenizer.")
    original_lengths = list(map(len, encoded))
    truncated = [length > limit for length in original_lengths]
    if any(truncated) and not config.truncate_source:
        raise ValueError(f"Есть слишком длинные вопросы (>{limit} токенов); разрешите обрезку truncate_source=True.")
    encoded = [ids[:limit - 1] + [eos] if len(ids) > limit else ids for ids in encoded]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i])) if config.sort_by_length else list(range(len(encoded)))
    codes, lengths, ended, hit_limit = [None] * len(encoded), [0] * len(encoded), [False] * len(encoded), [False] * len(encoded)
    batch_stats = []
    was_training = model.training
    model.to(device)
    model.eval()
    context = torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16) if precision != "no" else nullcontext()
    try:
        with torch.inference_mode(), context:
            for offset in range(0, len(order), config.batch_size):
                indices = order[offset:offset + config.batch_size]
                width = max(len(encoded[index]) for index in indices)
                input_ids = torch.tensor([encoded[index] + [pad] * (width - len(encoded[index])) for index in indices], device=device)
                attention_mask = input_ids.ne(pad).long()
                _synchronize(device)
                batch_started = perf_counter()
                sequences = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=config.max_new_tokens, num_beams=config.num_beams)
                _synchronize(device)
                seconds = perf_counter() - batch_started
                if sequences.ndim != 2 or sequences.shape[0] != len(indices):
                    raise ValueError("generate должен вернуть [batch, length] с одним кандидатом на вопрос.")
                for index, sequence in zip(indices, sequences.cpu().tolist()):
                    ids = sequence[1:] if sequence and sequence[0] == bos else sequence
                    ended[index] = eos in ids
                    ids = ids[:ids.index(eos) + 1] if ended[index] else ids
                    lengths[index] = len(ids)
                    hit_limit[index] = not ended[index] and len(ids) >= config.max_new_tokens
                    codes[index] = tokenizer.decode(ids, skip_special_tokens=True)
                batch_stats.append({"batch": len(batch_stats), "num_examples": len(indices), "source_width": width, "seconds": seconds})
    finally:
        model.train(was_training)
    columns = {"generated_code": codes, "source_tokens": original_lengths, "source_truncated": truncated,
               "generated_tokens": lengths, "ended_with_eos": ended, "hit_token_limit": hit_limit}
    predictions = dataset.remove_columns([name for name in columns if name in dataset.column_names])
    for name, values in columns.items():
        predictions = predictions.add_column(name, values)
    total_seconds = perf_counter() - started
    generation_seconds = sum(batch["seconds"] for batch in batch_stats)
    metrics = {"num_examples": len(dataset), "total_seconds": total_seconds, "generation_seconds": generation_seconds,
               "examples_per_second": len(dataset) / total_seconds, "generated_tokens_per_second": sum(lengths) / max(generation_seconds, 1e-12),
               "mean_generated_tokens": sum(lengths) / len(dataset), "source_truncated_rate": sum(truncated) / len(dataset),
               "hit_token_limit_rate": sum(hit_limit) / len(dataset), "eos_rate": sum(ended) / len(dataset),
               "device": str(device), "mixed_precision": precision, "batch_stats": batch_stats}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_json(str(output_dir / "predictions.jsonl"), force_ascii=False)
        (output_dir / "codes.json").write_text(json.dumps(codes, ensure_ascii=False, indent=2), encoding="utf-8")
        for filename, value in (("metrics.json", metrics), ("inference_config.json", config.to_dict())):
            (output_dir / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return InferenceResult(predictions, metrics, output_dir)
