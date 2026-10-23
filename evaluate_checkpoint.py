"""Оценить сохранённый checkpoint на воспроизводимой выборке held-out test.

CLI и evaluate_checkpoint.ipynb используют один runner. Только локальные данные,
веса и готовый BPE; обучение и скачивание с Hub здесь не запускаются.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter


SPLIT_SIZES = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_SEED = 42  # Должен совпадать с разбиением предыдущего обучения.
DEFAULT_CHECKPOINT = Path("artifacts/experiments/baseline_001/checkpoint-147871")
DEFAULT_RESULTS_ROOT = Path("artifacts/evaluations")


@dataclass
class CheckpointEvaluationConfig:
    checkpoint: str | Path = DEFAULT_CHECKPOINT
    data_dir: str | Path | None = None
    output_dir: str | Path | None = None
    tokenizer_path: str | Path | None = None
    sample_size: int = 10_000
    seed: int = 42  # Только выборка внутри уже зафиксированной test-части.
    batch_size: int = 64
    max_source_length: int = 256
    max_new_tokens: int = 256
    precision: str = "fp16"
    device: str = "cuda:0"
    workers: int = 8
    timeout_seconds: float = 2.0
    clean_num_proc: int = 1

    def to_dict(self):
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self).items()}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def _has_reference_code(dataset):
    """Есть ли хотя бы один не-null code, включая пустую строку.

    Parquet loader объединяет схемы train/test и добавляет официальному
    test полностью null-столбец code. Null count читает метаданные Arrow;
    при отборе по индексам проверяем только видимые строки, порциями.
    """
    if "code" not in dataset.column_names or not len(dataset):
        return False
    if dataset._indices is None:
        column = dataset.data.column("code")
        return column.null_count < len(column)
    codes = dataset.select_columns(["code"]).with_format(None)
    return any(value is not None for batch in codes.iter(batch_size=10_000) for value in batch["code"])


def load_local_dataset(data_dir=None):
    """Прочитать локальный snapshot или сохранённый ИСХОДНЫЙ DatasetDict.

    Уже разбитый train/val/test запрещён: повторное разбиение его train
    выбрало бы для оценки строки, на которых модель обучалась.
    """
    import datasets as ds
    from load_dataset import load_dataset_from_files

    if data_dir is None:
        candidates = [Path("data/tinysms_qa_12m_ssh"), Path("data/tinysms_qa_12m")]
        data_dir = next((path for path in candidates if path.is_dir()), None)
        if data_dir is None:
            raise FileNotFoundError("Укажите --data-dir с локальным tinysms_qa_12m или DatasetDict.")
    path = Path(data_dir).expanduser().resolve()
    data = ds.load_from_disk(str(path)) if (path / "dataset_dict.json").is_file() else load_dataset_from_files(path)
    if not isinstance(data, ds.DatasetDict) or "train" not in data:
        raise ValueError("Ожидается исходный DatasetDict с train, не заранее отфильтрованная часть.")
    if ("val" in data or "validation" in data or
            ("test" in data and _has_reference_code(data["test"]))):
        raise ValueError("DatasetDict уже разбит на train/val/test. Передайте исходный raw DatasetDict "
                         "с полным train и официальным test без эталонов code (нет столбца или все значения null); "
                         "повторное разбиение train запрещено.")
    if not {"question", "code"}.issubset(data["train"].column_names):
        raise ValueError("Исходный train должен содержать question и code.")
    return data, path


def select_heldout_sample(raw_train, *, sample_size=10_000, seed=42, clean_num_proc=1):
    """Повторить разбиение обучения, выбрать test, затем очистить только sample.

    Дополнительный source_row_index не меняет перестановку Dataset.shuffle:
    это исходная позиция строки в raw['train'], до разбиения и очистки.
    Изменение seed меняет только выборку внутри test; split_seed всегда 42.
    """
    import numpy as np
    from prepare_datasets import remove_dataset_comments, split_dataset

    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("sample_size должен быть положительным целым числом.")
    fingerprint = raw_train._fingerprint
    indexed = raw_train.select_columns(["question", "code"]).add_column(
        "source_row_index", np.arange(len(raw_train), dtype=np.int64))
    parts = split_dataset(indexed, sizes=SPLIT_SIZES, seed=SPLIT_SEED,
                          columns=["question", "code", "source_row_index"])
    heldout = parts["test"]
    if sample_size > len(heldout):
        raise ValueError(f"Запрошено {sample_size} строк, но held-out test содержит {len(heldout)}.")
    sample = heldout.shuffle(seed=seed).select(range(sample_size))
    sample = sample.add_column("sample_row", list(range(sample_size)))
    # Никакой фильтрации по source/target длине, исполнению или содержимому ответа.
    sample = remove_dataset_comments(sample, columns=["code"],
                                     num_proc=None if clean_num_proc == 1 else clean_num_proc)
    manifest = {
        "raw_train_size": len(raw_train), "raw_train_fingerprint": fingerprint,
        "split_sizes": SPLIT_SIZES, "split_seed": SPLIT_SEED,
        "split_counts": {name: len(part) for name, part in parts.items()},
        "heldout_split": "test", "heldout_size": len(heldout),
        "sample_size": len(sample), "sample_seed": seed,
        "sample_fraction_of_heldout": len(sample) / len(heldout),
        "selection": "split(seed=42)['test'].shuffle(seed=sample_seed).select(range(sample_size))",
        "source_row_index": "zero-based position in the original local raw train",
        "sample_row": "generation and evaluation outputs preserve this zero-based sample order",
        "comment_removal": "only sampled reference code; no rows removed",
        "source_length_policy": "truncate source with EOS preserved; record source_truncated",
        "target_length_policy": "no target-length filtering or reference truncation",
    }
    return sample, manifest


def run_evaluation(config: CheckpointEvaluationConfig) -> Path:
    """Выполнить один запуск и вернуть новую папку с результатами и графиками."""
    checkpoint = Path(config.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Локальный checkpoint не найден: {checkpoint}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (Path(config.output_dir).expanduser() if config.output_dir is not None else
              DEFAULT_RESULTS_ROOT / f"{checkpoint.name}_test{config.sample_size}_seed{config.seed}_{timestamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    run = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
           "config": config.to_dict(), "checkpoint": str(checkpoint)}
    _write_json(output / "run.json", run)
    _log(f"Результаты: {output}")
    try:
        return _evaluate_into(config, checkpoint, output, run, started)
    except Exception as error:
        run.update(status="failed", failed_at=datetime.now(timezone.utc).isoformat(),
                   error_type=type(error).__name__, error=str(error),
                   run_seconds=perf_counter() - started)
        _write_json(output / "run.json", run)
        _log(f"Оценка завершилась с ошибкой {type(error).__name__}: {error}")
        raise


def _evaluate_into(config, checkpoint, output, run, started):
    _log("Загрузка локальных данных и воспроизведение разбиения 80/10/10, split seed=42")
    raw, data_path = load_local_dataset(config.data_dir)
    sample, manifest = select_heldout_sample(raw["train"], sample_size=config.sample_size,
                                            seed=config.seed, clean_num_proc=config.clean_num_proc)
    manifest["data_dir"] = str(data_path)
    manifest["official_test_size"] = len(raw["test"]) if "test" in raw else None
    manifest["official_test_used"] = False
    _write_json(output / "sample_manifest.json", manifest)
    sample.save_to_disk(str(output / "sample"))
    # Не удерживаем полное исходное разбиение во время генерации.
    del raw
    _log(f"Сохранено {len(sample)} строк из held-out test ({manifest['heldout_size']} строк)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evaluation import EvaluationConfig, evaluate_code
    from inference import InferenceConfig, generate_code, load_inference_model
    from plotting import plot_evaluation, plot_training_history

    _log(f"Загрузка checkpoint и готового BPE: {checkpoint}")
    model, tokenizer = load_inference_model(checkpoint, tokenizer_path=config.tokenizer_path,
                                           device=config.device)
    _log(f"Генерация: {config.device}, {config.precision}, batch_size={config.batch_size}")
    generated = generate_code(sample, model, InferenceConfig(
        tokenizer=tokenizer, batch_size=config.batch_size,
        max_source_length=config.max_source_length, max_new_tokens=config.max_new_tokens,
        num_beams=1, sort_by_length=True, truncate_source=True,
        mixed_precision=config.precision, device=config.device, output_dir=output / "inference"))
    if list(generated.predictions["sample_row"]) != list(range(len(sample))):
        raise RuntimeError("Генерация нарушила исходный порядок выборки.")
    _log(f"Генерация завершена за {generated.metrics['total_seconds']:.1f} с; сравнение solve(), workers={config.workers}")
    evaluated = evaluate_code(generated, EvaluationConfig(
        workers=config.workers, timeout_seconds=config.timeout_seconds,
        output_dir=output / "evaluation"))
    if list(evaluated.predictions["sample_row"]) != list(range(len(sample))):
        raise RuntimeError("Оценка нарушила исходный порядок выборки.")
    metrics = dict(evaluated.metrics)
    metrics.update({
        "checkpoint": str(checkpoint), "sample_size": len(sample),
        "split_seed": SPLIT_SEED, "sample_seed": config.seed,
        "heldout_size": manifest["heldout_size"],
        "sample_fraction_of_heldout": manifest["sample_fraction_of_heldout"],
        "num_reference_errors": len(sample) - metrics["num_evaluated"],
        "passed_fraction_of_all_sampled": metrics["num_passed"] / len(sample),
        "num_source_truncated": sum(generated.predictions["source_truncated"]),
        "num_hit_token_limit": sum(generated.predictions["hit_token_limit"]),
        "source_length_policy": manifest["source_length_policy"],
        "target_length_policy": manifest["target_length_policy"],
        "pass_at_1_definition": "num_passed / num_evaluated; reference solve() must execute successfully",
        "official_test_used": False,
    })
    evaluated.metrics = metrics
    figure = plot_evaluation(evaluated, save_path=output / "plots" / "evaluation.png")
    plt.close(figure)
    history_paths = [checkpoint / "trainer_state.json", checkpoint.parent / "log_history.json"]
    history = []
    for path in history_paths:
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            history = value.get("log_history", []) if isinstance(value, dict) else value
            if history:
                break
    metrics["training_history_available"] = bool(history)
    if history:
        _write_json(output / "training_history.json", history)
        figure = plot_training_history(history, save_path=output / "plots" / "training.png")
        plt.close(figure)
    metrics["run_seconds"] = perf_counter() - started
    _write_json(output / "metrics.json", metrics)
    run.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
               output_dir=str(output), run_seconds=metrics["run_seconds"])
    _write_json(output / "run.json", run)
    _log(f"Оценка сохранена: pass@1={metrics['pass@1']}, bad_code_rate={metrics['bad_code_rate']:.4f}, coverage={metrics['evaluation_coverage']:.4f}")
    return output


def _checkpoint_identity(checkpoint):
    path = Path(checkpoint).expanduser().resolve()
    # Сохранённые результаты можно перенести с SSH, сохранив путь эксперимента.
    # Учитываем baseline_001 и все подпапки, а не только checkpoint-N.
    for index in range(len(path.parts) - 1):
        if path.parts[index:index + 2] == ("artifacts", "experiments"):
            return Path(*path.parts[index:]).as_posix()
    return str(path)


def find_latest_result(*, checkpoint=DEFAULT_CHECKPOINT, sample_size=10_000, seed=42,
                       results_root=DEFAULT_RESULTS_ROOT):
    """Найти завершённый запуск для параметров ноутбука без повторной генерации."""
    matches = []
    for manifest_path in Path(results_root).glob("*/run.json"):
        run = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = run.get("config", {})
        if (run.get("status") == "complete" and
                _checkpoint_identity(run["checkpoint"]) == _checkpoint_identity(checkpoint) and
                config.get("sample_size") == sample_size and config.get("seed") == seed):
            matches.append((run.get("completed_at", ""), manifest_path.parent))
    return max(matches, key=lambda item: item[0])[1] if matches else None


def load_saved_result(result_dir):
    """Читать готовую оценку; solve() и модель повторно не запускаются."""
    import datasets as ds
    from evaluation import EvaluationResult

    path = Path(result_dir).expanduser()
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    predictions = ds.Dataset.from_json(str(path / "evaluation" / "evaluation.jsonl"))
    return EvaluationResult(predictions=predictions, metrics=metrics, output_dir=path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, help="Локальный Parquet snapshot или сохранённый DatasetDict")
    parser.add_argument("--output-dir", type=Path, help="Новая папка; по умолчанию уникальная artifacts/evaluations/...")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42, help="Seed выборки; seed разбиения обучения зафиксирован 42")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--precision", choices=["auto", "no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--clean-num-proc", type=int, default=1)
    args = parser.parse_args()
    run_evaluation(CheckpointEvaluationConfig(**vars(args)))


if __name__ == "__main__":
    main()
