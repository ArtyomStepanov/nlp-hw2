"""Локальный pass@1 по результату solve(), отдельно от доли сломанного кода."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import datasets as ds


@dataclass
class EvaluationConfig:
    prediction_column: str = "generated_code"
    reference_column: str = "code"
    timeout_seconds: float = 2.0
    workers: int = 4
    memory_limit_mb: int = 256
    max_code_bytes: int = 65536
    relative_tolerance: float = 1e-6
    absolute_tolerance: float = 1e-8
    output_dir: str | Path | None = None

    def __post_init__(self) -> None:
        for name in ("timeout_seconds", "relative_tolerance", "absolute_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0 or (name == "timeout_seconds" and value == 0):
                raise ValueError(f"Некорректное значение {name}.")
        for name in ("workers", "memory_limit_mb", "max_code_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} должен быть положительным целым числом.")
        for name in ("prediction_column", "reference_column"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} должен быть непустой строкой.")
        if self.prediction_column == self.reference_column:
            raise ValueError("Столбцы предсказания и эталона должны различаться.")

    def to_dict(self) -> dict:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        values["output_dir"] = str(self.output_dir) if self.output_dir is not None else None
        return values


@dataclass
class EvaluationResult:
    predictions: ds.Dataset
    metrics: dict
    output_dir: Path | None = None


def _execute(code, config: EvaluationConfig) -> dict:
    if code is None:
        code = ""
    if not isinstance(code, str):
        return {"status": "unsupported_code", "error": "Код должен быть строкой.", "value": None}
    if len(code.encode("utf-8")) > config.max_code_bytes:
        return {"status": "unsupported_code", "error": "Превышен max_code_bytes.", "value": None}
    request = {"code": code, "cpu_seconds": max(1, math.ceil(config.timeout_seconds)), "memory_bytes": config.memory_limit_mb * 1024**2}
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(__file__).with_name("_code_runner.py"))],
            input=json.dumps(request), capture_output=True, text=True,
            timeout=config.timeout_seconds, cwd=Path(__file__).parent,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"Лимит {config.timeout_seconds:g} с.", "value": None}
    if process.returncode:
        # SIGXCPU/SIGKILL от CPU-лимита; остальные аварии — runtime_error.
        status = "timeout" if process.returncode in {-9, -24} else "runtime_error"
        return {"status": status, "error": f"Процесс завершился с кодом {process.returncode}.", "value": None}
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("Evaluator вернул некорректный JSON.") from None
    result.setdefault("value", None)
    return result


def _equal(actual, expected, config: EvaluationConfig) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if isinstance(actual, int) and isinstance(expected, int):
            return actual == expected
        try:
            return math.isclose(actual, expected, rel_tol=config.relative_tolerance, abs_tol=config.absolute_tolerance)
        except OverflowError:
            return actual == expected
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_equal(a, b, config) for a, b in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_equal(actual[key], expected[key], config) for key in actual)
    return type(actual) is type(expected) and actual == expected


def evaluate_code(predictions, config: EvaluationConfig | None = None) -> EvaluationResult:
    """Оценить Dataset с generated_code или результат generate_code.

    Каждый кандидат и эталон выполняется в отдельном ограниченном процессе.
    pass@1 = доля совпавших результатов solve() среди строк с исполнимым
    эталоном. Ошибочные эталоны явно исключены, reference_error_rate показывает
    их долю. bad_code_rate считается по ВСЕМ предсказаниям; wrong_answer_rate
    — по строкам с исполнимым эталоном. Если code отсутствует (официальный test),
    pass@1 и wrong_answer_rate равны None, доступна только диагностика кода.

    Поддерживается арифметический Python: функции, циклы, коллекции, безопасные
    builtins и math. Файлы, сеть, introspection и прочие imports недоступны:
    они получают отдельный статус unsupported_code. Это локальное сравнение
    с эталонным solve(), а не воспроизведение скрытого checker соревнования.
    """
    config = EvaluationConfig() if config is None else config
    inference_metrics = dict(predictions.metrics) if hasattr(predictions, "metrics") else {}
    dataset = predictions.predictions if hasattr(predictions, "predictions") else predictions
    if not isinstance(dataset, ds.Dataset) or not len(dataset):
        raise ValueError("Передайте непустой Dataset или InferenceResult.")
    if config.prediction_column not in dataset.column_names:
        raise ValueError(f"Нет столбца {config.prediction_column!r}.")
    output_dir = Path(config.output_dir).expanduser() if config.output_dir is not None else None
    if output_dir is not None and output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_dir должен быть пустым, чтобы не перезаписать оценку.")
    has_references = config.reference_column in dataset.column_names
    started = perf_counter()

    def evaluate_row(row):
        generated = _execute(row[config.prediction_column], config)
        reference = _execute(row[config.reference_column], config) if has_references else {"status": "not_available", "value": None, "error": None}
        eligible = reference["status"] == "ok"
        passed = eligible and generated["status"] == "ok" and _equal(generated["value"], reference["value"], config)
        status = generated["status"]
        if status == "ok" and eligible:
            status = "passed" if passed else "wrong_answer"
        return {"status": status, "execution_status": generated["status"], "error": generated.get("error"),
                "result_json": json.dumps(generated["value"], ensure_ascii=False) if generated["status"] == "ok" else None,
                "reference_status": reference["status"], "reference_error": reference.get("error"),
                "reference_result_json": json.dumps(reference["value"], ensure_ascii=False) if eligible else None,
                "passed": passed if eligible else None, "evaluable": eligible}

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        records = list(executor.map(evaluate_row, dataset))
    statuses = Counter(record["status"] for record in records)
    reference_statuses = Counter(record["reference_status"] for record in records)
    evaluated = sum(record["evaluable"] for record in records)
    passed = sum(record["passed"] is True for record in records)
    bad_code = sum(record["execution_status"] != "ok" for record in records)
    metrics = {**inference_metrics, "num_examples": len(dataset), "num_evaluated": evaluated, "num_passed": passed,
               "num_bad_code": bad_code, "pass@1": passed / evaluated if evaluated else None,
               "bad_code_rate": bad_code / len(dataset), "wrong_answer_rate": statuses["wrong_answer"] / evaluated if evaluated else None,
               "reference_error_rate": (len(dataset) - evaluated) / len(dataset) if has_references else None,
               "evaluation_coverage": evaluated / len(dataset), "evaluation_seconds": perf_counter() - started,
               "status_counts": dict(statuses), "reference_status_counts": dict(reference_statuses),
               "prediction_column": config.prediction_column, "reference_column": config.reference_column,
               "comparison": "reference_solve" if has_references else "code_health_only"}
    result_dataset = dataset.remove_columns([name for name in records[0] if name in dataset.column_names])
    for name in records[0]:
        result_dataset = result_dataset.add_column(name, [record[name] for record in records])
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        result_dataset.to_json(str(output_dir / "evaluation.jsonl"), force_ascii=False)
        for filename, value in (("metrics.json", metrics), ("evaluation_config.json", config.to_dict())):
            (output_dir / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return EvaluationResult(result_dataset, metrics, output_dir)
