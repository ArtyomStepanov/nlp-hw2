"""Графики обучения, качества генерации и сравнения экспериментов."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig, save_path):
    if save_path is not None:
        path = Path(save_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_history(history, *, save_path: str | Path | None = None):
    """Принять TrainingResult, Trainer, список логов или путь к папке запуска.

    Отобразить train loss, eval_token_loss (при наличии, иначе eval_loss) и
    learning rate. Отсутствующие записи не заменяются нулями. Вернуть Figure;
    в ноутбуке достаточно оставить fig последним выражением или plt.show().
    """
    if isinstance(history, (str, Path)):
        path = Path(history).expanduser()
        if path.is_dir():
            path = path / "log_history.json"
        history = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(history, dict):
            history = history.get("log_history", [])  # trainer_state.json
    elif hasattr(history, "trainer"):
        history = history.trainer.state.log_history
    elif hasattr(history, "state"):
        history = history.state.log_history
    if not isinstance(history, list) or not history:
        raise ValueError("Нужен непустой log_history Trainer.")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    validation_metric = "eval_token_loss" if any("eval_token_loss" in row for row in history) else "eval_loss"
    for metric, label, color in (("loss", "Train loss", "#247BA0"), (validation_metric, validation_metric, "#F25F5C")):
        points = [(row.get("step", row.get("epoch", index)), row[metric]) for index, row in enumerate(history) if isinstance(row.get(metric), (float, int))]
        if points:
            x, y = zip(*points)
            axes[0].plot(x, y, marker="o", markersize=3, label=label, color=color)
    lr = [(row.get("step", row.get("epoch", index)), row["learning_rate"]) for index, row in enumerate(history) if isinstance(row.get("learning_rate"), (float, int))]
    if lr:
        x, y = zip(*lr)
        axes[1].plot(x, y, color="#70A288")
    else:
        axes[1].text(0.5, 0.5, "Learning rate was not logged", ha="center", va="center", transform=axes[1].transAxes)
    axes[0].set(title="Training and validation", xlabel="Step / epoch", ylabel="Cross-entropy")
    if axes[0].lines:
        axes[0].legend()
    axes[1].set(title="Learning rate", xlabel="Step / epoch", ylabel="Learning rate")
    for ax in axes:
        ax.grid(alpha=0.2)
    return _save(fig, save_path)


def plot_evaluation(result, *, save_path: str | Path | None = None):
    """Принять EvaluationResult: метрики, статусы, длины и время батчей.

    Без эталонов pass@1 не рисуется. Для оценки сохранённого результата можно
    повторно вызвать evaluate_code на Dataset из evaluation.jsonl.
    """
    if not hasattr(result, "metrics") or not hasattr(result, "predictions"):
        raise TypeError("Передайте EvaluationResult из evaluate_code.")
    metrics, predictions = result.metrics, result.predictions
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    pairs = [(label, metrics.get(key)) for label, key in (("pass@1", "pass@1"), ("Bad code", "bad_code_rate"), ("Wrong answer", "wrong_answer_rate"))]
    pairs = [(label, value) for label, value in pairs if value is not None]
    if pairs:
        labels, values = zip(*pairs)
        bars = ax.bar(labels, values, color=["#247BA0", "#F25F5C", "#F4B942"][:len(values)])
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
    ax.set(title=f"Quality ({metrics.get('num_evaluated', 0)} labelled examples)", ylabel="Fraction", ylim=(0, 1.1))
    counts = metrics.get("status_counts", {})
    ax = axes[0, 1]
    if counts:
        names = list(counts)
        ax.barh(names, [counts[name] for name in names], color="#70A288")
        ax.invert_yaxis()
    ax.set(title="Prediction statuses", xlabel="Examples")
    ax = axes[1, 0]
    if "generated_tokens" in predictions.column_names:
        values = list(predictions["generated_tokens"])
        ax.hist(values, bins=min(30, max(1, len(set(values)))), color="#247BA0", alpha=0.8)
        ax.set(xlabel="Generated tokens (including EOS)")
    else:
        column = metrics.get("prediction_column", "generated_code")
        ax.hist([len(code) if isinstance(code, str) else 0 for code in predictions[column]], bins=20, color="#247BA0", alpha=0.8)
        ax.set(xlabel="Generated code characters")
    ax.set(title="Generation lengths", ylabel="Examples")
    ax = axes[1, 1]
    batches = metrics.get("batch_stats", [])
    if batches:
        ax.plot([row["batch"] for row in batches], [row["seconds"] for row in batches], marker="o", markersize=3, color="#F25F5C")
        ax.set(xlabel="Batch", ylabel="Seconds")
        ax.set_title(f"Generation: {metrics.get('total_seconds', 0):.2f}s total")
    else:
        ax.text(0.5, 0.5, "No generation timing recorded", ha="center", va="center", transform=ax.transAxes)
        ax.set(title="Generation timing")
    for ax in axes.flat:
        ax.grid(alpha=0.15, axis="y")
    return _save(fig, save_path)


def plot_experiment_comparison(experiments: dict, *, save_path: str | Path | None = None):
    """Сравнить {название: result / metrics / путь к metrics.json}.

    Сравнивайте качество на одной и той же отложенной части, время — на одном
    устройстве и наборе вопросов. Пропущенные метрики остаются пропущенными.
    """
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError("Нужен непустой словарь {название: метрики}. ")
    runs = {}
    for name, value in experiments.items():
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser()
            value = json.loads((path / "metrics.json" if path.is_dir() else path).read_text(encoding="utf-8"))
        elif hasattr(value, "metrics"):
            value = value.metrics
        if not isinstance(value, dict):
            raise TypeError(f"{name}: нужны result, dict с метриками или путь к metrics.json.")
        runs[name] = value
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(len(runs))
    for offset, (key, label, color) in enumerate((("pass@1", "pass@1", "#247BA0"), ("bad_code_rate", "Bad code", "#F25F5C"))):
        values = [np.nan if run.get(key) is None else run[key] for run in runs.values()]
        axes[0].bar(x + (offset - 0.5) * 0.35, values, width=0.35, label=label, color=color)
    axes[0].set(title="Quality comparison", ylabel="Fraction", ylim=(0, 1.05))
    axes[0].legend()
    times = [np.nan if run.get("total_seconds") is None else run["total_seconds"] for run in runs.values()]
    axes[1].bar(x, times, color="#70A288")
    axes[1].set(title="End-to-end inference time", ylabel="Seconds")
    for ax in axes:
        ax.set_xticks(x, list(runs), rotation=20, ha="right")
        ax.grid(alpha=0.2, axis="y")
    return _save(fig, save_path)
