"""Plot the successful training trajectory and inspect its final plateau.

Use saved logs only. The failed original branch after 50k and the final
evaluation of restored best weights are not part of this training curve.
"""
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nlp-hw2-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "nlp-hw2-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
TRAIN_COLOR = "#2374a4"
VAL_COLOR = "#d76527"
BEST_COLOR = "#208359"


def moving_mean(records, window=20):
    """Trailing mean; restart at resume so smoothing cannot bridge the LR change."""
    buffer = deque(maxlen=window)
    steps, values = [], []
    for row in records:
        if row["step"] == 50100:
            buffer.clear()
        buffer.append(row["loss"])
        if len(buffer) == window:
            steps.append(row["step"])
            values.append(sum(buffer) / len(buffer))
    return steps, values


def format_axis(ax, ylabel="Loss на токен"):
    ax.set_xlabel("Шаг оптимизатора")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
    ax.grid(alpha=0.18)


def save(fig, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(PLOTS / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    PLOTS.mkdir(exist_ok=True)
    history_path = ROOT / "final/log_history.json"
    history = json.loads(history_path.read_text())
    state = json.loads((ROOT / "final/trainer_state.json").read_text())
    end = next(i for i, row in enumerate(history) if "train_runtime" in row)
    updates, restored = history[:end], history[end + 1:]
    train = [r for r in updates if "loss" in r]
    val = [r for r in updates if "eval_token_loss" in r]
    resumed_train = [r for r in train if r["step"] > 50000]
    resumed_val = [r for r in val if r["step"] >= 50000]
    best = min(val, key=lambda r: r["eval_token_loss"])
    assert best["step"] == 360000 and state["global_step"] == 362284 and state["epoch"] == 4.0
    assert state["best_metric"] == best["eval_token_loss"]
    assert len(restored) == 1 and restored[0]["eval_token_loss"] == best["eval_token_loss"]
    assert all(b["step"] > a["step"] for a, b in zip(val, val[1:]))
    assert all(b["step"] > a["step"] for a, b in zip(train, train[1:]))
    assert all(b["step"] - a["step"] == 100 for a, b in zip(resumed_train, resumed_train[1:]))
    by_step = {r["step"]: r for r in val}
    train_by_step = {r["step"]: r for r in train}
    smooth_steps, smooth_loss = moving_mean(resumed_train)
    changes = {}
    for start in (50000, 90000, 300000, 330000, 350000):
        before, after = by_step[start]["eval_token_loss"], best["eval_token_loss"]
        changes[f"{start}_to_360000"] = {
            "from_step": start, "to_step": 360000, "from_loss": before, "to_loss": after,
            "absolute_improvement": before - after,
            "relative_improvement_percent": 100 * (before - after) / before,
        }
    summary = {
        "source": str(history_path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(history_path.read_bytes()).hexdigest(),
        "best_step": best["step"], "best_eval_token_loss": best["eval_token_loss"],
        "finished_step": state["global_step"], "finished_epoch": state["epoch"],
        "stopped_by": "planned four epochs; early stopping did not trigger",
        "early_stopping_counter": state["stateful_callbacks"]["EarlyStoppingCallback"]["attributes"]["early_stopping_patience_counter"],
        "num_train_log_points": len(train), "num_validation_points": len(val),
        "num_continuation_validation_points": len(resumed_val) - 1,
        "restored_best_eval_excluded_from_curve": restored[0],
        "last_updated_weights_eval_token_loss": val[-1]["eval_token_loss"],
        "improvement_intervals": changes,
        "logged_learning_rates": {str(step): train_by_step[step]["learning_rate"]
                                  for step in (50100, 300000, 330000, 350000, 360000, 362200)},
        "train_smoothing": "Trailing arithmetic mean of 20 logged values (~2000 steps); resets at resume",
        "train_val_comparison_note": "Train averages optimization minibatches with dropout; validation uses the fixed full split in eval mode. These are not matched-dataset loss measurements.",
        "interpretation": "The final portion is nearly flat under the decayed learning rate. There is no sustained validation-loss rise. This does not establish an optimum under a different training schedule.",
    }
    (ROOT / "convergence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "figure.facecolor": "white"})

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle("Train и validation · финальный запуск 8+8 слоёв, d=768", fontsize=17)
    for ax, left, title, ylim in (
        (axes[0, 0], 50000, "Продолжение с 50k: loss постепенно снижается", (0.178, 0.300)),
        (axes[0, 1], 300000, "Последние 62k шагов: train и validation", (0.181, 0.198)),
    ):
        ax.plot([r["step"] for r in resumed_train], [r["loss"] for r in resumed_train],
                color=TRAIN_COLOR, alpha=0.20, lw=0.7, label="Train: лог каждые 100 шагов")
        ax.plot(smooth_steps, smooth_loss, color=TRAIN_COLOR, lw=1.6,
                label="Train: среднее по 2000 шагам")
        ax.plot([r["step"] for r in resumed_val], [r["eval_token_loss"] for r in resumed_val],
                color=VAL_COLOR, marker="o", ms=3, lw=1.7, label="Validation: все 118 302 примера")
        ax.scatter([best["step"]], [best["eval_token_loss"]], marker="*", s=150,
                   color=BEST_COLOR, zorder=5)
        ax.set(title=title, xlim=(left, 365000), ylim=ylim)
        format_axis(ax)
    axes[0, 0].legend(fontsize=8.5, loc="upper right")

    tail = [r for r in val if r["step"] >= 300000]
    ax = axes[1, 0]
    ax.plot([r["step"] for r in tail], [r["eval_token_loss"] for r in tail],
            "o-", color=VAL_COLOR, ms=4, lw=1.8)
    ax.scatter([best["step"]], [best["eval_token_loss"]], marker="*", s=170,
               color=BEST_COLOR, zorder=5, label="Лучший checkpoint: 360k")
    ax.axvspan(350000, 362284, color=BEST_COLOR, alpha=0.07)
    ax.set(title="Validation крупнее: к концу почти плато", xlim=(298000, 365000),
           ylim=(0.19261, 0.19416))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.5f}"))
    ax.text(0.97, 0.94, "350k → 360k\n0,1927118 → 0,1927033\nУлучшение: 0,0044%", ha="right", va="top",
            transform=ax.transAxes, fontsize=10, bbox={"boxstyle": "round,pad=0.5", "fc": "#f2f6f3", "ec": "#d2e0d7"})
    ax.legend(loc="lower left", fontsize=9)
    format_axis(ax, "Validation loss на токен")

    ax = axes[1, 1]
    ax.plot([r["step"] for r in resumed_train], [r["learning_rate"] for r in resumed_train],
            color=BEST_COLOR, lw=1.8)
    ax.set(title="Learning rate: cosine почти обнулил шаг", xlim=(50000, 365000), yscale="log")
    for step in (300000, 350000, 360000):
        ax.scatter([step], [train_by_step[step]["learning_rate"]], color=BEST_COLOR, s=25, zorder=4)
    ax.text(0.05, 0.08, "300k: 7,87 × 10⁻⁶\n350k: 3,14 × 10⁻⁷\n360k: 1,09 × 10⁻⁸", va="bottom",
            transform=ax.transAxes, fontsize=10)
    format_axis(ax, "Learning rate · логарифмическая шкала")
    fig.supxlabel("Логи фиксированного запуска; обучение завершилось по плану на 362 284 шагах / 4 эпохах.\n"
                  "Повторная оценка после восстановления checkpoint-360000 не включена в кривую; train и validation считаются в разных режимах.", fontsize=9)
    save(fig, "training_validation")

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True, layout="constrained",
                              gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Вся успешная траектория: 0–50k + продолжение до 4 эпох", fontsize=16)
    ax = axes[0]
    full_steps, full_smooth = moving_mean(train)
    ax.plot([r["step"] for r in train], [r["loss"] for r in train], color=TRAIN_COLOR, alpha=0.20, lw=0.6)
    ax.plot(full_steps, full_smooth, color=TRAIN_COLOR, lw=1.3, label="Train: среднее по 20 точкам")
    ax.plot([r["step"] for r in val], [r["eval_token_loss"] for r in val], "o-", color=VAL_COLOR,
            ms=3, lw=1.6, label="Validation: полный фиксированный split")
    ax.scatter([best["step"]], [best["eval_token_loss"]], color=BEST_COLOR, marker="*", s=130, zorder=4)
    ax.set(yscale="log", ylim=(0.17, 10), title="Train и validation loss · логарифмическая шкала")
    ax.legend(fontsize=9)
    ax.text(0.19, 0.90, "50k: продолжение\nс уменьшением LR втрое", transform=ax.transAxes, fontsize=9)
    axes[1].plot([r["step"] for r in train], [r["learning_rate"] for r in train], color=BEST_COLOR, lw=1.6)
    axes[1].set(title="Warmup → cosine; в точке продолжения LR уменьшен втрое", ylim=(0, 0.00032))
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in axes:
        ax.axvline(50000, color="#64748b", linestyle="--", lw=1)
        ax.set_xlim(0, 365000)
        format_axis(ax, "Loss на токен" if ax is axes[0] else "Learning rate")
    fig.supxlabel("Неудачная исходная ветка 50k–75k исключена: эти веса были отброшены при возобновлении с checkpoint-50000.\n"
                  "Точка повторной проверки лучших весов после окончания обучения также исключена.", fontsize=9)
    save(fig, "training_full_history")
    print(json.dumps({"best_step": best["step"], "best_loss": best["eval_token_loss"],
                      "last_10k_improvement": changes["350000_to_360000"],
                      "plots": ["training_validation", "training_full_history"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
