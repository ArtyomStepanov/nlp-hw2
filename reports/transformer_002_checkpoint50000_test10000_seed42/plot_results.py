"""Render saved training logs and evaluation results without running models."""
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nlp-hw2-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 12, "figure.facecolor": "white"})


def save(fig, name):
    fig.savefig(PLOTS / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(PLOTS / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def training_plots():
    history = json.loads((ROOT / "training/log_history.json").read_text())
    state = json.loads((ROOT / "training/trainer_state.json").read_text())
    # The last eval at step 75000 measures restored checkpoint-50000.
    # Keep it out of the curve of weights being updated during training.
    finish = next(i for i, row in enumerate(history) if "train_runtime" in row)
    updates, after_restore = history[:finish], history[finish + 1:]
    train = [r for r in updates if "loss" in r]
    val = [r for r in updates if "eval_token_loss" in r]
    best = min(val, key=lambda r: r["eval_token_loss"])
    assert best["step"] == 50000 and state["global_step"] == 75000
    assert len([r for r in val if r["step"] > best["step"]]) == 5
    assert len(after_restore) == 1
    assert math.isclose(after_restore[0]["eval_token_loss"], best["eval_token_loss"])
    steps = [r["step"] for r in train]
    loss = [r["loss"] for r in train]
    val_steps = [r["step"] for r in val]
    val_loss = [r["eval_token_loss"] for r in val]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4), layout="constrained")
    fig.suptitle("Эксперимент 002 · 8+8 слоёв, hidden 768 · 139,3 млн параметров", fontsize=16)
    for ax, title, zoom in [(axes[0, 0], "Loss за весь запуск · логарифмическая шкала", False),
                            (axes[0, 1], "Скачок loss после лучшего checkpoint", True)]:
        ax.plot(steps, loss, color="#2374a4", lw=1.1, label="Train loss (лог каждые 100 шагов)")
        ax.plot(val_steps, val_loss, "o-", color="#d76527", ms=4, lw=1.7, label="Validation token loss")
        ax.scatter([best["step"]], [best["eval_token_loss"]], marker="*", s=150,
                   color="#208359", zorder=5, label="Лучшие веса: шаг 50 000")
        ax.set(title=title, ylabel="Cross-entropy")
        if zoom:
            ax.set(xlim=(44000, 77000), ylim=(0.2, 1.5))
            ax.annotate("50 300: train loss ≈ 0,912", xy=(50300, 0.91184),
                        xytext=(53500, 1.28), arrowprops={"arrowstyle": "->", "color": "#52616b"}, fontsize=9)
        else:
            ax.set_yscale("log")
            ax.set_ylim(0.23, 11)
            ax.legend(fontsize=8, loc="upper right")
    axes[1, 0].plot(steps, [r["learning_rate"] * 1e4 for r in train], color="#208359", lw=1.6)
    axes[1, 0].set(title="Warmup 5% → cosine · план 362 284 шага",
                   ylabel="Learning rate × 10⁻⁴", ylim=(0, 3.2))
    axes[1, 1].plot(steps, [r["grad_norm"] for r in train], color="#7851a9", lw=0.9, alpha=0.85)
    axes[1, 1].set(title="Норма градиента · точки из лога", ylabel="Grad norm", yscale="log")
    for ax in axes.flat:
        ax.axvline(50000, color="#208359", ls="--", lw=0.9, alpha=0.7)
        ax.axvline(75000, color="#b54444", ls=":", lw=1.1, alpha=0.8)
        ax.set_xlabel("Optimizer step")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
        ax.grid(alpha=0.18)
    fig.supxlabel("Остановка: 75 000 шагов / 0,828 эпохи. После восстановления лучших весов loss снова 0,2825;\n"
                  "эта финальная проверка не включена в кривую обучения. Диагностика скачка loss сохранена отдельным отчётом.",
                  fontsize=10)
    save(fig, "training")
    summary = {"best_step": best["step"], "best_eval_token_loss": best["eval_token_loss"],
               "stop_step": state["global_step"], "stop_epoch": state["epoch"],
               "last_updated_weights_eval_token_loss": val[-1]["eval_token_loss"],
               "restored_best_eval_token_loss": after_restore[0]["eval_token_loss"],
               "early_stopping_patience": 5, "eval_steps": 5000,
               "training_runtime_seconds": history[finish]["train_runtime"],
               "observed_loss_jump_step": 50300,
               "cause_of_loss_jump": "Gradient instability reproduced with original BF16/LR; LR divided by 3 prevents collapse in 600-step control. Exact triggering operation not established.",
               "investigation_report": "../transformer_002_loss_investigation/summary.json",
               "validation_history": val}
    (ROOT / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def wilson(passed, count):
    p, z = passed / count, 1.959963984540054
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [center - half, center + half]


def evaluation_plots():
    if not all((ROOT / mode / "metrics.json").exists() for mode in ("greedy", "beam4")):
        print("Training plots ready; evaluation results not downloaded yet.")
        return
    modes = ["greedy", "beam4"]
    metrics = {mode: json.loads((ROOT / mode / "metrics.json").read_text()) for mode in modes}
    predictions = {mode: [json.loads(line) for line in
                         (ROOT / mode / "evaluation/evaluation.jsonl").read_text().splitlines()]
                   for mode in modes}
    for mode in modes:
        assert [r["sample_row"] for r in predictions[mode]] == list(range(10000))
        metrics[mode]["pass_at_1_wilson95"] = wilson(metrics[mode]["num_passed"], metrics[mode]["num_evaluated"])
    assert [r["test_row_index"] for r in predictions["greedy"]] == [r["test_row_index"] for r in predictions["beam4"]]
    paired = [(g, b) for g, b in zip(predictions["greedy"], predictions["beam4"]) if g["evaluable"] and b["evaluable"]]
    comparison = {"both_evaluable": len(paired),
                  "beam_only_passed": sum(not g["passed"] and b["passed"] for g, b in paired),
                  "greedy_only_passed": sum(g["passed"] and not b["passed"] for g, b in paired),
                  "both_passed": sum(g["passed"] and b["passed"] for g, b in paired),
                  "neither_passed": sum(not g["passed"] and not b["passed"] for g, b in paired),
                  "different_reference_statuses": sum(g["reference_status"] != b["reference_status"]
                                                       for g, b in zip(predictions["greedy"], predictions["beam4"]))}
    summary = {"modes": metrics, "paired_comparison": comparison,
               "evaluation_scope": "10,000 held-out examples from the new 1% test; not the official leaderboard",
               "tolerances": {"rtol": 1e-6, "atol": 1e-6}}
    (ROOT / "quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), layout="constrained")
    colors = ["#2374a4", "#208359"]
    values = [100 * metrics[m]["pass@1"] for m in modes]
    bars = axes[0].bar(["Greedy", "Beam 4"], values, color=colors, width=0.55)
    axes[0].bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=7)
    intervals = [metrics[m]["pass_at_1_wilson95"] for m in modes]
    axes[0].errorbar([0, 1], values, yerr=[[values[i]-100*intervals[i][0] for i in range(2)],
                                        [100*intervals[i][1]-values[i] for i in range(2)]],
                     fmt="none", color="#253846", capsize=4)
    axes[0].set(title="Локальный pass@1 · 95% Wilson CI", ylim=(0, 100), ylabel="% исполнимых эталонов")
    bad = [100 * metrics[m]["bad_code_rate"] for m in modes]
    bars = axes[1].bar(["Greedy", "Beam 4"], bad, color=colors, width=0.55)
    axes[1].bar_label(bars, labels=[f"{v:.2f}%" for v in bad], padding=4)
    axes[1].set(title="Код с ошибками выполнения", ylim=(0, max(5, max(bad) * 1.3)), ylabel="% всех 10 000 ответов")
    for mode, color in zip(modes, colors):
        axes[2].hist([r["generated_tokens"] for r in predictions[mode]], bins=range(0, 273, 16),
                     histtype="step", color=color, lw=1.8, label="Greedy" if mode == "greedy" else "Beam 4")
    axes[2].set(title="Длины ответов", xlabel="Токены, включая EOS", ylabel="Примеры")
    axes[2].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Checkpoint-50000 · новый test 98/1/1 · 10 000 строк, seed 42", fontsize=15)
    fig.supxlabel("Pass@1: совпадение результата solve() с исполнимым эталоном. Локальный ограниченный evaluator;\n"
                  "результат отличается по протоколу от публичного leaderboard. Оба режима: FP16, max_new_tokens=256.", fontsize=9)
    save(fig, "quality")
    print(json.dumps({"pass_at_1": {m: metrics[m]["pass@1"] for m in modes}, "paired": comparison}, ensure_ascii=False))


if __name__ == "__main__":
    training_plots()
    evaluation_plots()
