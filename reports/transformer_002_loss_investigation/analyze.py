"""Verify replay comparability and plot the measured loss instability."""
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nlp-hw2-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def load(mode):
    assert (ROOT / f"{mode}.exitcode").read_text().strip() == "0"
    ranks = [[json.loads(line) for line in (ROOT / mode / f"steps-rank{rank}.jsonl").read_text().splitlines()]
             for rank in (0, 1)]
    assert all(len(rows) == 600 for rows in ranks)
    assert all(r["step"] == 50001 + i for rows in ranks for i, r in enumerate(rows))
    for a, b in zip(*ranks):
        assert a["global_target_tokens"] == a["target_tokens"] + b["target_tokens"]
        assert a["global_target_tokens"] == b["global_target_tokens"]
        assert a["model_training"] and b["model_training"]
        assert math.isclose(a["grad_norm_before_clip"], b["grad_norm_before_clip"], rel_tol=1e-6)
        assert all(math.isfinite(r[key]) for r in (a, b) for key in ("loss", "grad_norm_before_clip", "lr"))
    losses = np.mean([[r["loss"] for r in rows] for rows in ranks], axis=0)
    grad = np.array([r["grad_norm_before_clip"] for r in ranks[0]])
    resumed = read(ROOT / mode / "resumed_state.json")
    assert resumed["optimizer_steps"] == [50000] and resumed["max_steps"] == 362284
    return ranks, losses, grad, {
        "start_step": 50001, "stop_step": 50600,
        "max_grad_norm_before_clip": float(grad.max()),
        "max_grad_step": 50001 + int(grad.argmax()),
        "steps_grad_gt_1": int((grad > 1).sum()),
        "steps_grad_gt_3": int((grad > 3).sum()),
        "last_100_steps_loss": float(losses[-100:].mean()),
        "mean_loss_per_100_steps": [float(losses[i:i+100].mean()) for i in range(0, 600, 100)],
        "initial_lr": resumed["lr"][0],
        "final_validation": read(ROOT / mode / "final_validation.json"),
    }


def main():
    modes = ["original_lr", "lr_div3", "fp32_original_lr"]
    data = {mode: load(mode) for mode in modes}
    for mode in modes[1:]:
        for rank in (0, 1):
            assert all(a["input_sha256"] == b["input_sha256"] for a, b in zip(data[modes[0]][0][rank], data[mode][0][rank]))
    history = read(ROOT.parent / "transformer_002_checkpoint50000_test10000_seed42/training/log_history.json")
    original = {r["step"]: r["loss"] for r in history if "loss" in r}
    reference = [original[s] for s in range(50100, 50601, 100)]
    reproduced = data["original_lr"][3]["mean_loss_per_100_steps"]
    error = max(abs(a-b) for a, b in zip(reference, reproduced))
    assert error < 1e-6
    assert len({d[3]["final_validation"]["eval_target_tokens"] for d in data.values()}) == 1
    summary = {
        "modes": {m: d[3] for m, d in data.items()},
        "original_history_loss": reference, "max_reproduction_absolute_error": error,
        "identical_batches_all_600_steps_both_ranks": True,
        "checkpoint": "checkpoint-50000", "fixed_validation_rows": 2048,
        "no_intermediate_eval_or_checkpoint_saving": True,
        "tests": {"ddp_global_token_average_and_resume": "passed", "warmup_cosine": "passed"},
        "limits": "600-step diagnostic runs; no long-run stability or test-quality guarantee. No replay weights saved.",
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.5), layout="constrained")
    colors = {"original_lr": "#c35136", "lr_div3": "#21855b", "fp32_original_lr": "#327ca4"}
    labels = {"original_lr": "BF16 · исходный LR ≈ 2,94e−4",
              "lr_div3": "BF16 · LR / 3 ≈ 9,79e−5", "fp32_original_lr": "FP32 · исходный LR"}
    steps = np.arange(50001, 50601)
    for mode, (_, loss, grad, result) in data.items():
        axes[0, 0].plot(steps[19:], np.convolve(loss, np.ones(20)/20, mode="valid"),
                        color=colors[mode], label=labels[mode], lw=1.7)
        axes[0, 1].plot(steps, grad, color=colors[mode], label=labels[mode], lw=.9, alpha=.85)
    axes[0, 0].set(title="Train loss · скользящее среднее за 20 шагов", ylabel="Cross-entropy")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title="Норма градиента до clipping · каждый шаг", ylabel="Grad norm", yscale="log")
    axes[0, 1].annotate("50 149: 60,62", xy=(50149, 60.62), xytext=(50235, 60),
                       arrowprops={"arrowstyle": "->", "color": "#c35136"}, fontsize=9)
    for ax in axes[0]:
        ax.axvline(50149, color="#777777", ls=":", lw=.9)
        ax.axvline(50208, color="#777777", ls=":", lw=.9)
        ax.set_xlabel("Optimizer step")
    axes[1, 0].plot(range(50100, 50601, 100), reference, "o-", color="#a64530", ms=7, label="Исходное обучение")
    axes[1, 0].plot(range(50100, 50601, 100), reproduced, "x--", color="#20313d", ms=7, label="Диагностическое воспроизведение")
    axes[1, 0].set(title=f"Совпадение с исходным запуском · max Δ = {error:.1e}",
                   ylabel="Train loss за 100 шагов", xlabel="Optimizer step")
    axes[1, 0].legend(fontsize=9)
    values = [data[m][3]["final_validation"]["eval_token_loss"] for m in modes]
    bars = axes[1, 1].bar(["BF16\nисходный LR", "BF16\nLR / 3", "FP32\nисходный LR"], values,
                          color=[colors[m] for m in modes], width=.6)
    axes[1, 1].bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=5)
    axes[1, 1].set(title="Validation loss после 600 шагов", ylabel="Token cross-entropy",
                   ylim=(0, max(values)*1.2))
    for ax in axes.flat:
        ax.grid(alpha=.18, axis="y")
    fig.suptitle("Скачок loss · воспроизведение и контрольные прогоны из checkpoint-50000", fontsize=16)
    fig.supxlabel("Одинаковые веса, optimizer, 2 GPU, global batch 128 и последовательность батчей.\n"
                  "Validation: одна фиксированная выборка 2048 строк. Между шагами нет валидаций и сохранения checkpoint.", fontsize=10)
    fig.savefig(ROOT / "loss_investigation.png", dpi=170, bbox_inches="tight")
    fig.savefig(ROOT / "loss_investigation.svg", bbox_inches="tight")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
