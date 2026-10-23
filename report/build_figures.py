"""Build report tables and figures from saved results; never runs a model.

Run from any directory with Python + matplotlib. Inputs stay in ../reports/;
the manifest records their SHA256 so a later report update is reviewable.
"""
from pathlib import Path
import hashlib
import json
import math
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
IMAGE = HERE / "images"
GEN = HERE / "generated"
IMAGE.mkdir(exist_ok=True)
GEN.mkdir(exist_ok=True)
SOURCES = {}


def read(name):
    path = ROOT / name
    raw = path.read_bytes()
    SOURCES[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return json.loads(raw)


def save(fig, name):
    fig.savefig(IMAGE / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(IMAGE / f"{name}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def table(name, headers, rows, columns):
    text = "\\begin{tabular}{" + columns + "}\n\\toprule\n"
    text += " & ".join(headers) + " \\\\\n\\midrule\n"
    text += "".join(" & ".join(str(x) for x in row) + " \\\\\n" for row in rows)
    text += "\\bottomrule\n\\end{tabular}\n"
    (GEN / f"{name}.tex").write_text(text, encoding="utf-8")


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.facecolor": "white",
})
BLUE, GREEN, GREY, ORANGE = "#3465A4", "#21866B", "#A6AFBA", "#C57931"

h100 = read("reports/h100_sweep_005_20261019/combined_summary.json")
rows = h100["results"]
assert len(rows) == 8 and all(r["data_sha256"] == rows[0]["data_sha256"] for r in rows)
table("h100", ["E+D", "LR", "Парам., млн", "Token loss", "Время, с"], [
    [f'{r["encoder"]}+{r["decoder"]}', f'{r["lr"]:.0e}', f'{r["parameters"]/1e6:.2f}',
     f'{r["loss"]:.6f}', f'{r["seconds"]:.1f}'] for r in rows], "lrrrr")
fig, ax = plt.subplots(figsize=(8.2, 3.6), layout="constrained")
labels = [f'{r["encoder"]}+{r["decoder"]}, LR {r["lr"]:.0e}' for r in rows]
bars = ax.barh(labels, [r["loss"] for r in rows], color=[GREEN] + [GREY]*7)
ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
ax.invert_yaxis()
ax.set(xlim=(0, .98), xlabel="Validation token loss (lower is better)")
ax.set_title("H100: BART, 1M train / 20k validation, one epoch")
save(fig, "h100_sweep")

diag = read("reports/transformer_002_loss_investigation/summary.json")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
diag_labels = {"original_lr": "BF16, original LR", "lr_div3": "BF16, LR / 3", "fp32_original_lr": "FP32, original LR"}
for key, color in zip(diag_labels, [ORANGE, GREEN, BLUE]):
    mode = diag["modes"][key]
    axes[0].plot([50100+i*100 for i in range(6)], mode["mean_loss_per_100_steps"],
                 marker="o", ms=3, color=color, label=diag_labels[key])
axes[0].set(xlabel="Step", ylabel="Mean train loss per 100 steps")
axes[0].ticklabel_format(axis="x", style="plain", useOffset=False)
axes[0].legend(fontsize=8)
bars = axes[1].bar(range(3), [diag["modes"][k]["final_validation"]["eval_token_loss"] for k in diag_labels],
                    color=[ORANGE, GREEN, BLUE])
axes[1].bar_label(bars, fmt="%.4f", padding=3)
axes[1].set(xticks=range(3), xticklabels=["BF16\noriginal LR", "BF16\nLR / 3", "FP32\noriginal LR"],
            ylabel="Token loss on fixed 2,048 validation rows", ylim=(0, 1.0))
save(fig, "loss_diagnostic")

old = read("reports/transformer_002_checkpoint360000_test10000_seed42/comparison_summary.json")
table("old002", ["Checkpoint", "Greedy, \\%", "Beam 4, \\%", "Решено, beam 4"], [
    [step, f'{old["modes"]["greedy"]["checkpoints"][step]["pass@1"]*100:.2f}',
     f'{old["modes"]["beam4"]["checkpoints"][step]["pass@1"]*100:.2f}',
     f'{old["modes"]["beam4"]["checkpoints"][step]["num_passed"]}/9946']
    for step in ["50000", "90000", "360000"]], "lrrr")

kaggle_train = read("reports/kaggle_20261019/results_summary.json")
kaggle = read("reports/kaggle_20261020_pass/comparison_summary.json")
table("kaggle", ["Модель", "Парам., млн", "Token loss", "Решено / 19886", "\\pass{}, \\%"], [
    [key.upper() if key == "t5" else "Copy", f'{kaggle_train["models"][key]["parameters"]/1e6:.2f}',
     f'{kaggle_train["models"][key]["metrics"]["eval_token_loss"]:.6f}',
     kaggle["models"][key]["num_passed"], f'{kaggle["models"][key]["pass@1"]*100:.4f}']
    for key in ["t5", "copy"]], "lrrrr")

s1 = read("reports/a100_sequential_20261022/stage1_comparison.json")
s1_complete = {key: read(f"reports/a100_sequential_20261022/{key}/complete.json") for key in ["bart", "copy"]}
table("stage1", ["Модель", "Token loss", "Решено / 19886", "\\pass{}, \\%", "Ошибки / 20k"], [
    [key.upper() if key == "bart" else "Copy", f'{s1_complete[key]["metrics"]["eval_token_loss"]:.6f}',
     s1["models"][key]["num_passed"], f'{s1["models"][key]["pass@1"]*100:.4f}',
     s1["models"][key]["num_bad_code"]] for key in ["bart", "copy"]], "lrrrr")
s2 = read("reports/a100_sequential_20261022/stage2_recovery/stage2_comparison_checks.json")
assert s2["status"] == "passed" and not s2["findings"]
assert s1["models"]["bart"]["validation_sha256"] == s2["validation"]["validation_sha256"]
table("stage2", ["Train / предъявления", "Token loss", "Решено / 19886", "\\pass{}, \\%"], [
    [label, f'{s2["models"][key]["token_loss"]:.6f}', s2["models"][key]["num_passed"],
     f'{s2["models"][key]["pass@1"]*100:.4f}']
    for key, label in [("repeat4m", "1M $\\times$ 4"), ("distinct4m", "4M различных")]], "lrrr")
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), layout="constrained")
bars = axes[0].bar(["1M repeated 4x", "4M distinct"], [s2["models"][k]["pass@1"]*100 for k in ["repeat4m", "distinct4m"]],
                  color=[GREY, GREEN], width=.6)
axes[0].bar_label(bars, fmt="%.4f%%", padding=3)
axes[0].set(ylabel="Validation pass@1 (%)", ylim=(0, 80), title="Same 4M exposure budget")
paired = s2["paired"]
bars = axes[1].bar(["Distinct only", "Repeat only"], [paired["distinct4m_only"], paired["repeat4m_only"]],
                  color=[GREEN, GREY], width=.6)
axes[1].bar_label(bars, padding=3)
axes[1].set(ylabel="Eligible validation examples", ylim=(0, 1100), title="Paired outcomes: net +212")
save(fig, "data_diversity")

stage4 = read("reports/a100_sequential_20261022/stage4_decoding/stage4_complete.json")
checks4 = read("reports/a100_sequential_20261022/stage4_decoding/comparison_checks.json")
assert checks4["status"] == "passed" and not checks4["findings"]
candidates = stage4["candidates"]
assert len(candidates) == 5 and all(r["num_evaluated"] == 1988 for r in candidates)
table("decoding", ["Beam", "LP", "Batch", "Решено / 1988", "\\pass{}, \\%", "A100, с"], [
    [r["num_beams"], f'{r["length_penalty"]:.1f}', r["batch_size"], r["num_passed"],
     f'{r["pass@1"]*100:.4f}', f'{r["generation_seconds"]:.2f}'] for r in candidates], "rrrrrr")
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1), layout="constrained")
labels = [f'b{r["num_beams"]}\nLP {r["length_penalty"]:g}' for r in candidates]
colors = [GREY, GREEN, GREY, BLUE, BLUE]
bars = axes[0].bar(labels, [r["num_passed"] for r in candidates], color=colors)
axes[0].bar_label(bars, padding=2, fontsize=8)
axes[0].set(ylabel="Solved / 1,988 eligible", ylim=(0, 1650))
bars = axes[1].bar(labels, [r["generation_seconds"] for r in candidates], color=colors)
axes[1].bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
axes[1].set(ylabel="Generation time on A100 (seconds)", ylim=(0, 115))
save(fig, "decoding")

full = read("reports/a100_full_20261023/final_report/training/trial/complete.json")
full_summary = read("reports/a100_full_20261023/completion_summary.json")
history = read("reports/a100_full_20261023/final_report/training/trial/log_history.json")
independent = read("reports/a100_full_20261023/final_report/independent_checks.json")
audit = read("reports/a100_sequential_20261022/stage3_audit/review_summary.json")
assert full_summary["pipeline_exitcode"] == 0
assert full["global_step"] == 181144 and full["best_global_step"] == 180000
assert full_summary["final_weights_exactly_match_best"]
assert full_summary["independent_checks"] == "passed"
assert all(math.isfinite(float(row[k])) for row in history for k in ("loss", "eval_loss", "eval_token_loss") if k in row)
assert audit["coverage"]["validation_failures_reviewed"] == audit["coverage"]["original_train_reviewed"] == 200
train = [r for r in history if "loss" in r]
validation = {}
for r in history:
    if "eval_token_loss" in r:
        validation.setdefault(r["step"], r)  # First record is the actual training checkpoint, before best-weight restore.
validation = list(validation.values())
assert min(validation, key=lambda r: r["eval_token_loss"])["step"] == full["best_global_step"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
axes[0].plot([r["step"]/1000 for r in train], [r["loss"] for r in train], color=BLUE, alpha=.8, lw=.8, label="Logged train loss")
axes[0].plot([r["step"]/1000 for r in validation], [r["eval_token_loss"] for r in validation], color=GREEN, marker="o", ms=3, lw=1.3, label="Validation token loss")
axes[0].set(xlabel="Scheduled step attempts (thousands)", ylabel="Loss (log scale)", yscale="log")
axes[0].legend(fontsize=8)
late = [r for r in validation if r["step"] >= 100000]
axes[1].plot([r["step"]/1000 for r in late], [r["eval_token_loss"] for r in late], color=GREEN, marker="o", ms=4)
axes[1].scatter([180], [full["best_metric"]], s=90, marker="*", color=ORANGE, zorder=4, label="Best: step 180,000")
axes[1].set(xlabel="Scheduled step attempts (thousands)", ylabel="Validation token loss", title="Late validation (expanded scale)")
axes[1].legend(fontsize=8)
save(fig, "full_training")

full_eval_root = "reports/a100_full_eval_20261023"
new_quality = read(f"{full_eval_root}/downloaded/evaluation/metrics.json")
new_checks = read(f"{full_eval_root}/comparison_checks.json")
t4_root = f"{full_eval_root}/t4_batch128"
t4 = read(f"{t4_root}/benchmark/metrics.json")
t4_request = read(f"{t4_root}/benchmark/request.json")
t4_complete = read(f"{t4_root}/benchmark/complete.json")
t4_receipt = read(f"{t4_root}/download_receipt.json")
t4_bundle = read(f"{t4_root}/bundle_manifest.json")
assert new_checks["status"] == "passed" and not new_checks["findings"]
assert all(new_checks["checks"].values())
assert new_quality == {k: new_checks["full_model"][k] for k in new_quality}
assert new_quality["validation_sha256"] == s2["validation"]["validation_sha256"]
assert new_quality["num_evaluated"] == 19886
assert new_quality["num_passed"] == 14775
assert new_checks["paired"]["net_additional_full_passes"] == 1118
assert t4_complete["status"] == "complete" and t4_complete["within_180_seconds"]
assert t4["num_examples"] == t4_request["official_test_rows"] == 1919
assert t4_request["gpus_used"] == t4_request["visible_gpus"] == 1
assert t4["gpu"] == t4_request["gpu"] == "Tesla T4"
assert t4["checkpoint_step"] == full["best_global_step"] == 180000
assert t4["weight_sha256"] == t4_request["weight_sha256"] == new_checks["source_checkpoint"]["weight_sha256"]
assert t4["wall_generation_seconds"] < t4["official_limit_seconds"] == 180
assert t4["batch_size"] == t4_request["batch_size"] == 128
for item in t4_receipt["files"]:
    raw = (ROOT / t4_root / item["path"]).read_bytes()
    assert len(raw) == item["bytes"]
    assert hashlib.sha256(raw).hexdigest() == item["sha256"]
submission = ROOT / t4_root / "benchmark/a100_full_best180000_beam4_t4.jsonl"
assert len(submission.read_text().splitlines()) == 1919
assert hashlib.sha256(submission.read_bytes()).hexdigest() == t4_complete["submission_sha256"]
table("full_quality", ["Модель / бюджет", "Решено / 19886", "\\pass{}, \\%", "Ошибки / 20k"], [
    ["Distinct4m / 4M", s2["models"]["distinct4m"]["num_passed"],
     f'{s2["models"]["distinct4m"]["pass@1"]*100:.4f}', s2["models"]["distinct4m"]["num_bad_code"]],
    ["Полный train / 4 эпохи", new_quality["num_passed"],
     f'{new_quality["pass@1"]*100:.4f}', new_quality["num_bad_code"]],
], "lrrr")

public = read("report/evidence/public_comparison.json")
old_config = read("reports/transformer_002_resume_lr1e4/model_config.json")
old_training = read("reports/transformer_002_resume_lr1e4/training_config.json")
old_final = read("reports/transformer_002_resume_lr1e4/final_training_summary.json")
old_layout = read("reports/transformer_002_checkpoint50000_test10000_seed42/training/training_layout.json")
new_request = read("reports/a100_full_20261023/run_request.json")
assert public["public_rows"] == 960
assert public["old_model"]["num_passed"] == 735 and public["new_model"]["num_passed"] == 729
delta = 100 * (public["new_model"]["num_passed"] - public["old_model"]["num_passed"]) / public["public_rows"]
assert abs(delta - public["delta_new_minus_old"]["percentage_points"]) < 1e-12
assert public["delta_new_minus_old"]["num_passed"] == -6
assert public["new_model"]["submission_sha256"] == t4_complete["submission_sha256"]
assert hashlib.sha256((ROOT / public["new_model"]["screenshot"]).read_bytes()).hexdigest() == public["new_model"]["screenshot_sha256"]
assert public["private_result"] is None and public["paired_public_outcomes"] is None
assert old_layout["effective_batch_size"] == 128 and new_request["config"]["global_batch"] == 256
assert old_final["restored_best_validation"][0]["eval_target_tokens"] == full["metrics"]["eval_target_tokens"] == 7970835
new_config = new_request["model"]
public_rows = [
    ["Encoder + decoder", f'{old_config["encoder_layers"]}+{old_config["decoder_layers"]}',
     f'{new_config["encoder_layers"]}+{new_config["decoder_layers"]}'],
    ["Hidden / FFN", f'{old_config["d_model"]} / {old_config["ffn_dim"]}',
     f'{new_config["d_model"]} / {new_config["ffn_dim"]}'],
    ["Параметры, млн", f'{public["old_model"]["parameters"]/1e6:.2f}', f'{public["new_model"]["parameters"]/1e6:.2f}'],
    ["Train / эпохи", "11\\,593\\,060 / 4", "11\\,593\\,060 / 4"],
    ["BPE / seed", "8000 / 42", "8000 / 42"],
    ["Global batch", old_layout["effective_batch_size"], new_request["config"]["global_batch"]],
    ["LR", "$3\\cdot10^{-4}$; после 50k $10^{-4}$", "$2\\cdot10^{-4}$"],
    ["Точность обучения", old_training["mixed_precision"].upper(), new_request["config"]["precision"].upper()],
    ["Лучший checkpoint", public["old_model"]["best_checkpoint"], full["best_global_step"]],
    ["Token loss Val-full", f'{old_final["best_eval_token_loss"]:.7f}', f'{full["best_metric"]:.7f}'],
    ["Beam / max tokens", "4 / 256", "4 / 256"],
    ["Public", f'{public["old_model"]["num_passed"]}/960 (76.5625\\%)',
     f'{public["new_model"]["num_passed"]}/960 (75.9375\\%)'],
]
table("public_comparison", ["Параметр", "Прежняя 002", "Новая полная"], public_rows,
      r">{\raggedright\arraybackslash}p{.30\linewidth}>{\raggedright\arraybackslash}p{.28\linewidth}>{\raggedright\arraybackslash}p{.28\linewidth}")

supporting_sources = {}
for name in ["README.md", "models/baseline.py", "models/t5.py", "models/copy.py",
             "evaluation.py", "configs/transformer_003_10x10_d1024_h100/model_config.json",
             "report/evidence/public_new_bot_20261023.png"]:
    raw = (ROOT / name).read_bytes()
    supporting_sources[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
manifest = {
    "document_version": "0.4", "date": "2026-10-23",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "sources": SOURCES, "supporting_sources": supporting_sources,
    "checks": {
        "h100_shared_data_hash": True, "stage1_stage2_same_validation_hash": True,
        "stage2_saved_comparison_passed": True, "stage4_saved_comparison_passed": True,
        "full_training_exit0": True, "full_training_losses_finite": True,
        "full_training_best_step_matches_history": True,
        "full_model_pass_at_1_included": True, "full_model_t4_timing_included": True,
        "full_eval_same_validation_and_pass_flags_verified": True,
        "full_eval_and_t4_best_checkpoint_weight_identity": True,
        "t4_all_download_receipt_hashes_verified": True,
        "t4_official_submission_sha_and_1919_rows_verified": True,
        "public_integer_counts_and_delta_verified": True,
        "new_public_screenshot_hash_verified": True,
        "new_public_submission_matches_t4_verified_file": True,
        "private_result_unknown_not_inferred": True,
    },
    "new_public_result": {"old_passed":735,"new_passed":729,"public_rows":960,"delta_pp":delta,
                          "new_source":"User's bot screenshot","old_source":"Prior author-reported bot result in README","private":None},
    "scope": "Build figures and tables from saved reports only. No model runs or remote operations.",
}
(HERE / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Built 5 figures and 8 tables from {len(SOURCES)} saved JSON files.")
