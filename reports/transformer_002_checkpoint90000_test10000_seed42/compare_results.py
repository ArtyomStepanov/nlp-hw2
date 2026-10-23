"""Verify paired 50k/90k results, report changes, and plot saved metrics only."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_result(folder, mode, expected_hash):
    records = rows(folder / mode / "evaluation/evaluation.jsonl")
    metrics = read(folder / mode / "metrics.json")
    assert len(records) == 10000
    assert [r["sample_row"] for r in records] == list(range(10000))
    digest = hashlib.sha256()
    for record in records:
        source = {key: record[key] for key in ("question", "code", "test_row_index", "sample_row")}
        digest.update((json.dumps(source, ensure_ascii=False, sort_keys=True) + "\n").encode())
    assert digest.hexdigest() == expected_hash, (folder, mode, "sample mismatch")
    assert sum(r["evaluable"] for r in records) == metrics["num_evaluated"]
    assert sum(r["passed"] is True for r in records) == metrics["num_passed"]
    assert sum(r["execution_status"] != "ok" for r in records) == metrics["num_bad_code"]
    assert math.isclose(metrics["pass@1"], metrics["num_passed"] / metrics["num_evaluated"])
    assert read(folder / mode / "run.json")["status"] == "complete"
    return records, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    old, new = args.previous, args.current
    manifest = read(new / "sample_manifest.json")
    assert manifest["sample_sha256"] == read(old / "sample_manifest.json")["sample_sha256"]
    assert read(new / "exitcodes.json") == {"greedy": 0, "beam4": 0}
    summary = {"sample_size": 10000, "sample_sha256": manifest["sample_sha256"],
               "previous_step": 50000, "current_step": 90000,
               "scope": "Same 10,000 examples from saved 98/1/1 test; not the public leaderboard",
               "timing_note": "90k inference ran alongside training; no speed comparison",
               "modes": {}}
    all_new = {}
    for mode in ("greedy", "beam4"):
        before, bm = verify_result(old, mode, manifest["sample_sha256"])
        after, am = verify_result(new, mode, manifest["sample_sha256"])
        all_new[mode] = after
        bc = read(old / mode / "inference/inference_config.json")
        ac = read(new / mode / "inference/inference_config.json")
        assert {k: v for k, v in bc.items() if k not in {"device", "output_dir"}} == {
            k: v for k, v in ac.items() if k not in {"device", "output_dir"}}
        be = read(old / mode / "evaluation/evaluation_config.json")
        ae = read(new / mode / "evaluation/evaluation_config.json")
        assert {k: v for k, v in be.items() if k != "output_dir"} == {
            k: v for k, v in ae.items() if k != "output_dir"}
        for b, a in zip(before, after):
            for key in ("reference_status", "reference_result_json", "evaluable",
                        "source_tokens", "source_truncated"):
                assert b[key] == a[key], (mode, a["sample_row"], key)
        paired = [(b, a) for b, a in zip(before, after) if a["evaluable"]]
        gained = sum(b["passed"] is False and a["passed"] is True for b, a in paired)
        lost = sum(b["passed"] is True and a["passed"] is False for b, a in paired)
        assert gained - lost == am["num_passed"] - bm["num_passed"]
        summary["modes"][mode] = {
            "num_evaluated": am["num_evaluated"], "num_reference_errors": 10000 - am["num_evaluated"],
            "previous_passed": bm["num_passed"], "current_passed": am["num_passed"],
            "previous_pass_at_1": bm["pass@1"], "current_pass_at_1": am["pass@1"],
            "delta_percentage_points": 100 * (am["pass@1"] - bm["pass@1"]),
            "gained": gained, "lost": lost,
            "previous_bad_code_rate": bm["bad_code_rate"], "current_bad_code_rate": am["bad_code_rate"],
            "current_status_counts": am["status_counts"],
            "execution_status_counts": dict(Counter(r["execution_status"] for r in after)),
            "mean_generated_tokens": am["mean_generated_tokens"],
            "num_hit_token_limit": am["num_hit_token_limit"],
        }
    assert [(r["reference_status"], r["reference_result_json"]) for r in all_new["greedy"]] == [
        (r["reference_status"], r["reference_result_json"]) for r in all_new["beam4"]]
    (new / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nlp-hw2-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    colors = ["#718096", "#208359"]
    for ax, before_key, after_key, title in (
        (axes[0], "previous_pass_at_1", "current_pass_at_1", "Локальный pass@1"),
        (axes[1], "previous_bad_code_rate", "current_bad_code_rate", "Код с ошибками выполнения"),
    ):
        for offset, key, color, label in [(-0.19, before_key, colors[0], "50 000 шагов"),
                                          (0.19, after_key, colors[1], "90 000 шагов")]:
            values = [100 * summary["modes"][m][key] for m in ("greedy", "beam4")]
            bars = ax.bar([offset, 1 + offset], values, width=0.36, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=4, fontsize=10)
        ax.set_xticks([0, 1], ["Greedy", "Beam 4"])
        ax.set_title(title)
        ax.set_ylabel("%")
        ax.set_ylim(bottom=0, top=85 if ax is axes[0] else 3.1)
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Checkpoint 50k → 90k · те же 10 000 тестовых примеров", fontsize=15)
    fig.supxlabel("FP16 · max_new_tokens=256 · seed=42 · совпадение solve(), rtol=atol=1e-6\n"
                  "54 некорректных эталона исключены из pass@1; ошибки кода считаются по всем 10 000 ответам.", fontsize=9)
    plots = new / "plots"
    plots.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(plots / f"comparison.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
