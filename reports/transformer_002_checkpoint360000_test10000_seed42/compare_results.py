"""Validate paired 50k/90k/final evaluations and render saved results."""
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


def load_result(folder, mode, expected_hash):
    records = [json.loads(line) for line in (folder / mode / "evaluation/evaluation.jsonl").read_text().splitlines()]
    metrics = read(folder / mode / "metrics.json")
    assert len(records) == 10000
    assert [r["sample_row"] for r in records] == list(range(10000))
    digest = hashlib.sha256()
    for record in records:
        original = {k: record[k] for k in ("question", "code", "test_row_index", "sample_row")}
        digest.update((json.dumps(original, ensure_ascii=False, sort_keys=True) + "\n").encode())
    assert digest.hexdigest() == expected_hash, (folder, mode, "sample mismatch")
    assert sum(r["evaluable"] for r in records) == metrics["num_evaluated"]
    assert sum(r["passed"] is True for r in records) == metrics["num_passed"]
    assert sum(r["execution_status"] != "ok" for r in records) == metrics["num_bad_code"]
    assert math.isclose(metrics["pass@1"], metrics["num_passed"] / metrics["num_evaluated"])
    assert read(folder / mode / "run.json")["status"] == "complete"
    return records, metrics


def paired_comparison(before, after):
    for b, a in zip(before, after):
        for key in ("reference_status", "reference_result_json", "evaluable", "source_tokens", "source_truncated"):
            assert b[key] == a[key], (a["sample_row"], key)
    paired = [(b, a) for b, a in zip(before, after) if a["evaluable"]]
    gained = sum(b["passed"] is False and a["passed"] is True for b, a in paired)
    lost = sum(b["passed"] is True and a["passed"] is False for b, a in paired)
    return {"num_evaluated": len(paired), "gained": gained, "lost": lost,
            "net_more_solved": gained - lost, "delta_percentage_points": 100 * (gained - lost) / len(paired)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--intermediate", type=Path, required=True)
    parser.add_argument("--current", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    folders = {"50000": args.baseline, "90000": args.intermediate, "360000": args.current}
    manifest = read(args.current / "sample_manifest.json")
    assert manifest["checkpoint_step"] == 360000
    assert read(args.current / "exitcodes.json") == {"greedy": 0, "beam4": 0}
    for folder in folders.values():
        assert read(folder / "sample_manifest.json")["sample_sha256"] == manifest["sample_sha256"]
    assert read(args.current / "source/sha256.json") == read(args.intermediate / "source/sha256.json")
    summary = {"sample_size": 10000, "sample_sha256": manifest["sample_sha256"],
               "checkpoint_steps": list(map(int, folders)), "final_checkpoint_step": 360000,
               "scope": "Same saved 10,000 examples from 98/1/1 held-out test; not the public leaderboard",
               "protocol": {"max_new_tokens": 256, "precision": "fp16", "rtol": 1e-6, "atol": 1e-6},
               "modes": {}}
    final_rows = {}
    for mode in ("greedy", "beam4"):
        data = {step: load_result(folder, mode, manifest["sample_sha256"]) for step, folder in folders.items()}
        final_rows[mode] = data["360000"][0]
        for relative, excluded in [("inference/inference_config.json", {"device", "output_dir"}),
                                   ("evaluation/evaluation_config.json", {"output_dir"})]:
            configs = [{k: v for k, v in read(folder / mode / relative).items() if k not in excluded}
                       for folder in folders.values()]
            assert configs[0] == configs[1] == configs[2], (mode, relative)
        metrics = {}
        for step, (records, raw) in data.items():
            metrics[step] = {k: raw[k] for k in ("num_evaluated", "num_passed", "pass@1", "num_bad_code",
                                                 "bad_code_rate", "mean_generated_tokens", "num_hit_token_limit")}
            metrics[step]["execution_status_counts"] = dict(Counter(r["execution_status"] for r in records))
        changes = {}
        for step in ("50000", "90000"):
            change = paired_comparison(data[step][0], data["360000"][0])
            assert change["net_more_solved"] == metrics["360000"]["num_passed"] - metrics[step]["num_passed"]
            changes["from_" + step] = change
        summary["modes"][mode] = {"checkpoints": metrics, "final_changes": changes}
    summary["final_beam_vs_greedy"] = paired_comparison(final_rows["greedy"], final_rows["beam4"])
    summary["num_reference_errors"] = 10000 - summary["modes"]["greedy"]["checkpoints"]["360000"]["num_evaluated"]
    (args.current / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nlp-hw2-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "nlp-hw2-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for ax, key, title in [(axes[0], "pass@1", "Локальный pass@1"),
                            (axes[1], "bad_code_rate", "Код с ошибками выполнения")]:
        for offset, step, color, label in [(-0.27, "50000", "#94a3b8", "50k"),
                                           (0, "90000", "#327ba6", "90k"),
                                           (0.27, "360000", "#208359", "360k · финальная")]:
            values = [100 * summary["modes"][mode]["checkpoints"][step][key] for mode in ("greedy", "beam4")]
            bars = ax.bar([offset, 1 + offset], values, width=0.25, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=4, fontsize=9)
        ax.set_xticks([0, 1], ["Greedy", "Beam 4"])
        ax.set_title(title)
        ax.set_ylabel("%")
        ax.set_ylim(bottom=0, top=100 if key == "pass@1" else 3.1)
        ax.grid(axis="y", alpha=0.18)
    axes[0].legend(loc="upper left", fontsize=9, ncol=3)
    fig.suptitle("Финальная модель · сравнение на тех же 10 000 тестовых примеров", fontsize=15)
    fig.supxlabel("FP16 · max_new_tokens=256 · seed=42 · совпадение solve(), rtol=atol=1e-6\n"
                  f"Из pass@1 исключены {summary['num_reference_errors']} некорректных эталона; ошибки кода — по всем 10 000 ответам.", fontsize=9)
    plots = args.current / "plots"
    plots.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(plots / f"comparison.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
