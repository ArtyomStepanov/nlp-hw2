import json

import datasets as ds
import pytest

from evaluate_checkpoint import (
    CheckpointEvaluationConfig,
    find_latest_result,
    load_local_dataset,
    run_evaluation,
    select_heldout_sample,
)
from prepare_datasets import split_dataset


def test_sample_reproduces_training_split_and_preserves_source_identity():
    raw = ds.Dataset.from_dict({
        "question": [f"question-{index}" for index in range(100)],
        "code": [f"def solve():\n    return {index} # comment" for index in range(100)],
    })
    parts = split_dataset(raw, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42,
                          columns=["question", "code"])
    expected = parts["test"].shuffle(seed=42).select(range(7))

    sample, manifest = select_heldout_sample(raw, sample_size=7, seed=42)

    assert list(sample["question"]) == list(expected["question"])
    assert all(raw[int(row["source_row_index"])]["question"] == row["question"] for row in sample)
    assert list(sample["sample_row"]) == list(range(7))
    assert all("# comment" not in code for code in sample["code"])
    assert manifest["split_seed"] == 42 and manifest["sample_seed"] == 42
    assert manifest["heldout_size"] == 10 and manifest["sample_size"] == 7
    assert set(sample["question"]).isdisjoint(parts["train"]["question"])
    assert set(sample["question"]).isdisjoint(parts["val"]["question"])
    other, other_manifest = select_heldout_sample(raw, sample_size=7, seed=17)
    assert set(other["question"]).issubset(parts["test"]["question"])
    assert other_manifest["split_seed"] == 42


@pytest.mark.parametrize("validation_name", ["val", "validation", None])
@pytest.mark.parametrize("reference", ["def solve():\n    return 1", ""])
def test_rejects_pre_split_datasetdict(tmp_path, validation_name, reference):
    part = ds.Dataset.from_dict({"question": ["q"], "code": [reference]})
    parts = {"train": part, "test": part}
    if validation_name is not None:
        parts[validation_name] = part
    ds.DatasetDict(parts).save_to_disk(str(tmp_path / "pre_split"))

    with pytest.raises(ValueError, match="уже разбит"):
        load_local_dataset(tmp_path / "pre_split")


def test_accepts_official_parquet_test_with_schema_unified_null_code(tmp_path):
    train_path = tmp_path / "data" / "train"
    test_path = tmp_path / "data" / "test"
    train_path.mkdir(parents=True)
    test_path.mkdir(parents=True)
    ds.Dataset.from_dict({"question": ["train"], "code": ["def solve():\n    return 1"]}).to_parquet(
        str(train_path / "part-00000.parquet"))
    ds.Dataset.from_dict({"question": ["official one", "official two"]}).to_parquet(
        str(test_path / "test.parquet"))

    data, path = load_local_dataset(tmp_path)

    assert path == tmp_path
    assert "code" in data["test"].column_names
    assert list(data["test"]["code"]) == [None, None]
    assert list(data["test"]["question"]) == ["official one", "official two"]


def test_reference_detection_respects_selected_rows():
    from evaluate_checkpoint import _has_reference_code

    source = ds.Dataset.from_dict({"question": ["a", "b", "c"], "code": [None, "", None]})
    assert _has_reference_code(source) is True
    assert _has_reference_code(source.select([2, 0])) is False
    assert _has_reference_code(source.select([2, 1, 0])) is True


def test_latest_result_does_not_confuse_checkpoints_with_same_step(tmp_path):
    for experiment, timestamp in [("baseline_001", "2026-10-12"), ("other_experiment", "2026-10-13")]:
        output = tmp_path / experiment
        output.mkdir()
        (output / "run.json").write_text(json.dumps({
            "status": "complete", "completed_at": timestamp,
            "checkpoint": f"/remote/repo/artifacts/experiments/{experiment}/checkpoint-147871",
            "config": {"sample_size": 10_000, "seed": 42},
        }), encoding="utf-8")

    assert find_latest_result(
        checkpoint="artifacts/experiments/baseline_001/checkpoint-147871", results_root=tmp_path,
    ) == tmp_path / "baseline_001"


def test_run_records_exception_as_failed(tmp_path):
    checkpoint = tmp_path / "checkpoint-147871"
    checkpoint.mkdir()
    output = tmp_path / "run"
    config = CheckpointEvaluationConfig(checkpoint=checkpoint, data_dir=tmp_path / "missing_data",
                                        output_dir=output, device="cpu", precision="no")

    with pytest.raises(FileNotFoundError):
        run_evaluation(config)

    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    assert run["error_type"] == "FileNotFoundError"
    assert "missing_data" in run["error"]
    assert run["failed_at"] >= run["started_at"]
