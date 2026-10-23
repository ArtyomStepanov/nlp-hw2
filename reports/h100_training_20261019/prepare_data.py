"""Reproduce experiment 002's 98/1/1 split on the new machine."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from huggingface_hub import snapshot_download
import datasets as ds
from load_dataset import load_dataset_from_files
from prepare_datasets import remove_dataset_comments, split_dataset

destination = ROOT / "artifacts/datasets/tinysms_98_1_1_seed42"
if (destination / "split_manifest.json").is_file():
    print("Already prepared:", destination, flush=True)
else:
    started = time.perf_counter()
    raw_dir = ROOT / "data/tinysms_qa_12m"
    snapshot_download("SlavaYus/tinysms_qa_12m", repo_type="dataset", local_dir=raw_dir,
                      allow_patterns=["data/train/*.parquet", "data/test/test.parquet"], max_workers=8)
    print("DOWNLOAD COMPLETE", flush=True)
    raw = load_dataset_from_files(raw_dir)
    assert len(raw["train"]) == 11838003
    clean = remove_dataset_comments(raw["train"], num_proc=16)
    parts = split_dataset(clean, {"train": .98, "val": .01, "test": .01}, seed=42,
                          columns=["question", "code"])
    staging = destination.with_name(destination.name + ".preparing")
    if staging.exists():
        raise FileExistsError(f"Incomplete output requires inspection: {staging}")
    parts.save_to_disk(str(staging))
    manifest = {"dataset_id": "SlavaYus/tinysms_qa_12m", "source_split": "train",
                "sizes": {"train": .98, "val": .01, "test": .01}, "seed": 42,
                "columns": ["question", "code"], "remove_code_comments": True,
                "preprocessing_sha256": hashlib.sha256((ROOT / "prepare_datasets.py").read_bytes()).hexdigest(),
                "rows": {name: len(part) for name, part in parts.items()},
                "prepare_seconds": time.perf_counter()-started}
    (staging / "split_manifest.json").write_text(json.dumps(manifest,indent=2))
    staging.rename(destination)
    print("READY", json.dumps(manifest), flush=True)
