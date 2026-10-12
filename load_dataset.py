import argparse
from pathlib import Path

import datasets as ds
import huggingface_hub as hf


DATASET_ID = "SlavaYus/tinysms_qa_12m"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "tinysms_qa_12m"


def load_dataset(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    print(f"Скачивание {DATASET_ID} в {output_dir}", flush=True)
    hf.snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        local_dir=output_dir,
    )
    print(f"Датасет сохранён в {output_dir}")


def load_dataset_from_file(file_path: str | Path) -> ds.Dataset:
    """Прочитать один локальный Parquet-файл в datasets.Dataset."""
    file_path = Path(file_path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Parquet-файл не найден: {file_path}")

    return ds.Dataset.from_parquet(
        str(file_path),
        cache_dir=str(file_path.parent / ".cache" / "datasets"),
    )


def load_dataset_from_files(
    dataset_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> ds.DatasetDict:
    """Прочитать train и test из локальной копии SlavaYus/tinysms_qa_12m."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    train_files = sorted((dataset_dir / "data" / "train").glob("part-*.parquet"))
    test_file = dataset_dir / "data" / "test" / "test.parquet"

    if not train_files:
        raise FileNotFoundError(f"Файлы train не найдены в {dataset_dir / 'data' / 'train'}")
    if not test_file.is_file():
        raise FileNotFoundError(f"Файл test не найден: {test_file}")

    return ds.load_dataset(
        "parquet",
        data_files={
            "train": [str(path) for path in train_files],
            "test": str(test_file),
        },
        cache_dir=str(dataset_dir / ".cache" / "datasets"),
    )
