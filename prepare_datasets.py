"""Очистка Python-кода в Hugging Face datasets и воспроизводимое разбиение."""

import io
import math
import tokenize
from collections.abc import Mapping, Sequence

import datasets as ds


def remove_code_comments(code: str) -> str:
    """Удалить Python-комментарии #, сохранив строки, docstring и переносы строк.

    Если код нельзя токенизировать, возникает ValueError. Код не исполняется.
    """
    if not isinstance(code, str):
        raise TypeError("Код должен быть строкой.")

    lines = io.StringIO(code).readlines()
    comments: dict[int, list[tuple[int, int]]] = {}
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type == tokenize.COMMENT:
                line_index = token.start[0] - 1
                start, end = token.start[1], token.end[1]
                comments.setdefault(line_index, []).append((start, end))
            elif token.type == tokenize.ERRORTOKEN and not token.string.isspace():
                raise ValueError(f"Не удалось токенизировать код: {token.string!r}.")
    except (tokenize.TokenError, SyntaxError) as error:
        raise ValueError(f"Не удалось токенизировать код: {error}") from error

    for line_index, spans in comments.items():
        # Удаляем с конца, чтобы сохранить исходные позиции остальных токенов.
        for start, end in reversed(spans):
            lines[line_index] = lines[line_index][:start] + lines[line_index][end:]
    return "".join(lines)


def _column_names(columns: str | Sequence[str]) -> list[str]:
    names = [columns] if isinstance(columns, str) else list(columns)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Укажите непустые названия столбцов.")
    if len(names) != len(set(names)):
        raise ValueError("Названия столбцов не должны повторяться.")
    return names


def _validate_columns(columns: Sequence[str], available: Sequence[str]) -> None:
    missing = set(columns) - set(available)
    if missing:
        raise ValueError(f"Столбцы не найдены: {', '.join(sorted(missing))}.")


def _clean_code_batch(batch: dict, columns: Sequence[str]) -> dict:
    return {
        column: [None if code is None else remove_code_comments(code) for code in batch[column]]
        for column in columns
    }


def remove_dataset_comments(
    dataset: ds.Dataset | ds.DatasetDict,
    columns: str | Sequence[str] = ("code",),
    *,
    num_proc: int | None = None,
    batch_size: int = 1000,
) -> ds.Dataset | ds.DatasetDict:
    """Очистить выбранные столбцы с Python-кодом, не изменяя исходный датасет.

    В DatasetDict части без выбранных столбцов сохраняются как есть: например,
    test в tinysms_qa_12m содержит только question. Каждый запрошенный столбец
    должен существовать хотя бы в одной части. Значения None сохраняются.
    num_proc задаёт число процессов для Dataset.map.
    """
    if not isinstance(dataset, (ds.Dataset, ds.DatasetDict)):
        raise TypeError("Ожидается datasets.Dataset или datasets.DatasetDict.")
    columns = _column_names(columns)

    parts = dataset.values() if isinstance(dataset, ds.DatasetDict) else [dataset]
    available = [column for part in parts for column in part.column_names]
    _validate_columns(columns, available)

    def clean(part: ds.Dataset) -> ds.Dataset:
        selected = [column for column in columns if column in part.column_names]
        if not selected:
            return part
        return part.map(
            _clean_code_batch,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={"columns": selected},
            num_proc=num_proc,
            desc="Удаление комментариев из кода",
        )

    if isinstance(dataset, ds.DatasetDict):
        return ds.DatasetDict({name: clean(part) for name, part in dataset.items()})
    return clean(dataset)


def _split_counts(sizes: Mapping[str, int | float], num_rows: int) -> list[int]:
    if not isinstance(sizes, Mapping) or not sizes:
        raise ValueError("Укажите размеры частей, например {'train': 0.8, 'val': 0.1, 'test': 0.1}.")
    if any(not isinstance(name, str) or not name for name in sizes):
        raise ValueError("Названия частей должны быть непустыми строками.")
    values = list(sizes.values())
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("Размеры должны быть целыми числами строк или дробными долями.")

    if all(isinstance(value, int) for value in values):
        if any(value < 0 for value in values) or sum(values) > num_rows:
            raise ValueError("Числа строк должны быть неотрицательными и не превышать размер датасета в сумме.")
        return values

    if not all(isinstance(value, float) for value in values):
        raise ValueError("Нельзя смешивать числа строк и доли в одном разбиении.")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Доли должны быть конечными числами от 0.0 до 1.0.")
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Сумма долей должна равняться 1.0.")

    exact_counts = [num_rows * (value / total) for value in values]
    counts = [math.floor(value) for value in exact_counts]
    order = sorted(
        range(len(counts)),
        key=lambda index: exact_counts[index] - counts[index],
        reverse=True,
    )
    for index in order[:num_rows - sum(counts)]:
        counts[index] += 1
    return counts


def split_dataset(
    dataset: ds.Dataset,
    sizes: Mapping[str, int | float],
    *,
    seed: int = 42,
    columns: str | Sequence[str] | None = None,
) -> ds.DatasetDict:
    """Перемешать Dataset и разделить его на непересекающиеся именованные части.

    sizes={'train': 0.8, 'val': 0.1, 'test': 0.1} задаёт доли с суммой 1.0;
    округление сохраняет все строки, лишние строки получают части с наибольшим
    дробным остатком (при равенстве — в порядке sizes).
    sizes={'train': 1000, 'val': 100, 'test': 100} задаёт точные числа строк;
    если сумма меньше размера датасета, остальные строки не включаются.
    columns выбирает сохраняемые столбцы; None сохраняет все.

    Для DatasetDict передавайте конкретную часть, например dataset['train'].
    Исходные данные и их порядок не меняются.
    """
    if not isinstance(dataset, ds.Dataset):
        raise TypeError("Ожидается datasets.Dataset; для DatasetDict передайте dataset['train'].")
    counts = _split_counts(sizes, len(dataset))
    if columns is not None:
        columns = _column_names(columns)
        _validate_columns(columns, dataset.column_names)
        dataset = dataset.select_columns(columns)
    shuffled = dataset.shuffle(seed=seed)

    parts = {}
    start = 0
    for name, count in zip(sizes, counts):
        indices = range(start, start + count) if count else range(0)
        parts[name] = shuffled.select(indices)
        start += count
    return ds.DatasetDict(parts)
