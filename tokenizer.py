"""Обучение собственного byte-level BPE на Hugging Face Dataset."""

from collections.abc import Iterator, Sequence
from pathlib import Path

import datasets as ds
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def _text_batches(
    dataset: ds.Dataset,
    columns: Sequence[str],
    batch_size: int,
) -> Iterator[list[str]]:
    has_text = False
    for batch in dataset.select_columns(list(columns)).iter(batch_size=batch_size):
        texts = []
        for column in columns:
            for text in batch[column]:
                if text is None:
                    continue
                if not isinstance(text, str):
                    raise TypeError(f"Столбец {column!r} должен содержать строки или None.")
                if text:
                    texts.append(text)
        if texts:
            has_text = True
            yield texts
    if not has_text:
        raise ValueError("В выбранных столбцах нет непустых текстов для обучения.")


def train_bpe_tokenizer(
    dataset: ds.Dataset | ds.DatasetDict,
    vocab_size: int,
    columns: str | Sequence[str] = ("question", "code"),
    *,
    min_frequency: int = 2,
    batch_size: int = 1000,
    save_path: str | Path | None = None,
) -> Tokenizer:
    """Обучить BPE с нуля на выбранных текстовых столбцах и вернуть Tokenizer.

    Для DatasetDict используется только часть train. Передавайте обучающую
    часть после выделения валидации. Тексты читаются батчами, без создания
    общей Python-копии корпуса. Нормализация и добавление пробелов отключены.

    vocab_size включает 256 базовых байтовых токенов и четыре специальных
    токена; минимальное значение — 260. На маленьком корпусе фактический
    словарь может быть меньше запрошенного. Encode по умолчанию добавляет
    <bos> и <eos>; encode(text, add_special_tokens=False) возвращает только текст.

    save_path позволяет сохранить все настройки и словарь в один JSON-файл.
    Загрузка: Tokenizer.from_file(str(save_path)).
    """
    if isinstance(dataset, ds.DatasetDict):
        if "train" not in dataset:
            raise ValueError("DatasetDict должен содержать часть 'train'.")
        dataset = dataset["train"]
    if not isinstance(dataset, ds.Dataset):
        raise TypeError("Ожидается datasets.Dataset или datasets.DatasetDict с частью train.")
    if not len(dataset):
        raise ValueError("Обучающий датасет пуст.")

    columns = [columns] if isinstance(columns, str) else list(columns)
    if not columns or any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("Укажите непустые названия текстовых столбцов.")
    if len(columns) != len(set(columns)):
        raise ValueError("Названия столбцов не должны повторяться.")
    missing = set(columns) - set(dataset.column_names)
    if missing:
        raise ValueError(f"Столбцы не найдены: {', '.join(sorted(missing))}.")

    alphabet = pre_tokenizers.ByteLevel.alphabet()
    for name, value, minimum in (
        ("vocab_size", vocab_size, len(alphabet) + len(SPECIAL_TOKENS)),
        ("min_frequency", min_frequency, 0),
        ("batch_size", batch_size, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} должен быть целым числом не меньше {minimum}.")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=alphabet,
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        _text_batches(dataset, columns, batch_size),
        trainer=trainer,
        length=len(dataset) * len(columns),
    )
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    if save_path is not None:
        save_path = Path(save_path).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path))
    return tokenizer
