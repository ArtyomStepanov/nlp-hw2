# Генерация Python-кода по тексту задачи

## Установка

Python 3.10+; команды выполняются из корня репозитория.

```bash
git clone https://github.com/ArtyomStepanov/nlp-hw2.git
cd nlp-hw2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) фиксирует проверенные версии, включая
PyTorch 2.9.1 и Transformers 5.10.2. Для обучения на NVIDIA нужен PyTorch
с поддержкой установленной CUDA. В Colab/Kaggle сохраняйте готовую сборку
PyTorch/CUDA среды; устанавливайте остальные зависимости через
[requirements-kaggle.txt](requirements-kaggle.txt). После изменения библиотек,
которые уже импортировались в ноутбуке, перезапустите runtime/kernel.

## Где находятся файлы

| Путь | Назначение |
| --- | --- |
| [train.ipynb](train.ipynb) | Загрузка данных, BPE и обучение базовой модели. |
| [models/](models/) | Архитектуры и конфигурация модели. |
| [training.py](training.py), [train_distributed.py](train_distributed.py) | Обучение и запуск на нескольких GPU. |
| [train_kaggle.py](train_kaggle.py), [kaggle_experiments.py](kaggle_experiments.py) | CLI для заранее подготовленного токенизированного набора. |
| [inference.py](inference.py) | Загрузка весов и батчевая генерация `generate_code`. |
| [evaluation.py](evaluation.py), [evaluate_checkpoint.py](evaluate_checkpoint.py) | Локальная оценка кода на отложенных примерах. |
| [colab_final_submission.ipynb](colab_final_submission.ipynb) | Готовый Colab-инференс модели002, checkpoint-360000. |
| [colab_a100_full_inference.ipynb](colab_a100_full_inference.ipynb) | Готовый Colab-инференс полной A100-модели, checkpoint-180000. |
| `data/tinysms_qa_12m/` | Локальные исходные Parquet, скачиваются отдельно. |
| `artifacts/` | Локальные веса, кэши, результаты запусков и резервные копии; не входят в Git. |
| [reports/](reports/) | Сохранённые метрики, графики и контрольные суммы. |
| [tests/](tests/) | Проверки реализации; запуск: `python -m pytest -q`. |
| [docs/PROJECT_NOTES.md](docs/PROJECT_NOTES.md) | Дополнительное описание решения и журнал экспериментов. |

## Инференс в Colab: получить JSONL для бота

1. Скачайте **один** архив из
   [релиза с обеими моделями](https://github.com/ArtyomStepanov/nlp-hw2/releases/tag/models-2026-10-24)
   через браузер.
2. Скачайте соответствующий `.ipynb` из таблицы ниже и откройте его в Colab
   через **File → Upload notebook**.
3. В **Runtime → Change runtime type** выберите **T4 GPU**.
4. Загрузите ZIP через боковую панель **Files → Upload** в `/content/`.
   Распаковывать архив вручную не нужно.
5. Выполните ячейки сверху вниз. Ноутбук проверит SHA256, загрузит веса,
   обработает все 1919 вопросов и предложит скачать готовый JSONL.

| Модель | Архив весов, BPE, кода и вопросов | Ноутбук | Итоговый файл для бота |
| --- | --- | --- | --- |
| Модель002, best360000 | [final_checkpoint360000_colab.zip](https://github.com/ArtyomStepanov/nlp-hw2/releases/download/models-2026-10-24/final_checkpoint360000_colab.zip) | [colab_final_submission.ipynb](colab_final_submission.ipynb) | `transformer_002_checkpoint360000_beam4_t4.jsonl` |
| Полный BART A100, best180000 | [a100_full_best180000_colab.zip](https://github.com/ArtyomStepanov/nlp-hw2/releases/download/models-2026-10-24/a100_full_best180000_colab.zip) | [colab_a100_full_inference.ipynb](colab_a100_full_inference.ipynb) | `a100_full_best180000_beam4_t4.jsonl` |

Архивы опубликованы как **Release assets**: клонирование репозитория не скачивает
веса. Каждый архив самодостаточен для инференса, обучающий датасет не нужен.
GitHub-токен в Colab вводить не требуется. Настройки ноутбуков: FP16 autocast,
beam 4, batch 128, вход/выход до 256 токенов. Ячейки инференса не обучают модель
и не выполняют сгенерированный код. После смены GPU или среды фактическое время
генерации нужно проверить по выводу ноутбука.

## Локальный инференс на своих вопросах

Распакуйте выбранный архив, например в `artifacts/inference_bundle/`.
В архиве модели002 папка `checkpoint/` находится в корне; в A100-архиве —
внутри `a100_full_best180000_t4/`. Она содержит веса, конфигурацию и
`tokenizer.json`. Ниже пример для модели002; для A100 добавьте этот вложенный
каталог в путь. Запускайте код из корня репозитория:

```python
from datasets import Dataset
from inference import InferenceConfig, generate_code, load_inference_model

model, tokenizer = load_inference_model(
    "artifacts/inference_bundle/checkpoint", device="cuda:0"
)
questions = Dataset.from_dict({"question": [
    "A box contains 12 pencils. How many pencils are in 3 boxes?"
]})
result = generate_code(questions, model, InferenceConfig(
    tokenizer=tokenizer,
    device="cuda:0",
    mixed_precision="fp16",
    batch_size=128,
    num_beams=4,
    max_source_length=256,
    max_new_tokens=256,
    truncate_source=False,
    output_dir="artifacts/inference/my_questions",
))
print(result.codes[0])
```

Для CPU замените `device` на `"cpu"` в обоих вызовах и
`mixed_precision` на `"no"`. Результаты записываются в `output_dir`;
для повторного запуска укажите новую или пустую папку. `result.codes`
содержит по одному ответу на вопрос в исходном порядке.

## Обучение

Откройте [train.ipynb](train.ipynb) из корня проекта. Ноутбук скачивает
`SlavaYus/tinysms_qa_12m` в `data/tinysms_qa_12m/`, выделяет train/val/test
из исходного train, обучает BPE и вызывает `train_model`. Перед запуском
настройте `TransformerConfig` и `TrainingConfig`, в том числе размеры выборок,
число эпох и **новую** `output_dir`. Веса и история сохраняются в `artifacts/`.
Официальный test содержит только вопросы и используется для финального инференса.

Для уже подготовленного набора из 1 млн train /20 тыс. validation доступен CLI:

```bash
python train_kaggle.py \
  --bundle-root /path/to/prepared_bundle \
  --architecture bart \
  --output-dir artifacts/experiments/bart_new \
  --micro-batch 16 --global-batch 256 --precision fp16
```

`prepared_bundle` должен содержать `bundle_manifest.json`, `data/`,
`selection/` и `tokenizer.json`; это **обучающий** пакет, не Colab-архив весов
из релиза. Его подготовка реализована в
[tools/prepare_kaggle_bundle.py](tools/prepare_kaggle_bundle.py) на основе
сохранённой выборки; подробности и происхождение данных приведены в
[дополнительных материалах](docs/PROJECT_NOTES.md).
Для проверки двух шагов перед обучением добавьте `--preflight` и используйте
отдельную `output_dir`. Доступные параметры: `python train_kaggle.py --help`.
