"""Generate the two thin notebooks around the tested shared runner."""
import json
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]


def cell(kind,source):
    value={'cell_type':kind,'metadata':{},'source':textwrap.dedent(source).strip()+'\n'}
    if kind=='code':
        value.update(execution_count=None,outputs=[])
    return value


def make_notebooks(manifest_sha):
    for architecture,experiment in [('t5','t5_007_1m'),('copy','copy_008_1m')]:
        title='T5-style: Pre-RMSNorm + GEGLU' if architecture=='t5' else 'Transformer с механизмом копирования'
        cells=[cell('markdown',f'''
            # {title}

            С нуля, тот же BPE8000 и **1 000 000 train / 20 000 val**, seed42.
            Архитектура8+8, d_model1024; одна эпоха, effective batch256,
            LR2e-4, warmup5% и cosine, валидация каждые1000 шагов и в конце.
            T5: GEGLU d_ff2816; copy-контроль: BART FF4096 + отдельная copy-head.

            Добавьте приватный dataset с `nlp_hw2_kaggle.zip` через **Add Input**.
            Выберите доступный GPU (T4 x2 использует DDP), включите Internet для
            установки библиотек. Приватность ноутбука и dataset сохраняйте Private.
            Первая проверка обучает отдельную модель2 шага, затем полный запуск
            начинает обучение заново с seed42. Готовых весов нет.
        '''),cell('code',f'''
            from pathlib import Path
            import hashlib, json, os, subprocess, sys

            ARCHITECTURE = {architecture!r}
            RUN_NAME = {experiment!r}
            MICRO_BATCH = 16       # на GPU; уменьшите до8 при OOM, global batch останется256
            GLOBAL_BATCH = 256
            LEARNING_RATE = 2e-4
            PRECISION = 'auto'     # BF16 при поддержке; иначе FP16 (T4)
            USE_COMPILE = False   # опционально; TF32-конфликт исправлен в общем Trainer
            GRADIENT_CHECKPOINTING = True
            RESUME_FROM = None     # полный путь к прежнему checkpoint-N, при необходимости
            EXPECTED_MANIFEST_SHA = {manifest_sha!r}

            candidates = list(Path('/kaggle/input').rglob('bundle_manifest.json'))
            matches = [p for p in candidates if hashlib.sha256(p.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA]
            if len(matches) != 1:
                raise RuntimeError('Добавьте dataset с подготовленным пакетом этой версии; найдено совпадений: ' + str(len(matches)))
            BUNDLE_ROOT = matches[0].parent
            OUTPUT = Path('/kaggle/working') / RUN_NAME
            print('Bundle:', BUNDLE_ROOT, 'Output:', OUTPUT)
        '''),cell('code','''
            # Сохраняем рабочий PyTorch/CUDA из образа Kaggle.
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                str(BUNDLE_ROOT / 'requirements-kaggle.txt')], check=True)
            sys.path.insert(0, str(BUNDLE_ROOT))
            import torch, transformers
            print('PyTorch:', torch.__version__, 'Transformers:', transformers.__version__)
            if not torch.cuda.is_available():
                raise RuntimeError('Выберите GPU в Settings → Accelerator перед запуском.')
            NUM_GPUS = min(2, torch.cuda.device_count())
            print('GPU:', [torch.cuda.get_device_name(i) for i in range(NUM_GPUS)])
            from kaggle_experiments import accumulation_steps, load_bundle
            print('GPUs:', NUM_GPUS, 'micro batch:', MICRO_BATCH,
                'accumulation:', accumulation_steps(GLOBAL_BATCH, MICRO_BATCH, NUM_GPUS),
                'effective batch:', GLOBAL_BATCH)
            data, tokenizer, identity = load_bundle(BUNDLE_ROOT,
                expected_manifest_sha=EXPECTED_MANIFEST_SHA)
            print(data)
            print('Original data provenance:', identity['selection']['identity'])
        '''),cell('code','''
            def launch(*, preflight):
                from uuid import uuid4
                destination = (Path('/kaggle/working') /
                    f'preflight_{ARCHITECTURE}_{uuid4().hex[:10]}') if preflight else OUTPUT
                command = [sys.executable, '-m', 'torch.distributed.run', '--master-addr=127.0.0.1', '--master-port=29501',
                    '--nproc_per_node', str(NUM_GPUS), str(BUNDLE_ROOT / 'train_kaggle.py'),
                    '--bundle-root', str(BUNDLE_ROOT), '--architecture', ARCHITECTURE,
                    '--output-dir', str(destination), '--micro-batch', str(MICRO_BATCH),
                    '--global-batch', str(GLOBAL_BATCH), '--learning-rate', str(LEARNING_RATE),
                    '--precision', PRECISION, '--expected-manifest-sha', EXPECTED_MANIFEST_SHA]
                if preflight:
                    command.append('--preflight')
                elif RESUME_FROM:
                    command += ['--resume-from', str(RESUME_FROM)]
                if USE_COMPILE:
                    command.append('--torch-compile')
                if not GRADIENT_CHECKPOINTING:
                    command.append('--no-gradient-checkpointing')
                environment = os.environ | {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                    'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1', 'MPLBACKEND': 'Agg',
                    'ACCELERATE_DYNAMO_USE_DYNAMIC': 'true', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}
                print('Running:', ' '.join(command), flush=True)
                subprocess.run(command, env=environment, check=True)

            # Реальные полные модели на GPU,2 optimizer steps, finite loss/gradients,
            # validation, checkpoint и экспорт. Ошибка остановит notebook до полного обучения.
            launch(preflight=True)
        '''),cell('code','''
            # Полный эксперимент: одна эпоха на точной фиксированной выборке.
            launch(preflight=False)
        '''),cell('code','''
            from IPython.display import Image, display
            result = json.loads((OUTPUT / 'complete.json').read_text())
            display(result)
            display(Image(filename=str(OUTPUT / 'training.png')))
            print('Финальная модель:', OUTPUT / 'final')
            print('Checkpoints:', sorted(p.name for p in OUTPUT.glob('checkpoint-*')))
            print('Метрики:', OUTPUT / 'complete.json')
        '''),cell('markdown','''
            Полный запуск сохраняет HF-веса и tokenizer в `final/`, полные
            `checkpoint-N` с optimizer/scheduler/RNG и `run_manifest.json`,
            `log_history.json`, `training.png`, `complete.json`.
            Для автономного запуска выберите **Save Version → Save & Run All**
            вместо ручного Run All: Kaggle выполнит notebook и сохранит output.
            При разрыве сессии прикрепите сохранённый output как input
            и укажите `RESUME_FROM`; данные и настройки должны совпадать.

            Copy-веса загружаются после `import models` через
            `AutoModelForSeq2SeqLM.from_pretrained(...)` (тип `copy_bart`).
            Для SDPA после загрузки T5 вызовите `enable_t5_sdpa(model)` из
            `models.t5_attention`. Генерация из обоих моделей поддерживает beam.
            Validation loss не является pass@1; после отбора отдельно проверяем
            качество кода и лимит180 секунд на T4 с beam4.
        ''')]
        notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
            'language_info':{'name':'python','version':'3.11'},'kaggle':{'isInternetEnabled':True,'language':'python','sourceType':'notebook'}},
            'nbformat':4,'nbformat_minor':5}
        for i,c in enumerate(cells):
            c['id']=f'{architecture}-{i}'
            if c['cell_type']=='code':
                compile(c['source'],f'kaggle_{architecture}.ipynb cell{i}','exec')
        (ROOT/f'kaggle_{architecture}.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')


if __name__=='__main__':
    metadata=json.loads((ROOT/'artifacts/kaggle_20261019/nlp_hw2_kaggle.json').read_text())
    make_notebooks(metadata['bundle_manifest_sha256'])
