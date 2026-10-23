"""Sequential, equal-budget H100 screen using an existing tokenized train/val cache."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def select_indices(total, count, seed):
    if not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= total:
        raise ValueError('Subset size must be a positive integer no greater than the source split')
    return np.random.default_rng(seed).choice(total, size=count, replace=False)


def build_trials(root, micro_batch=128, specs=None):
    if micro_batch not in (128, 256):
        raise ValueError('Use physical batch 128 or 256, with global batch fixed at 256')
    if specs is None:
        specs = [dict(encoder_layers=e, decoder_layers=d, learning_rate=lr)
                 for e, d in ((10, 10), (12, 8)) for lr in (1e-4, 2e-4)]
    if not isinstance(specs, list) or not specs:
        raise ValueError('trials must be a nonempty list')
    trials = []
    names = set()
    for spec in specs:
        required = {'encoder_layers', 'decoder_layers', 'learning_rate'}
        if not isinstance(spec, dict) or not required <= set(spec) or set(spec) - required - {'architecture'}:
            raise ValueError('Each trial needs encoder_layers, decoder_layers, learning_rate and optional architecture')
        architecture = spec.get('architecture', 'bart')
        if architecture not in ('bart', 't5'):
            raise ValueError('architecture must be bart or t5')
        encoder, decoder, lr = spec['encoder_layers'], spec['decoder_layers'], spec['learning_rate']
        if any(type(n) is not int or n <= 0 for n in (encoder, decoder)):
            raise ValueError('Layer counts must be positive integers')
        if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
            raise ValueError('learning_rate must be positive and finite')
        mantissa, exponent = f'{lr:.12e}'.split('e')
        lr_label = mantissa.rstrip('0').rstrip('.') + 'e' + exponent
        name = f'e{encoder}_d{decoder}_lr{lr_label}'
        if architecture == 't5':
            name = 't5_' + name
        if name in names:
            raise ValueError(f'Duplicate trial output: {name}')
        names.add(name)
        trials.append(dict(name=name, output_dir=str(Path(root).resolve()/'trials'/name),
                           encoder_layers=encoder, decoder_layers=decoder, learning_rate=lr,
                           train_batch_size=micro_batch, gradient_accumulation_steps=256//micro_batch,
                           num_train_epochs=1, seed=42))
        if architecture == 't5':
            trials[-1].update(architecture='t5', d_model=1024, ffn_dim=2816, attention_heads=8)
    return trials


def build_trial_model(trial, tokenizer, smoke=False):
    from models import BaselineTransformer, T5Transformer, TransformerConfig
    architecture = trial.get('architecture', 'bart')
    config = TransformerConfig.from_tokenizer(tokenizer,
        d_model=32 if smoke else trial.get('d_model', 1024),
        encoder_layers=1 if smoke else trial['encoder_layers'],
        decoder_layers=1 if smoke else trial['decoder_layers'],
        attention_heads=4 if smoke else trial.get('attention_heads', 8),
        ffn_dim=64 if smoke else trial.get('ffn_dim', 4096))
    if architecture == 't5':
        return T5Transformer(config)
    if architecture == 'bart':
        return BaselineTransformer(config)
    raise ValueError('architecture must be bart or t5')


def trial_action(output, trial):
    output = Path(output)
    if not output.exists() or not any(output.iterdir()):
        return 'new'
    manifest = output/'trial.json'
    if not manifest.is_file() or json.loads(manifest.read_text()) != trial:
        raise ValueError(f'Refusing to overwrite a different experiment: {output}')
    if (output/'complete.json').is_file():
        return 'skip'
    for checkpoint in output.glob('checkpoint-*'):
        if all((checkpoint/name).is_file() for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth')):
            return 'resume'
    raise ValueError(f'Incomplete trial without a resumable checkpoint: {output}; choose a new sweep output directory')


def prepared_root(config):
    return Path(config.get('prepared_data_dir', config['output_dir'])).resolve()


def prepare_data(config):
    import datasets as ds
    root = prepared_root(config)
    source = Path(config['tokenized_dataset']).resolve()
    tokenizer = Path(config['tokenizer']).resolve()
    identity = {k: config[k] for k in ('train_rows', 'validation_rows', 'seed')}
    identity.update(source=str(source), tokenizer_sha256=digest(tokenizer))
    manifest_path = root/'data_manifest.json'
    data_path = root/'data'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['identity'] != identity or not (data_path/'dataset_dict.json').is_file():
            raise ValueError('Prepared data does not match this sweep configuration')
        return old
    if 'prepared_data_dir' in config:
        raise ValueError('Shared prepared data must already exist with a matching manifest')
    if data_path.exists():
        raise ValueError('Partial prepared data exists; choose a new output directory')
    dataset = ds.load_from_disk(str(source))
    root.mkdir(parents=True, exist_ok=True)
    selected = {}
    details = {}
    for name, count, seed in [('train', config['train_rows'], config['seed']),
                              ('val', config['validation_rows'], config['seed']+1)]:
        split = dataset[name]
        required = {'input_ids', 'attention_mask', 'labels', 'length'}
        if not required.issubset(split.column_names):
            raise ValueError(f'{name} is not the existing length-filtered tokenized cache')
        indices = select_indices(len(split), count, seed)
        path = root/f'{name}_indices.npy'
        np.save(path, indices)
        selected[name] = split.select(indices)
        details[name] = dict(source_rows=len(split), selected_rows=count,
                             source_fingerprint=split._fingerprint, indices_sha256=digest(path), seed=seed)
    ds.DatasetDict(selected).save_to_disk(str(data_path))
    manifest = dict(identity=identity, splits=details, created_at=datetime.now(timezone.utc).isoformat())
    write_json(manifest_path, manifest)
    return manifest


def run_trial(config, index, smoke=False):
    import datasets as ds
    import torch
    from tokenizers import Tokenizer
    from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast, Seq2SeqTrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from training import _TokenLossTrainer, _TokenLossMetric, _eval_statistics, _token_normalized_loss

    root = Path(config['output_dir']).resolve()
    jobs = build_trials(root, config['micro_batch'], config.get('trials'))
    if not 0 <= index < len(jobs):
        raise ValueError('Trial index is outside the configured grid')
    trial = jobs[index]
    manifest = prepare_data(config)
    data_root = prepared_root(config)
    trial['data_manifest_sha256'] = digest(data_root/'data_manifest.json')
    trial['smoke'] = smoke
    if smoke:
        trial['output_dir'] = str(root/'smoke')
    output = Path(trial['output_dir'])
    action = trial_action(output, trial)
    if action == 'skip':
        return
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'trial.json', trial)
    data = ds.load_from_disk(str(data_root/'data'))
    if smoke:
        data = ds.DatasetDict({k: v.select(range(8)) for k, v in data.items()})
    assert len(data['train']) == (8 if smoke else config['train_rows'])
    assert len(data['val']) == (8 if smoke else config['validation_rows'])
    assert digest(config['tokenizer']) == manifest['identity']['tokenizer_sha256']
    set_seed(trial['seed'])
    torch.set_num_threads(4)
    tok = Tokenizer.from_file(config['tokenizer'])
    model = build_trial_model(trial, tok, smoke=smoke)
    model.network.config.use_cache = False
    if trial.get('architecture', 'bart') == 'bart':
        assert model.network.config._attn_implementation == 'sdpa'
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token='<bos>', eos_token='<eos>',
        pad_token='<pad>', unk_token='<unk>', model_max_length=512, clean_up_tokenization_spaces=False,
        model_input_names=['input_ids', 'attention_mask'])
    batch = 4 if smoke else trial['train_batch_size']
    accumulation = 1 if smoke else trial['gradient_accumulation_steps']
    args = Seq2SeqTrainingArguments(
        output_dir=str(output), num_train_epochs=1, max_steps=-1,
        per_device_train_batch_size=batch, per_device_eval_batch_size=4 if smoke else 128,
        gradient_accumulation_steps=accumulation, learning_rate=trial['learning_rate'],
        weight_decay=.01, warmup_steps=.05, lr_scheduler_type='cosine', max_grad_norm=1.,
        optim='adamw_torch_fused', train_sampling_strategy='group_by_length', length_column_name='length',
        torch_compile=not smoke, bf16=True, fp16=False, average_tokens_across_devices=True,
        eval_strategy='steps', eval_steps=1 if smoke else 1000, save_strategy='epoch', save_total_limit=1,
        load_best_model_at_end=False, logging_steps=1 if smoke else 100, logging_first_step=True,
        report_to='none', prediction_loss_only=False, batch_eval_metrics=True, predict_with_generate=False,
        dataloader_num_workers=0 if smoke else 4, dataloader_pin_memory=True,
        seed=42, data_seed=42, label_names=['labels'])
    trainer = _TokenLossTrainer(model=model.network, args=args,
        train_dataset=data['train'], eval_dataset=data['val'], processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model.network, pad_to_multiple_of=8, label_pad_token_id=-100),
        compute_loss_func=_token_normalized_loss, compute_metrics=_TokenLossMetric(),
        preprocess_logits_for_metrics=_eval_statistics)
    trainer.model_accepts_loss_kwargs = False
    trainer._loss_shifts_labels = False
    write_json(output/'model_config.json', model.config.to_dict())
    write_json(output/'hf_model_config.json', model.network.config.to_dict())
    import transformers
    write_json(output/'environment.json', dict(torch=str(torch.__version__), transformers=transformers.__version__,
        device=torch.cuda.get_device_name(), attention_backend='t5_sdpa_adapter' if trial.get('architecture') == 't5' else 'sdpa'))
    (output/'trainer_args.json').write_text(args.to_json_string())
    tok.save(str(output/'tokenizer.json'))
    begin = time.perf_counter()
    trained = trainer.train(resume_from_checkpoint=get_last_checkpoint(str(output)) if action == 'resume' else None)
    # Final model after identical example budget; do not rank an earlier best checkpoint.
    metrics = trainer.evaluate()
    expected_steps = math.ceil(len(data['train'])/(batch*accumulation))
    assert trainer.state.global_step == expected_steps
    if not math.isfinite(metrics['eval_token_loss']):
        raise RuntimeError('Nonfinite final validation loss')
    write_json(output/'log_history.json', trainer.state.log_history)
    trainer.save_state()
    complete = dict(trial=trial, metrics=metrics, train_metrics=trained.metrics,
                    steps=trainer.state.global_step, train_rows=len(data['train']), validation_rows=len(data['val']),
                    parameters=sum(p.numel() for p in model.parameters()), wall_seconds=time.perf_counter()-begin,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    checkpoint=get_last_checkpoint(str(output)), completed_at=datetime.now(timezone.utc).isoformat())
    write_json(output/'complete.json', complete)
    print(json.dumps(complete, ensure_ascii=False), flush=True)


def run_sweep(config, config_path):
    import fcntl
    root = Path(config['output_dir']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root/'sweep.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_data(config)
        jobs = build_trials(root, config['micro_batch'], config.get('trials'))
        write_json(root/'grid.json', jobs)
        logdir = root/'logs'
        logdir.mkdir(exist_ok=True)
        summary = dict(status='running', completed=[], failures=[])
        write_json(root/'summary.json', summary)
        env = os.environ | dict(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
                               ACCELERATE_DYNAMO_USE_DYNAMIC='true', PYTHONUNBUFFERED='1')
        for i, trial in enumerate(jobs):
            print('START', trial['name'], flush=True)
            summary['active_trial'] = trial['name']
            write_json(root/'summary.json', summary)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            logfile = logdir/f'{trial["name"]}-{stamp}.log'
            with logfile.open('w') as log:
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--config', str(config_path),
                                         '--trial', str(i)], env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                summary['failures'].append(dict(trial=trial['name'], returncode=result.returncode, log=str(logfile)))
            else:
                completed = json.loads((Path(trial['output_dir'])/'complete.json').read_text())
                summary['completed'].append(completed)
                summary['completed'].sort(key=lambda x: x['metrics']['eval_token_loss'])
            print('DONE', trial['name'], result.returncode, flush=True)
            write_json(root/'summary.json', summary)
        summary['active_trial'] = None
        summary['status'] = 'complete' if not summary['failures'] else 'finished_with_failures'
        write_json(root/'summary.json', summary)
        if summary['failures']:
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--trial', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    path = args.config.resolve()
    config = json.loads(path.read_text())
    if args.prepare_only:
        prepare_data(config)
    elif args.smoke:
        run_trial(config, 0, smoke=True)
    elif args.trial is not None:
        run_trial(config, args.trial)
    else:
        run_sweep(config, path)


if __name__ == '__main__':
    main()
