"""Portable, equal-budget training on the already selected H100 Arrow dataset."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def load_bundle(root, *, expected_rows=(1000000,20000), expected_manifest_sha=None):
    """Check copied bytes, retain original source paths as provenance, never resample."""
    import datasets as ds
    import numpy as np
    from tokenizers import Tokenizer

    root = Path(root).resolve()
    manifest_path = root/'bundle_manifest.json'
    if expected_manifest_sha and file_sha256(manifest_path) != expected_manifest_sha:
        raise ValueError('Bundle manifest checksum does not match the notebook.')
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema'] != 1:
        raise ValueError('Unsupported bundle manifest schema.')
    derive_length = manifest.get('derive_length', False)
    if type(derive_length) is not bool:
        raise ValueError('derive_length must be a boolean manifest flag.')
    if derive_length and not expected_manifest_sha:
        raise ValueError('Derived lengths require a pinned expected_manifest_sha.')
    for entry in manifest['files']:
        path = (root/entry['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Bundle integrity: missing or unsafe path {entry['path']}")
        if path.stat().st_size != entry['bytes'] or file_sha256(path) != entry['sha256']:
            raise ValueError(f"Bundle integrity checksum/size mismatch: {entry['path']}")
    selection_path = root/'selection/data_manifest.json'
    if file_sha256(selection_path) != manifest['selection_sha256']:
        raise ValueError('Selection manifest checksum mismatch.')
    selection = json.loads(selection_path.read_text())
    identity = selection['identity']
    if (identity['train_rows'],identity['validation_rows']) != tuple(expected_rows) or identity['seed'] != 42:
        raise ValueError('Unexpected dataset selection sizes or seed.')
    if file_sha256(root/'tokenizer.json') != identity['tokenizer_sha256']:
        raise ValueError('Tokenizer checksum mismatch.')
    data = ds.load_from_disk(str(root/'data'))
    tokenizer = Tokenizer.from_file(str(root/'tokenizer.json'))
    for name,count in zip(('train','val'),expected_rows):
        indices = root/f'selection/{name}_indices.npy'
        if file_sha256(indices) != selection['splits'][name]['indices_sha256']:
            raise ValueError(f'{name} selection indices checksum mismatch.')
        if len(data[name]) != count or np.load(indices,mmap_mode='r').shape != (count,):
            raise ValueError(f'{name} size mismatch.')
        required_columns = {'input_ids','attention_mask','labels'}
        if not derive_length:
            required_columns.add('length')
        if not required_columns <= set(data[name].column_names):
            raise ValueError(f'{name} is not the existing pretokenized dataset.')
        if derive_length:
            import pyarrow.compute as pc
            # add_column would flatten an indexed view and rewrite all tokens.
            if data[name]._indices is not None:
                raise ValueError(f'{name} derived lengths require raw data without indices.')
            source_lengths = pc.list_value_length(data[name].data.column('input_ids'))
            target_lengths = pc.list_value_length(data[name].data.column('labels'))
            for lengths in (source_lengths, target_lengths):
                if (lengths.null_count or pc.min(lengths).as_py() is None
                        or pc.min(lengths).as_py() < 1 or pc.max(lengths).as_py() > 256):
                    raise ValueError(f'{name} token lengths violate the existing 1..256 contract.')
            lengths = pc.max_element_wise(source_lengths, target_lengths)
            if 'length' in data[name].column_names:
                if not pc.all(pc.equal(data[name].data.column('length'), lengths)).as_py():
                    raise ValueError(f'{name} stored length differs from its token lengths.')
            else:
                # Only a small integer column is allocated; Arrow token buffers
                # remain memory mapped and are never saved or tokenized again.
                data[name] = data[name].add_column('length', lengths)
        for row in data[name].select(range(min(16,count))):
            if not row['labels'] or row['labels'][-1] != tokenizer.token_to_id('<eos>'):
                raise ValueError('Target contract requires code+EOS without BOS.')
            if row['labels'][0] == tokenizer.token_to_id('<bos>'):
                raise ValueError('Target labels must exclude BOS.')
    return data,tokenizer,{'selection':selection,'bundle_sha256':file_sha256(manifest_path),
        'selection_sha256':manifest['selection_sha256']}


def accumulation_steps(global_batch, micro_batch, world_size):
    if any(type(x) is not int or x < 1 for x in (global_batch,micro_batch,world_size)):
        raise ValueError('Batch sizes and world size must be positive integers.')
    if global_batch % (micro_batch * world_size):
        raise ValueError('Global batch must be divisible by micro_batch * world_size.')
    return global_batch // (micro_batch * world_size)


def resolve_precision(precision, device):
    """Choose native BF16 only; T4 can allocate BF16 tensors but lacks its cores."""
    import torch
    native_bf16 = device.type=='cuda' and torch.cuda.is_bf16_supported(including_emulation=False)
    if precision=='auto':
        return ('bf16' if native_bf16 else 'fp16') if device.type=='cuda' else 'no'
    if precision in ('fp16','bf16') and device.type!='cuda':
        raise ValueError('FP16/BF16 training requires CUDA here.')
    if precision=='bf16' and not native_bf16:
        raise ValueError('This GPU lacks native BF16; use FP16 on T4.')
    return precision


def checked_token_loss(outputs, labels, num_items_in_batch=None):
    """Fail before backward; HF's log filter can otherwise hide an infinite CE."""
    import torch
    from training import _token_normalized_loss
    loss = _token_normalized_loss(outputs,labels,num_items_in_batch)
    if not torch.isfinite(loss.detach()).item():
        raise FloatingPointError('Nonfinite training loss; no optimizer update was performed.')
    return loss


def _safety_and_artifacts_callback(out, manifest, get_scaler):
    import torch
    from accelerate.optimizer import AcceleratedOptimizer
    from transformers import TrainerCallback

    class SafetyAndArtifacts(TrainerCallback):
        def __init__(self):
            self.overflow_count = 0
            self.overflow_step = None
            self.overflow_scale = None

        def amp_diagnostics(self):
            scaler = get_scaler()
            return {'overflow_count':self.overflow_count,
                'final_loss_scale':scaler.get_scale() if scaler is not None and scaler.is_enabled() else None,
                'scope':'current_process_session'}

        def on_pre_optimizer_step(self,args,state,control,model=None,optimizer=None,**kwargs):
            norms = [p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]
            if not norms:
                raise FloatingPointError('Missing gradients; stop before updating weights.')
            if not torch.isfinite(torch.stack(norms)).all().item():
                scaler = get_scaler()
                while isinstance(optimizer,AcceleratedOptimizer):
                    optimizer = optimizer.optimizer
                # Trainer has already unscaled/clipped. Only that optimizer's
                # recorded overflow permits GradScaler to finish skip/backoff.
                found_inf = (scaler._found_inf_per_device(optimizer)
                    if scaler is not None and scaler.is_enabled() and optimizer is not None else {})
                if any(value.item()>0 for value in found_inf.values()):
                    self.overflow_count += 1
                    self.overflow_step = state.global_step+1
                    self.overflow_scale = scaler.get_scale()
                    return
                raise FloatingPointError('Nonfinite gradients; stop before updating weights.')

        def on_log(self,args,state,control,logs=None,**kwargs):
            logs = {} if logs is None else logs
            scaler = get_scaler() if self.overflow_count else None
            if scaler is not None:
                logs['amp_overflow_count'] = self.overflow_count
                logs['amp_loss_scale'] = scaler.get_scale()
            for name,value in logs.items():
                if isinstance(value,(int,float)) and not math.isfinite(value):
                    if (name=='grad_norm' and self.overflow_step==state.global_step
                            and scaler is not None and scaler.get_scale()<self.overflow_scale):
                        logs[name] = None
                    else:
                        raise FloatingPointError(f'Nonfinite metric: {name}={value}')
            # Trainer.log appends a copy before dispatching callbacks. Sanitize
            # that row too, including the copy later saved in trainer_state.json.
            if state.log_history and state.log_history[-1].get('step')==state.global_step:
                state.log_history[-1].update(logs)
            if state.is_world_process_zero:
                write_json(out/'log_history.json',state.log_history)

        def on_save(self,args,state,control,**kwargs):
            if state.is_world_process_zero:
                write_json(out/f'checkpoint-{state.global_step}'/'run_manifest.json',manifest)

    return SafetyAndArtifacts()


@dataclass
class KaggleConfig:
    architecture: str
    output_dir: str | Path
    micro_batch: int = 16
    global_batch: int = 256
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.
    eval_steps: int = 1000
    precision: str = 'auto'
    gradient_checkpointing: bool = True
    torch_compile: bool = False
    torch_compile_backend: str = 'inductor'
    use_cpu: bool = False
    tiny_model: bool = False
    preflight: bool = False
    max_steps: int = -1
    expected_rows: tuple[int,int] = (1000000,20000)
    expected_manifest_sha: str | None = None
    seed: int = 42
    early_stopping_patience: int | None = None

    def __post_init__(self):
        if (self.early_stopping_patience is not None
                and (type(self.early_stopping_patience) is not int or self.early_stopping_patience < 1)):
            raise ValueError('early_stopping_patience must be a positive integer or None.')


def config_manifest(config: KaggleConfig):
    """Serialize scientific settings while preserving historical resume identity."""
    values = asdict(config)
    values.pop('output_dir')
    values['expected_rows'] = list(config.expected_rows)
    if values['early_stopping_patience'] is None:
        values.pop('early_stopping_patience')
    return values


def _warm_start_identity(spec, config, identity, architecture, tokenizer_path):
    """Validate a weights-only initialization; parent optimizer state is never read."""
    if config.architecture != 'bart' or config.early_stopping_patience is None:
        raise ValueError('Warm start requires BART and explicit early stopping.')
    if set(spec) != {'checkpoint','expected_weight_sha','dropout','attention_dropout'}:
        raise ValueError('Warm start requires checkpoint, expected_weight_sha and both dropout settings.')
    checkpoint = Path(spec['checkpoint']).resolve()
    parent = checkpoint.parent
    previous = json.loads((parent/'run_manifest.json').read_text())
    completed = json.loads((parent/'complete.json').read_text())
    best_step = completed.get('best_global_step')
    if type(best_step) is not int or best_step < 1 or checkpoint != parent/f'checkpoint-{best_step}':
        raise ValueError('Warm start requires the completed parent best checkpoint.')
    if Path(completed['best_model_checkpoint']).resolve() != checkpoint:
        raise ValueError('Parent best checkpoint provenance differs.')
    if completed.get('final_weights_source') != 'best_eval_token_loss':
        raise ValueError('Parent must have exported its verified best weights.')
    for key in ('bundle_sha256','selection_sha256'):
        if previous.get(key) != identity[key]:
            raise ValueError(f'Warm start parent {key} differs from the pinned training bundle.')
    if previous.get('train_rows') != config.expected_rows[0] or previous.get('validation_rows') != config.expected_rows[1]:
        raise ValueError('Parent training split sizes differ.')
    expected = dict(architecture.to_dict())
    old = dict(previous['model_config'])
    for key in ('dropout','attention_dropout'):
        expected.pop(key); old.pop(key)
    if expected != old or previous['config']['architecture'] != 'bart':
        raise ValueError('Warm start changes architecture beyond the requested dropout.')
    if json.loads((checkpoint/'run_manifest.json').read_text()) != previous:
        raise ValueError('Parent checkpoint manifest differs.')
    if json.loads((checkpoint/'trainer_state.json').read_text()).get('global_step') != best_step:
        raise ValueError('Parent checkpoint step differs.')
    weights = checkpoint/'model.safetensors'
    if file_sha256(weights) != spec['expected_weight_sha']:
        raise ValueError('Parent weight SHA mismatch.')
    tokenizer_sha = file_sha256(tokenizer_path)
    if any(file_sha256(path) != tokenizer_sha for path in (parent/'source_tokenizer.json',checkpoint/'tokenizer.json')):
        raise ValueError('Parent tokenizer SHA mismatch.')
    out = Path(config.output_dir).resolve()
    if out.is_relative_to(parent) or parent.is_relative_to(out):
        raise ValueError('Continuation output must be separate from immutable parent artifacts.')
    return dict(checkpoint=str(checkpoint),weight_sha256=spec['expected_weight_sha'],
        parent_manifest_sha256=file_sha256(parent/'run_manifest.json'),
        parent_completion_sha256=file_sha256(parent/'complete.json'),parent_best_step=best_step,
        dropout=spec['dropout'],attention_dropout=spec['attention_dropout'],
        optimizer_initialization='fresh_adamw_new_schedule',seed=42)


def run_experiment(bundle_root, config: KaggleConfig, *, resume_from_checkpoint=None, warm_start=None):
    import torch
    import transformers
    from accelerate import PartialState
    from packaging.version import Version
    from transformers import (DataCollatorForSeq2Seq, EarlyStoppingCallback, PreTrainedTokenizerFast,
        Seq2SeqTrainingArguments, TrainerState, set_seed)
    from models import BaselineTransformer, CopyTransformer, T5Transformer, TransformerConfig
    from training import (_TokenLossTrainer, _TokenLossMetric,
        _eval_statistics, _on_main_process, _validate_resume)

    if Version(torch.__version__.split('+')[0]) < Version('2.6'):
        raise RuntimeError('Use a Kaggle image with PyTorch>=2.6; keep its CUDA build.')
    if config.architecture not in ('t5','copy','bart'):
        raise ValueError('architecture must be t5, copy or bart.')
    if config.precision not in ('auto','no','fp16','bf16'):
        raise ValueError('precision must be auto, no, fp16 or bf16.')
    if config.seed != 42 or config.learning_rate <= 0 or config.num_train_epochs <= 0:
        raise ValueError('Keep selection seed42 and positive LR/epochs.')
    state = PartialState(cpu=config.use_cpu)
    if state.device.type == 'cpu':
        state.device = torch.device('cpu')
    accumulation = accumulation_steps(config.global_batch,config.micro_batch,state.num_processes)
    precision = resolve_precision(config.precision,state.device)
    data,tokenizer,identity = load_bundle(bundle_root,expected_rows=config.expected_rows,
        expected_manifest_sha=config.expected_manifest_sha)
    if config.preflight:
        import datasets as ds
        data = ds.DatasetDict({'train':data['train'].select(range(min(len(data['train']),config.global_batch*2))),
            'val':data['val'].select(range(min(len(data['val']),64)))})
    set_seed(config.seed)
    torch.set_num_threads(2)
    dropout_options = {}
    if warm_start is not None:
        for name in ('dropout','attention_dropout'):
            value = warm_start.get(name)
            if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f'Warm start {name} must be finite and in [0,1).')
            dropout_options[name] = value
    architecture = TransformerConfig.from_tokenizer(tokenizer,
        d_model=16 if config.tiny_model else 1024,
        encoder_layers=1 if config.tiny_model else 8,
        decoder_layers=1 if config.tiny_model else 8,
        attention_heads=2 if config.tiny_model else 8,
        ffn_dim=32 if config.tiny_model else (2816 if config.architecture=='t5' else 4096),**dropout_options)
    wrapper = {'t5':T5Transformer,'copy':CopyTransformer,'bart':BaselineTransformer}[config.architecture](architecture)
    network = wrapper.network
    network.config.use_cache = False
    out = Path(config.output_dir).resolve()
    run_config = config_manifest(config)
    manifest = {'config':run_config,'precision':precision,'world_size':state.num_processes,
        'gradient_accumulation_steps':accumulation,'bundle_sha256':identity['bundle_sha256'],
        'selection_sha256':identity['selection_sha256'],'model_config':architecture.to_dict(),
        'train_rows':len(data['train']),'validation_rows':len(data['val'])}
    if warm_start is not None:
        manifest['warm_start'] = _warm_start_identity(warm_start,config,identity,architecture,
            Path(bundle_root)/'tokenizer.json')
    checkpoint = Path(resume_from_checkpoint).resolve() if resume_from_checkpoint else None
    resume_step = 0
    if checkpoint:
        previous = json.loads((checkpoint/'run_manifest.json').read_text())
        if previous != manifest:
            raise ValueError('Resume requires the same data, architecture, batch layout, precision and schedule.')
        resume_step = _validate_resume(checkpoint,network,tokenizer)

    def initialize_output():
        if not checkpoint and out.exists() and any(out.iterdir()):
            raise FileExistsError('Output already exists; pass --resume-from checkpoint-N or choose a new run name.')
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'run_manifest.json',manifest)
    _on_main_process(state,initialize_output)
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer,bos_token='<bos>',eos_token='<eos>',
        pad_token='<pad>',unk_token='<unk>',model_max_length=512,
        model_input_names=['input_ids','attention_mask'],clean_up_tokenization_spaces=False)
    steps = 2 if config.preflight else config.max_steps
    early_stopping = config.early_stopping_patience is not None
    args = Seq2SeqTrainingArguments(output_dir=str(out),num_train_epochs=config.num_train_epochs,max_steps=steps,
        per_device_train_batch_size=config.micro_batch,per_device_eval_batch_size=config.micro_batch,
        gradient_accumulation_steps=accumulation,learning_rate=config.learning_rate,
        weight_decay=.01,warmup_steps=.05,lr_scheduler_type='cosine',max_grad_norm=1.,
        optim='adamw_torch_fused' if state.device.type=='cuda' else 'adamw_torch',
        train_sampling_strategy='group_by_length',length_column_name='length',
        torch_compile=config.torch_compile,
        # HF treats any explicit backend as an opt-in, even with torch_compile=False.
        torch_compile_backend=config.torch_compile_backend if config.torch_compile else None,
        bf16=precision=='bf16',fp16=precision=='fp16',use_cpu=config.use_cpu,
        accelerator_config={'use_configured_state':True},average_tokens_across_devices=True,
        ddp_find_unused_parameters=False,gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        eval_strategy='steps',eval_steps=1 if config.preflight else config.eval_steps,
        save_strategy='steps',save_steps=(1 if warm_start is not None else 2) if config.preflight else config.eval_steps,
        save_total_limit=1 if config.preflight else 2,
        load_best_model_at_end=early_stopping,
        metric_for_best_model='eval_token_loss' if early_stopping else None,
        greater_is_better=False if early_stopping else None,
        restore_callback_states_from_checkpoint=early_stopping,
        report_to='none',logging_steps=1 if config.preflight or config.tiny_model else 100,
        logging_first_step=True,logging_nan_inf_filter=False,batch_eval_metrics=True,predict_with_generate=False,
        dataloader_num_workers=0 if config.use_cpu else 2,dataloader_pin_memory=state.device.type=='cuda',
        seed=42,data_seed=42,label_names=['labels'])
    if args.n_gpu > 1:
        raise ValueError('Use torchrun for multiple GPUs; do not use DataParallel.')
    _on_main_process(state,lambda: write_json(out/'trainer_args.json',json.loads(args.to_json_string())))

    safety = _safety_and_artifacts_callback(out,manifest,lambda: trainer.accelerator.scaler)
    callbacks = [safety]
    if early_stopping:
        early_callback = EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        callbacks.append(early_callback)
    trainer = _TokenLossTrainer(model=network,args=args,train_dataset=data['train'],eval_dataset=data['val'],
        processing_class=fast,data_collator=DataCollatorForSeq2Seq(fast,model=network,
            pad_to_multiple_of=8,label_pad_token_id=-100),compute_loss_func=checked_token_loss,
        compute_metrics=_TokenLossMetric(),preprocess_logits_for_metrics=_eval_statistics,
        callbacks=callbacks)
    trainer.model_accepts_loss_kwargs=False
    trainer._loss_shifts_labels=False
    initial_validation = None
    if warm_start is not None:
        baseline_path = out/'initial_validation.json'
        if checkpoint:
            initial_validation = json.loads(baseline_path.read_text())
        else:
            trainer._load_from_checkpoint(manifest['warm_start']['checkpoint'])
            initial_validation = trainer.evaluate()
            if not math.isfinite(initial_validation['eval_token_loss']):
                raise FloatingPointError('Nonfinite parent baseline validation loss.')
            _on_main_process(state,lambda: write_json(baseline_path,initial_validation))
            set_seed(config.seed)
        original_begin = early_callback.on_train_begin
        def seed_baseline(args,state,control,**kwargs):
            original_begin(args,state,control,**kwargs)
            if state.global_step == 0:
                state.best_metric = initial_validation['eval_token_loss']
                state.best_global_step = 0
                # An external parent path must never enter HF checkpoint rotation.
                state.best_model_checkpoint = None
        early_callback.on_train_begin = seed_baseline
    if state.device.type=='cuda':
        torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    planned_steps = steps if steps>0 else math.ceil(config.num_train_epochs*math.ceil(len(data['train'])/config.global_batch))
    if checkpoint and resume_step >= planned_steps:
        trainer._load_from_checkpoint(str(checkpoint))
        trainer.state = TrainerState.load_from_json(checkpoint/'trainer_state.json')
        if early_stopping and trainer.state.best_model_checkpoint is not None:
            trainer._load_best_model()
    else:
        trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    improved = None
    if warm_start is not None:
        improved = (trainer.state.best_model_checkpoint is not None
            and trainer.state.best_metric < initial_validation['eval_token_loss'])
        if not improved:
            trainer._load_from_checkpoint(manifest['warm_start']['checkpoint'])
    metrics=trainer.evaluate()
    if not math.isfinite(metrics['eval_token_loss']):
        raise FloatingPointError('Nonfinite final validation loss.')
    trainer.save_model(str(out/'final'))
    trainer.save_state()
    latest_checkpoint = out/f'checkpoint-{trainer.state.global_step}'
    if not latest_checkpoint.is_dir():
        latest_checkpoint = checkpoint
    result={'architecture':config.architecture,'global_step':trainer.state.global_step,
        'metrics':metrics,'wall_seconds':time.perf_counter()-begin,'precision':precision,
        'world_size':state.num_processes,'effective_batch':config.global_batch,
        'parameters':wrapper.count_parameters(),'finished_at':datetime.now(timezone.utc).isoformat(),
        'torch':str(torch.__version__),'transformers':transformers.__version__,
        'best_model_checkpoint':trainer.state.best_model_checkpoint,
        'best_metric':trainer.state.best_metric,'best_global_step':trainer.state.best_global_step,
        'latest_checkpoint':str(latest_checkpoint) if latest_checkpoint else None,
        'final_weights_source':'best_eval_token_loss' if early_stopping else 'last_update',
        'amp_diagnostics':safety.amp_diagnostics(),
        'gpu':torch.cuda.get_device_name() if state.device.type=='cuda' else 'CPU',
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if state.device.type=='cuda' else None}
    if warm_start is not None:
        result.update(initial_validation=initial_validation,improved_over_parent=improved,
            selected_checkpoint=trainer.state.best_model_checkpoint if improved else manifest['warm_start']['checkpoint'],
            final_weights_source='best_eval_token_loss' if improved else 'parent_baseline')
    def finalize():
        # Pinned Transformers saves the final full checkpoint before loading the
        # best weights, including when the final step is off the save interval.
        write_json(out/'log_history.json',trainer.state.log_history)
        write_json(out/'final/model_config.json',architecture.to_dict())
        tokenizer.save(str(out/'source_tokenizer.json'))
        from plotting import plot_training_history
        import matplotlib.pyplot as plt
        fig=plot_training_history(trainer.state.log_history,save_path=out/'training.png')
        plt.close(fig)
        write_json(out/'complete.json',result)
    _on_main_process(state,finalize)
    return result
