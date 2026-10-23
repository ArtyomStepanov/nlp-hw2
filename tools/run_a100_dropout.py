"""One isolated BART dropout continuation, starting fresh AdamW from best180000."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from kaggle_experiments import KaggleConfig, config_manifest, file_sha256, run_experiment, write_json
from tools.run_a100_full import GPU_LOCK_DIRECTORY, _lock, verify_full_bundle

PARENT_WEIGHT_SHA = '2059268173416c77cd8c901c791eb86e7cbcea0ede756b2bb830ec23638b8165'
BUNDLE_SHA = '0f330f228599f4cff65ed6ce0ca11e2d667ea24e6b911b5f67f6fff7e24bcbe5'


def run_dropout(bundle_root,parent_checkpoint,output_dir,*,preflight=False):
    bundle,parent,output = map(lambda p:Path(p).resolve(),(bundle_root,parent_checkpoint,output_dir))
    for source in (bundle,parent.parent):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Fresh output must be separate from parent and bundle.')
    if output.exists():
        raise FileExistsError(f'Fresh continuation output required: {output}')
    output.mkdir(parents=True,exist_ok=False)
    with _lock(GPU_LOCK_DIRECTORY/'queue.lock'),_lock(output/'queue.lock'):
        state=dict(status='running',phase='verification',preflight=preflight,
                   started_at=datetime.now(timezone.utc).isoformat())
        try:
            write_json(output/'queue_state.json',state)
            identity=verify_full_bundle(bundle)
            if identity['bundle_sha256'] != BUNDLE_SHA:
                raise ValueError('Full bundle differs from the approved continuation source.')
            if parent.name != 'checkpoint-180000' or file_sha256(parent/'model.safetensors') != PARENT_WEIGHT_SHA:
                raise ValueError('Only verified full best180000 may initialize this continuation.')
            config=KaggleConfig(architecture='bart',output_dir=str(output/'trial'),micro_batch=128,
                global_batch=256,learning_rate=3e-5,num_train_epochs=2.,eval_steps=5000,
                precision='fp16',gradient_checkpointing=False,torch_compile=False,
                expected_rows=(11593060,118302),expected_manifest_sha=BUNDLE_SHA,
                early_stopping_patience=5,seed=42,preflight=preflight)
            warm=dict(checkpoint=str(parent),expected_weight_sha=PARENT_WEIGHT_SHA,dropout=.2,attention_dropout=.1)
            names=['kaggle_experiments.py','training.py','plotting.py','tools/run_a100_dropout.py',
                   'tools/run_a100_full.py','tools/prepare_a100_full.py']
            names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'models').glob('*.py'))]
            sources={name:file_sha256(ROOT/name) for name in names}
            request=dict(schema=1,status='authorized',initialization='parent_weights_fresh_adamw',
                parent_checkpoint=str(parent),parent_weight_sha256=PARENT_WEIGHT_SHA,
                bundle=identity,config=config_manifest(config),warm_start=warm,
                planned_step_attempts=2 if preflight else 2*math.ceil(11593060/256),sources=sources,
                baseline_selection='Initial validation seeds patience; immutable parent fallback if no improvement.',
                official_test_used=False)
            write_json(output/'run_request.json',request)
            state.update(phase='training');write_json(output/'queue_state.json',state)
            result=run_experiment(bundle,config,warm_start=warm)
            if {name:file_sha256(ROOT/name) for name in names} != sources:
                raise ValueError('Continuation source snapshot changed during training.')
            if result['global_step']<1 or not math.isfinite(result['metrics']['eval_token_loss']):
                raise ValueError('Continuation did not complete finite training/evaluation.')
            completion=dict(schema=1,status='complete',preflight=preflight,
                request_sha256=file_sha256(output/'run_request.json'),result=result,
                finished_at=datetime.now(timezone.utc).isoformat())
            write_json(output/'dropout_complete.json',completion)
            state.update(status='complete',phase='complete');write_json(output/'queue_state.json',state)
            return completion
        except BaseException as error:
            state.update(status='failed',error=f'{type(error).__name__}: {error}')
            write_json(output/'queue_state.json',state)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle-root','parent-checkpoint','output-dir'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--preflight',action='store_true')
    print(json.dumps(run_dropout(**vars(parser.parse_args())),indent=2))


if __name__=='__main__':
    main()
