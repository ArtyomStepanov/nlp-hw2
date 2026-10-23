"""Exercise the portable runner on actual saved Arrow data and tiny HF models."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
import zipfile

import datasets as ds
import numpy as np
import pytest
import torch
from transformers import AutoModelForSeq2SeqLM

from tokenizer import train_bpe_tokenizer


def api():
    assert importlib.util.find_spec('kaggle_experiments'), 'Portable Kaggle runner is missing'
    import kaggle_experiments as k
    return k


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path/'bundle'
    selection = root/'selection'
    selection.mkdir(parents=True)
    raw = ds.Dataset.from_dict({'question':['one','two','three','four'],
        'code':['return 1','return 2','return 3','return 4']})
    tokenizer = train_bpe_tokenizer(raw,vocab_size=280,min_frequency=1)
    tokenizer.save(str(root/'tokenizer.json'))
    def encoded(count):
        rows=[]
        for i in range(count):
            source = tokenizer.encode(raw[i%4]['question']).ids
            labels = tokenizer.encode(raw[i%4]['code']).ids[1:]
            rows.append(dict(input_ids=source,attention_mask=[1]*len(source),
                labels=labels,length=max(len(source),len(labels))))
        return ds.Dataset.from_list(rows)
    ds.DatasetDict({'train':encoded(8),'val':encoded(4)}).save_to_disk(str(root/'data'))
    details={}
    for name,count in [('train',8),('val',4)]:
        p=selection/f'{name}_indices.npy'
        np.save(p,np.arange(count))
        details[name]={'selected_rows':count,'indices_sha256':digest(p)}
    manifest={'identity':{'train_rows':8,'validation_rows':4,'seed':42,
        'source':'/original/h100/cache/intentionally-absent',
        'tokenizer_sha256':digest(root/'tokenizer.json')},'splits':details}
    (selection/'data_manifest.json').write_text(json.dumps(manifest))
    files=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':digest(p)}
        for p in sorted(root.rglob('*')) if p.is_file()]
    (root/'bundle_manifest.json').write_text(json.dumps({'schema':1,'files':files,
        'selection_sha256':digest(selection/'data_manifest.json')}))
    return root


def test_portable_data_load_preserves_rows_and_provenance(bundle):
    k=api()
    data,tokenizer,identity=k.load_bundle(bundle,expected_rows=(8,4))
    assert len(data['train'])==8 and len(data['val'])==4
    assert identity['selection']['identity']['source']=='/original/h100/cache/intentionally-absent'
    assert tokenizer.decode(data['train'][0]['labels'],skip_special_tokens=True)=='return 1'
    assert data['train'][0]['labels'][-1]==tokenizer.token_to_id('<eos>')


@pytest.mark.parametrize('target',['tokenizer.json','selection/train_indices.npy','data/val/data-00000-of-00001.arrow'])
def test_portable_data_rejects_changed_bytes(bundle,target):
    k=api()
    with (bundle/target).open('ab') as f:
        f.write(b'corruption')
    with pytest.raises(ValueError,match='integrity|checksum|size'):
        k.load_bundle(bundle,expected_rows=(8,4))


def test_effective_batch_layout_accounts_for_two_gpus():
    k=api()
    assert k.accumulation_steps(256,16,1)==16
    assert k.accumulation_steps(256,16,2)==8
    assert k.accumulation_steps(256,32,2)==4
    with pytest.raises(ValueError,match='divis'):
        k.accumulation_steps(256,24,2)


@pytest.mark.parametrize('architecture',['t5','copy','bart'])
def test_kaggle_runner_trains_evaluates_exports_and_resumes(bundle,tmp_path,monkeypatch,architecture):
    k=api()
    monkeypatch.setenv('ACCELERATE_USE_CPU','true')
    config=k.KaggleConfig(architecture=architecture,output_dir=tmp_path/'run',
        micro_batch=1,global_batch=2,precision='no',use_cpu=True,
        tiny_model=True,expected_rows=(8,4),max_steps=2,eval_steps=3,
        gradient_checkpointing=True)
    result=k.run_experiment(bundle,config)
    assert result['global_step']==2
    assert result['metrics']['eval_target_tokens']>0
    assert np.isfinite(result['metrics']['eval_token_loss'])
    output=Path(config.output_dir)
    assert (output/'checkpoint-2/optimizer.pt').is_file()
    assert (output/'log_history.json').is_file()
    restored=AutoModelForSeq2SeqLM.from_pretrained(output/'final').eval()
    assert restored.config.model_type=={'t5':'t5','copy':'copy_bart','bart':'bart'}[architecture]
    first=ds.load_from_disk(str(bundle/'data'))['train'][0]
    args={n:torch.tensor([first[n]]) for n in ['input_ids','attention_mask']}
    generated=restored.generate(**args,max_new_tokens=3,num_beams=2)
    assert generated.shape[0]==1
    if architecture == 'bart':
        from tools.evaluate_fixed_bundle import run
        scored = run(bundle,output/'final',tmp_path/'evaluation',architecture='bart',
            device='cpu',batch_size=2,max_new_tokens=3,num_beams=2,limit=2,
            expected_manifest_sha=None,expected_rows=(8,4),length_penalty=0.8)
        assert scored['num_examples'] == 2
        assert scored['num_beams'] == 2
    # Resume a checkpoint from the exact same schedule; no-op at the final step.
    again=k.run_experiment(bundle,config,resume_from_checkpoint=output/'checkpoint-2')
    assert again['global_step']==2
    wrong=k.KaggleConfig(**(config.__dict__|{'learning_rate':.0003}))
    with pytest.raises(ValueError,match='resume|Resume'):
        k.run_experiment(bundle,wrong,resume_from_checkpoint=output/'checkpoint-2')


@pytest.mark.parametrize('architecture',['t5','copy'])
def test_two_process_runner_counts_validation_tokens_once(bundle,tmp_path,architecture):
    api()
    script=tmp_path/'distributed_smoke.py'
    script.write_text('''
import sys, torch
from kaggle_experiments import KaggleConfig,run_experiment
config=KaggleConfig(architecture=sys.argv[3],output_dir=sys.argv[2],micro_batch=1,
    global_batch=4,precision='no',use_cpu=True,tiny_model=True,expected_rows=(8,4),
    max_steps=2,eval_steps=3,gradient_checkpointing=True)
run_experiment(sys.argv[1],config)
torch.distributed.destroy_process_group()
''')
    project=Path(__file__).resolve().parents[1]
    env=os.environ|{'ACCELERATE_USE_CPU':'true','OMP_NUM_THREADS':'1',
        'MKL_NUM_THREADS':'1','MPLBACKEND':'Agg','TOKENIZERS_PARALLELISM':'false',
        'PYTHONPATH':str(project)}
    env['GLOO_SOCKET_IFNAME']='lo0' if sys.platform=='darwin' else 'lo'
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        port=listener.getsockname()[1]
    result=subprocess.run([sys.executable,'-m','torch.distributed.run','--master-addr=127.0.0.1',
        f'--master-port={port}','--nproc_per_node=2',str(script),str(bundle),str(tmp_path/'ddp'),architecture],
        cwd=project,env=env,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    actual=json.loads((tmp_path/'ddp/complete.json').read_text())
    assert actual['world_size']==2 and actual['global_step']==2 and actual['effective_batch']==4
    original=ds.load_from_disk(str(bundle/'data'))['val']
    assert actual['metrics']['eval_target_tokens']==sum(len(row['labels']) for row in original)


def test_nonfinite_loss_with_finite_derivatives_stops_before_complete(bundle,tmp_path,monkeypatch):
    k=api()
    import training
    original=training._token_normalized_loss
    calls=0
    def nonfinite(*args,**kwargs):
        nonlocal calls
        calls+=1
        loss=original(*args,**kwargs)
        return loss + torch.tensor(float('inf'),device=loss.device) if calls==1 else loss
    monkeypatch.setattr(training,'_token_normalized_loss',nonfinite)
    monkeypatch.setenv('ACCELERATE_USE_CPU','true')
    output=tmp_path/'nonfinite'
    config=k.KaggleConfig(architecture='copy',output_dir=output,micro_batch=1,
        global_batch=2,precision='no',use_cpu=True,tiny_model=True,expected_rows=(8,4),
        max_steps=2,eval_steps=3,gradient_checkpointing=False)
    with pytest.raises(FloatingPointError,match='loss'):
        k.run_experiment(bundle,config)
    assert not (output/'complete.json').exists()
    assert not list(output.glob('checkpoint-*'))


def test_disabled_compile_does_not_implicitly_compile_the_model(bundle,tmp_path,monkeypatch):
    k=api()
    original=torch.compile
    def forbidden_module_compile(value,*args,**kwargs):
        if isinstance(value,torch.nn.Module):
            raise AssertionError('Trainer requested torch.compile while disabled in experiment config')
        return original(value,*args,**kwargs)
    monkeypatch.setattr(torch,'compile',forbidden_module_compile)
    monkeypatch.setenv('ACCELERATE_USE_CPU','true')
    config=k.KaggleConfig(architecture='copy',output_dir=tmp_path/'uncompiled',
        micro_batch=1,global_batch=2,precision='no',use_cpu=True,tiny_model=True,
        expected_rows=(8,4),max_steps=1,eval_steps=3,gradient_checkpointing=False,
        torch_compile=False)
    result=k.run_experiment(bundle,config)
    assert result['global_step']==1


def test_t4_auto_precision_uses_fp16_instead_of_emulated_bf16(monkeypatch):
    k=api()
    assert hasattr(k,'resolve_precision'), 'Hardware-aware precision selection is missing'
    # Actual PyTorch defaults including_emulation=True: allocating a BF16 tensor
    # alone can report support even without T4 BF16 tensor cores.
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda including_emulation=True: including_emulation)
    assert k.resolve_precision('auto',torch.device('cuda'))=='fp16'
    with pytest.raises(ValueError,match='native BF16'):
        k.resolve_precision('bf16',torch.device('cuda'))
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda including_emulation=True: True)
    assert k.resolve_precision('auto',torch.device('cuda'))=='bf16'
    assert k.resolve_precision('auto',torch.device('cpu'))=='no'
