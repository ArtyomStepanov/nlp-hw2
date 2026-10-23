"""Real tiny CPU continuation: exact initialization, fresh optimizer and safe fallback."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoModelForSeq2SeqLM

import kaggle_experiments as k
from test_kaggle_experiments import bundle


@pytest.fixture
def parent(bundle, tmp_path, monkeypatch):
    monkeypatch.setenv('ACCELERATE_USE_CPU','true')
    path=tmp_path/'parent'
    config=k.KaggleConfig(architecture='bart',output_dir=path,micro_batch=1,global_batch=2,
        precision='no',use_cpu=True,tiny_model=True,expected_rows=(8,4),max_steps=2,
        eval_steps=1,early_stopping_patience=5,gradient_checkpointing=False,
        expected_manifest_sha=k.file_sha256(bundle/'bundle_manifest.json'))
    result=k.run_experiment(bundle,config)
    checkpoint=Path(result['best_model_checkpoint'])
    return bundle,path,checkpoint


def config(output):
    return k.KaggleConfig(architecture='bart',output_dir=output,micro_batch=1,global_batch=2,
        precision='no',use_cpu=True,tiny_model=True,expected_rows=(8,4),max_steps=10,
        eval_steps=2,early_stopping_patience=2,gradient_checkpointing=False,learning_rate=3e-5)


def warm(checkpoint):
    return dict(checkpoint=str(checkpoint),expected_weight_sha=k.file_sha256(checkpoint/'model.safetensors'),
                dropout=.2,attention_dropout=.1)


def ranking(monkeypatch, *, improves):
    import training
    original=training._TokenLossMetric
    class Metric(original):
        evaluations=0
        def __call__(self,prediction,compute_result=False):
            result=super().__call__(prediction,compute_result=compute_result)
            if compute_result:
                self.evaluations+=1
                result['token_loss']=(1.-.1*(self.evaluations-1)) if improves else (1. if self.evaluations==1 else 2.)
            return result
    monkeypatch.setattr(training,'_TokenLossMetric',Metric)


def test_exact_parent_initialization_effective_dropout_fresh_updates_and_baseline_fallback(parent,tmp_path,monkeypatch):
    bundle,parent_root,checkpoint=parent
    before={str(p):k.file_sha256(p) for p in parent_root.rglob('*') if p.is_file()}
    saved=AutoModelForSeq2SeqLM.from_pretrained(checkpoint).eval()
    original_factory=k._safety_and_artifacts_callback
    observed=[]
    def factory(*args,**kwargs):
        callback=original_factory(*args,**kwargs)
        def begin(args,state,control,model=None,optimizer=None,**kwargs):
            assert not optimizer.state
            assert all(torch.equal(value,saved.state_dict()[name]) for name,value in model.state_dict().items())
            assert model.model.encoder.dropout==model.model.encoder.layers[0].dropout==.2
            assert model.model.encoder.layers[0].self_attn.dropout==.1
            assert model.model.decoder.layers[0].encoder_attn.dropout==.1
            ids=torch.tensor([[2,10,11,3]])
            batch=dict(input_ids=ids,attention_mask=torch.ones_like(ids),decoder_input_ids=ids)
            with torch.no_grad():
                model.eval()
                assert torch.equal(model(**batch).logits,saved(**batch).logits)
                model.train()
                assert not torch.equal(model(**batch).logits,model(**batch).logits)
            observed.append(True)
        callback.on_train_begin=begin
        return callback
    monkeypatch.setattr(k,'_safety_and_artifacts_callback',factory)
    ranking(monkeypatch,improves=False)
    output=tmp_path/'continued'
    result=k.run_experiment(bundle,config(output),warm_start=warm(checkpoint))
    assert observed==[True]
    assert result['global_step']==4  # Baseline counts toward patience from step zero.
    assert result['best_global_step']==0 and result['best_model_checkpoint'] is None
    assert result['final_weights_source']=='parent_baseline'
    assert result['improved_over_parent'] is False
    assert result['initial_validation']['eval_token_loss']==1.
    parent_weights=load_file(str(checkpoint/'model.safetensors'))
    final=load_file(str(output/'final/model.safetensors'))
    latest=load_file(str(output/'checkpoint-4/model.safetensors'))
    assert all(torch.equal(final[name],value) for name,value in parent_weights.items())
    assert any(not torch.equal(latest[name],value) for name,value in parent_weights.items())
    optimizer=torch.load(output/'checkpoint-4/optimizer.pt',weights_only=True)
    assert {int(value['step']) for value in optimizer['state'].values()}=={4}
    for name in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','run_manifest.json'):
        assert (output/'checkpoint-4'/name).is_file()
    assert before=={str(p):k.file_sha256(p) for p in parent_root.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        k.run_experiment(bundle,config(output),warm_start=warm(checkpoint))


def test_improved_continuation_exports_its_best_and_new_science_is_explicit(parent,tmp_path,monkeypatch):
    bundle,_,checkpoint=parent
    ranking(monkeypatch,improves=True)
    output=tmp_path/'improved'
    setup=config(output);setup.max_steps=3
    result=k.run_experiment(bundle,setup,warm_start=warm(checkpoint))
    assert result['improved_over_parent'] is True
    assert result['final_weights_source']=='best_eval_token_loss'
    assert result['best_global_step']==3
    manifest=json.loads((output/'run_manifest.json').read_text())
    assert manifest['warm_start']['optimizer_initialization']=='fresh_adamw_new_schedule'
    assert manifest['model_config']['dropout']==.2 and manifest['model_config']['attention_dropout']==.1
    assert manifest['config']['learning_rate']==3e-5
    assert manifest['warm_start']['parent_best_step']>0


def test_wrong_parent_weight_identity_is_rejected_before_training(parent,tmp_path):
    bundle,_,checkpoint=parent
    spec=warm(checkpoint);spec['expected_weight_sha']='0'*64
    with pytest.raises(ValueError,match='weight|SHA'):
        k.run_experiment(bundle,config(tmp_path/'wrong'),warm_start=spec)


@pytest.mark.parametrize('changed',['tokenizer','bundle'])
def test_parent_tokenizer_or_training_bundle_drift_is_rejected(parent,tmp_path,changed):
    bundle,parent_root,checkpoint=parent
    if changed=='tokenizer':
        path=checkpoint/'tokenizer.json'
        path.write_text(path.read_text()+' ')
    else:
        path=parent_root/'run_manifest.json'
        value=json.loads(path.read_text());value['bundle_sha256']='0'*64
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='tokenizer|bundle'):
        k.run_experiment(bundle,config(tmp_path/'bad'),warm_start=warm(checkpoint))
