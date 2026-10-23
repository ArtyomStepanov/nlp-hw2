"""Pointer-generator probabilities, learning, checkpoint and generation contracts."""
import pytest
import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForSeq2SeqLM

import models
from models import TransformerConfig


def tiny_model():
    assert hasattr(models, 'CopyTransformer'), 'Copy Transformer is not implemented'
    return models.CopyTransformer(TransformerConfig(vocab_size=24, d_model=24,
        encoder_layers=1, decoder_layers=1, attention_heads=3, ffn_dim=48,
        max_position_embeddings=32, dropout=0))


def batch():
    return dict(input_ids=torch.tensor([[2,7,7,3,0], [2,8,9,10,3]]),
        attention_mask=torch.tensor([[1,1,1,1,0],[1,1,1,1,1]]),
        labels=torch.tensor([[7,3,-100],[8,9,3]]))


def test_copy_mixture_sums_duplicate_ids_and_excludes_special_tokens():
    assert hasattr(models, 'CopyTransformer'), 'Copy Transformer is not implemented'
    from models.copy import PointerGenerator
    head = PointerGenerator(vocab_size=6, hidden_size=4, copy_dim=2, excluded_ids=(0,1,2,3))
    with torch.no_grad():
        for param in head.parameters():
            param.zero_()
    ids = torch.tensor([[4,4,5,0,2,3,1]])
    logits = head(torch.zeros(1,1,6), torch.zeros(1,1,4),
        torch.zeros(1,7,4), ids, torch.ones_like(ids))
    # p_gen=1/2, uniform vocabulary=1/6; copy(4)=2/3, copy(5)=1/3.
    expected = torch.tensor([[[1/12,1/12,1/12,1/12,5/12,1/4]]])
    torch.testing.assert_close(logits.exp(), expected)
    torch.testing.assert_close(logits.exp().sum(-1), torch.ones(1,1))


def test_copy_all_masked_source_falls_back_to_vocab_with_finite_gradients():
    assert hasattr(models, 'CopyTransformer'), 'Copy Transformer is not implemented'
    from models.copy import PointerGenerator
    head = PointerGenerator(vocab_size=6, hidden_size=4, copy_dim=2, excluded_ids=(0,1,2,3))
    vocab = torch.tensor([[[0.,1.,2.,3.,4.,5.]]], requires_grad=True)
    hidden = torch.randn(1,1,4,requires_grad=True)
    encoder = torch.randn(1,3,4,requires_grad=True)
    actual = head(vocab, hidden, encoder, torch.tensor([[0,2,3]]), torch.zeros(1,3))
    torch.testing.assert_close(actual, F.log_softmax(vocab,-1))
    (-actual[...,4].mean()).backward()
    assert torch.isfinite(vocab.grad).all()
    assert torch.isfinite(hidden.grad).all() and torch.isfinite(encoder.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_copy_loss_target_shift_and_gradients():
    model = tiny_model()
    data = batch()
    actual = model(**data)
    expected = F.nll_loss(actual.logits.flatten(0,1), data['labels'].flatten(), ignore_index=-100)
    torch.testing.assert_close(actual.loss, expected)
    torch.testing.assert_close(actual.logits.exp().sum(-1), torch.ones(2,3))
    shifted = model.network.prepare_decoder_input_ids_from_labels(data['labels'])
    torch.testing.assert_close(shifted, torch.tensor([[2,7,3],[2,8,9]]))
    actual.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad.abs().sum()>0 for p in model.network.copy_head.parameters())


@pytest.mark.parametrize('beams',[1,4])
def test_copy_cached_beam_generation_matches_full_prefix_and_individual_rows(beams):
    model = tiny_model().eval()
    inputs = {k:v for k,v in batch().items() if k!='labels'}
    kwargs = dict(max_new_tokens=4,min_new_tokens=4,num_beams=beams,do_sample=False)
    actual = model.network.generate(**inputs,**kwargs,use_cache=True)
    expected = model.network.generate(**inputs,**kwargs,use_cache=False)
    torch.testing.assert_close(actual,expected)
    for i in range(2):
        row = {k:v[i:i+1] for k,v in inputs.items()}
        separate = model.network.generate(**row,**kwargs,use_cache=True)
        torch.testing.assert_close(actual[i:i+1],separate)


def test_copy_masks_prevent_padding_and_future_token_leaks():
    model = tiny_model().eval()
    data = batch()
    data['decoder_input_ids'] = model.network.prepare_decoder_input_ids_from_labels(data.pop('labels'))
    before = model(**data).logits
    data['input_ids'][0,-1] = 19
    data['decoder_input_ids'][:,-1] = 20
    after = model(**data).logits
    torch.testing.assert_close(before[:,:-1],after[:,:-1],atol=1e-6,rtol=1e-5)


def test_copy_wrapper_and_hf_checkpoint_keep_copy_head(tmp_path):
    model = tiny_model().eval()
    expected = model(**batch()).logits
    model.save_checkpoint(tmp_path/'model.pt')
    restored = models.CopyTransformer.load_checkpoint(tmp_path/'model.pt').eval()
    torch.testing.assert_close(expected,restored(**batch()).logits)
    model.network.save_pretrained(tmp_path/'hf')
    config = AutoConfig.from_pretrained(tmp_path/'hf')
    assert config.model_type == 'copy_bart'
    hf = AutoModelForSeq2SeqLM.from_pretrained(tmp_path/'hf').eval()
    assert type(hf) is type(model.network)
    torch.testing.assert_close(expected,hf(**batch()).logits)


def test_copy_can_learn_to_copy_new_source_tokens():
    torch.manual_seed(123)
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.02)
    data = dict(input_ids=torch.tensor([[2,7,3],[2,8,3],[2,9,3],[2,10,3]]),
        attention_mask=torch.ones(4,3,dtype=torch.long),
        labels=torch.tensor([[7,3],[8,3],[9,3],[10,3]]))
    for _ in range(35):
        optimizer.zero_grad()
        loss = model(**data).loss
        loss.backward()
        optimizer.step()
    assert loss.item() < .2
    generated = model.generate(torch.tensor([[2,11,3]]),max_new_tokens=3)
    assert generated[0,1].item() == 11
    known = model.generate(data['input_ids'],max_new_tokens=3)
    torch.testing.assert_close(known[:,:3],torch.tensor([[2,7,3],[2,8,3],[2,9,3],[2,10,3]]))


def test_copy_amp_keeps_large_attention_scores_and_mixture_gradients_finite():
    from models.copy import PointerGenerator
    head = PointerGenerator(24,24,8,excluded_ids=(0,1,2,3))
    hidden = (torch.randn(2,3,24)*1000).requires_grad_()
    encoder = (torch.randn(2,4,24)*1000).requires_grad_()
    with torch.autocast('cpu',dtype=torch.float16):
        logits = head(torch.randn(2,3,24)*100,hidden,encoder,
            torch.tensor([[4,5,6,0],[7,8,9,10]]),torch.tensor([[1,1,1,0],[1,1,1,1]]))
        loss = F.nll_loss(logits.flatten(0,1),torch.tensor([4,3,5,7,3,8]))
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(hidden.grad).all() and torch.isfinite(encoder.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_copy_resume_rejects_plain_bart_and_different_copy_recipe(tmp_path):
    import json
    from tokenizers import Tokenizer, models as tm
    from training import _validate_resume
    network = tiny_model().network
    tokenizer = Tokenizer(tm.WordLevel({'<unk>':0}))
    for name in ('model.safetensors','optimizer.pt','scheduler.pt'):
        (tmp_path/name).touch()
    tokenizer.save(str(tmp_path/'tokenizer.json'))
    (tmp_path/'trainer_state.json').write_text(json.dumps({'global_step':1}))
    saved = network.config.to_dict()
    for field,value in [('model_type','bart'),('copy_attention_dim',8),('copy_excluded_ids',[0])]:
        (tmp_path/'config.json').write_text(json.dumps(saved|{field:value}))
        with pytest.raises(ValueError,match='Конфигурация checkpoint'):
            _validate_resume(tmp_path,network,tokenizer)
