"""T5 must preserve our target contract and agree with HF eager attention."""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F
from transformers import T5ForConditionalGeneration

import models
from models import TransformerConfig


def tiny_model():
    assert hasattr(models, 'T5Transformer'), 'T5 model is not implemented'
    return models.T5Transformer(TransformerConfig(
        vocab_size=32, d_model=32, encoder_layers=2, decoder_layers=2,
        attention_heads=4, ffn_dim=64, max_position_embeddings=32, dropout=0,
    ))


def batch():
    return dict(input_ids=torch.tensor([[2, 7, 8, 3, 0], [2, 9, 10, 11, 3]]),
                attention_mask=torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
                labels=torch.tensor([[12, 3, -100], [13, 14, 3]]))


def eager_copy(model):
    reference = T5ForConditionalGeneration(deepcopy(model.network.config))
    reference.load_state_dict(model.network.state_dict())
    return reference.eval()


def test_t5_target_shift_loss_padding_and_training_gradients():
    model = tiny_model()
    data = batch()
    labels = model.prepare_labels(torch.tensor([[2, 12, 3, 0], [2, 13, 14, 3]]))
    torch.testing.assert_close(labels, data['labels'])
    shifted = model.network.prepare_decoder_input_ids_from_labels(labels)
    torch.testing.assert_close(shifted, torch.tensor([[2, 12, 3], [2, 13, 14]]))
    result = model(**data)
    expected = F.cross_entropy(result.logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
    torch.testing.assert_close(result.loss, expected)
    result.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.network.config.d_kv == 8
    assert model.network.config.feed_forward_proj == 'gated-gelu'
    assert model.network.shared.weight.data_ptr() == model.network.lm_head.weight.data_ptr()
    assert not any('embed_positions' in name for name, _ in model.named_parameters())


def test_t5_sdpa_matches_eager_logits_and_all_parameter_gradients():
    model = tiny_model().eval()
    reference = eager_copy(model)
    actual, expected = model(**batch()), reference(**batch())
    torch.testing.assert_close(actual.logits, expected.logits, rtol=2e-5, atol=2e-5)
    actual.loss.backward()
    expected.loss.backward()
    for (name, param), (ref_name, ref_param) in zip(model.network.named_parameters(), reference.named_parameters()):
        assert name == ref_name
        torch.testing.assert_close(param.grad, ref_param.grad, rtol=3e-4, atol=3e-6, msg=name)


@pytest.mark.parametrize('beams', [1, 4])
def test_t5_cached_generation_matches_full_decoding_and_eager(beams):
    model = tiny_model().eval()
    reference = eager_copy(model)
    data = {k: v for k, v in batch().items() if k != 'labels'}
    options = dict(max_new_tokens=5, min_new_tokens=5, num_beams=beams, do_sample=False)
    expected = reference.generate(**data, **options, use_cache=True)
    cached = model.network.generate(**data, **options, use_cache=True)
    uncached = model.network.generate(**data, **options, use_cache=False)
    torch.testing.assert_close(cached, expected)
    torch.testing.assert_close(cached, uncached)
    model.train()
    model.generate(**data, max_new_tokens=5, num_beams=beams)
    assert model.training


def test_t5_masks_prevent_padding_and_future_token_leakage():
    model = tiny_model().eval()
    data = batch()
    data['decoder_input_ids'] = model.network.prepare_decoder_input_ids_from_labels(data.pop('labels'))
    before = model(**data).logits
    data['input_ids'][0, -1] = 17  # same masked position, different token
    data['decoder_input_ids'][:, -1] = 18  # future token must not change earlier logits
    after = model(**data).logits
    torch.testing.assert_close(before[:, :-1], after[:, :-1], rtol=1e-5, atol=1e-5)


def test_t5_wrapper_and_standard_hf_checkpoint_round_trip(tmp_path):
    model = tiny_model().eval()
    expected = model(**batch()).logits
    model.save_checkpoint(tmp_path / 'model.pt')
    restored = models.T5Transformer.load_checkpoint(tmp_path / 'model.pt').eval()
    torch.testing.assert_close(expected, restored(**batch()).logits)
    model.network.save_pretrained(tmp_path / 'hf')
    hf = T5ForConditionalGeneration.from_pretrained(tmp_path / 'hf').eval()
    torch.testing.assert_close(expected, hf(**batch()).logits, rtol=2e-5, atol=2e-5)


def test_t5_attention_outputs_use_eager_fallback():
    model = tiny_model().eval()
    reference = eager_copy(model)
    actual = model.network(**batch(), output_attentions=True)
    expected = reference(**batch(), output_attentions=True)
    for name in ('encoder_attentions', 'decoder_attentions', 'cross_attentions'):
        for a, b in zip(getattr(actual, name), getattr(expected, name)):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize('field,value', [('num_heads', 2), ('num_layers', 3), ('d_ff', 128),
    ('feed_forward_proj', 'relu'), ('model_type', 'bart')])
def test_t5_resume_rejects_different_architecture(tmp_path, field, value):
    import json
    from tokenizers import Tokenizer, models as tm
    from training import _validate_resume
    model = tiny_model()
    tokenizer = Tokenizer(tm.WordLevel({'<unk>': 0}))
    for name in ('model.safetensors', 'optimizer.pt', 'scheduler.pt'):
        (tmp_path / name).touch()
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    (tmp_path / 'trainer_state.json').write_text(json.dumps({'global_step': 1}))
    saved = model.network.config.to_dict()
    (tmp_path / 'config.json').write_text(json.dumps(saved))
    assert _validate_resume(tmp_path, model.network, tokenizer) == 1
    saved[field] = value
    (tmp_path / 'config.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='Конфигурация checkpoint'):
        _validate_resume(tmp_path, model.network, tokenizer)
