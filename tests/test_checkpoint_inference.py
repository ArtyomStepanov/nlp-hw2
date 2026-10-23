import json

import datasets as ds
import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer

from inference import load_inference_model
from models import BaselineTransformer, TransformerConfig
from tokenizer import train_bpe_tokenizer


@pytest.fixture
def checkpoint_model():
    tokenizer = train_bpe_tokenizer(
        ds.Dataset.from_dict({"question": ["return one", "return two"]}),
        vocab_size=270, min_frequency=1, columns="question",
    )
    model = BaselineTransformer(TransformerConfig.from_tokenizer(
        tokenizer, d_model=16, encoder_layers=1, decoder_layers=1,
        attention_heads=2, ffn_dim=32, max_position_embeddings=32, dropout=0,
    )).eval()
    return model, tokenizer


def save_trainer_checkpoint(path, model, tokenizer):
    model.network.save_pretrained(path)
    tokenizer.save(str(path / "tokenizer.json"))
    return path


@pytest.mark.parametrize("checkpoint_format", ["own-directory", "own-file", "trainer", "trainer-eos-list"])
def test_checkpoint_roundtrip_preserves_logits_generation_and_tied_weights(checkpoint_model, tmp_path, checkpoint_format):
    original, tokenizer = checkpoint_model
    if checkpoint_format.startswith("trainer"):
        # A generation option that changes the generated sequence must survive.
        original.network.generation_config.forced_bos_token_id = 7
        if checkpoint_format == "trainer-eos-list":
            original.network.generation_config.eos_token_id = [original.config.eos_token_id]
        path = save_trainer_checkpoint(tmp_path / "checkpoint-12", original, tokenizer)
    else:
        original.save_checkpoint(tmp_path / "model.pt")
        tokenizer.save(str(tmp_path / "tokenizer.json"))
        path = tmp_path / "model.pt" if checkpoint_format == "own-file" else tmp_path
    restored, restored_tokenizer = load_inference_model(path, device="cpu")
    assert isinstance(restored_tokenizer, Tokenizer)
    assert restored_tokenizer.encode("return one").ids == tokenizer.encode("return one").ids
    assert restored.config == original.config
    assert not restored.training and not restored.network.training
    assert next(restored.parameters()).device.type == "cpu"
    inputs = torch.tensor([tokenizer.encode("return one").ids])
    decoder_inputs = torch.tensor([[2, 7, 8]])
    with torch.inference_mode():
        expected = original(inputs, decoder_input_ids=decoder_inputs).logits
        actual = restored(inputs, decoder_input_ids=decoder_inputs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(restored.generate(inputs, max_new_tokens=4), original.generate(inputs, max_new_tokens=4))
    network = restored.network
    assert network.model.shared.weight is network.model.encoder.embed_tokens.weight
    assert network.model.shared.weight is network.model.decoder.embed_tokens.weight
    assert network.model.shared.weight is network.lm_head.weight


@pytest.mark.parametrize("field,value", [
    ("model_type", "t5"),
    ("architectures", ["BartForSequenceClassification"]),
    ("decoder_attention_heads", 4),
    ("decoder_ffn_dim", 64),
    ("tie_word_embeddings", False),
    ("activation_function", "relu"),
    ("scale_embedding", True),
    ("decoder_start_token_id", 3),
])
def test_trainer_checkpoint_rejects_incompatible_architecture(checkpoint_model, tmp_path, field, value):
    model, tokenizer = checkpoint_model
    path = save_trainer_checkpoint(tmp_path / "checkpoint-12", model, tokenizer)
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    config[field] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=field):
        load_inference_model(path, device="cpu")


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape", "tied"])
def test_trainer_checkpoint_rejects_incomplete_or_inconsistent_weights(checkpoint_model, tmp_path, corruption):
    model, tokenizer = checkpoint_model
    path = save_trainer_checkpoint(tmp_path / "checkpoint-12", model, tokenizer)
    weights_path = path / "model.safetensors"
    weights = load_file(str(weights_path))
    if corruption == "missing":
        del weights["model.encoder.layers.0.fc1.weight"]
    elif corruption == "unexpected":
        weights["unrelated.weight"] = torch.ones(2)
    elif corruption == "shape":
        weights["final_logits_bias"] = torch.zeros(1, 3)
    else:
        weights["lm_head.weight"] = weights["model.shared.weight"] + 1
    save_file(weights, str(weights_path))
    with pytest.raises((ValueError, RuntimeError)):
        load_inference_model(path, device="cpu")


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors", "tokenizer.json"])
def test_trainer_checkpoint_reports_missing_required_file(checkpoint_model, tmp_path, filename):
    model, tokenizer = checkpoint_model
    path = save_trainer_checkpoint(tmp_path / "checkpoint-12", model, tokenizer)
    (path / filename).unlink()
    with pytest.raises(FileNotFoundError, match=filename):
        load_inference_model(path, device="cpu")


@pytest.mark.parametrize("field,value", [
    ("decoder_start_token_id", 3), ("eos_token_id", [4]), ("eos_token_id", [3, 4]),
])
def test_trainer_checkpoint_checks_generation_special_tokens(checkpoint_model, tmp_path, field, value):
    model, tokenizer = checkpoint_model
    path = save_trainer_checkpoint(tmp_path / "checkpoint-12", model, tokenizer)
    config_path = path / "generation_config.json"
    config = json.loads(config_path.read_text())
    config[field] = value
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=field):
        load_inference_model(path, device="cpu")


def test_load_checkpoint_saved_during_actual_trainer_training(checkpoint_model, tmp_path):
    from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments

    model, tokenizer = checkpoint_model
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, bos_token="<bos>", eos_token="<eos>",
        pad_token="<pad>", unk_token="<unk>", model_input_names=["input_ids", "attention_mask"],
    )
    ids = tokenizer.encode("return one").ids
    data = ds.Dataset.from_dict({"input_ids": [ids], "labels": [ids[1:]]})
    trainer = Seq2SeqTrainer(
        model=model.network,
        args=Seq2SeqTrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, max_steps=1,
            per_device_train_batch_size=1, save_strategy="steps", save_steps=1,
            logging_strategy="no", report_to="none", disable_tqdm=True,
        ),
        train_dataset=data, processing_class=fast_tokenizer,
        data_collator=DataCollatorForSeq2Seq(fast_tokenizer, model=model.network),
    )
    trainer.train()
    model.eval()
    restored, restored_tokenizer = load_inference_model(tmp_path / "checkpoint-1", device="cpu")
    assert restored_tokenizer.encode("return one").ids == ids
    inputs = torch.tensor([ids])
    with torch.inference_mode():
        torch.testing.assert_close(
            restored(inputs, decoder_input_ids=inputs[:, :-1]).logits,
            model(inputs, decoder_input_ids=inputs[:, :-1]).logits,
            rtol=0, atol=0,
        )
    assert torch.equal(restored.generate(inputs, max_new_tokens=4), model.generate(inputs, max_new_tokens=4))
