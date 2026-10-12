import json

import datasets as ds
import pytest
import torch

from models import BaselineTransformer, TransformerConfig
from tokenizer import train_bpe_tokenizer


@pytest.fixture
def small_model():
    data = ds.Dataset.from_dict({
        "question": ["short", "a much longer question", "middle"],
        "code": ["def solve():\n    return 1\n"] * 3,
    })
    tokenizer = train_bpe_tokenizer(data, vocab_size=300, min_frequency=1)
    model = BaselineTransformer(TransformerConfig.from_tokenizer(
        tokenizer, d_model=16, encoder_layers=1, decoder_layers=1,
        attention_heads=2, ffn_dim=32, max_position_embeddings=32, dropout=0,
    ))
    return data, tokenizer, model


def test_inference_preserves_order_and_saves_one_prediction_per_row(small_model, tmp_path):
    from inference import InferenceConfig, generate_code

    data, tokenizer, model = small_model
    sequences = {tuple(tokenizer.encode(text).ids): tokenizer.encode(f"def solve():\n    return {i}\n").ids for i, text in enumerate(data["question"])}

    # Script only the expensive generation boundary. Exercise real batching,
    # tokenization, decoding, sorting, row restoration and persistence.
    def scripted_generate(input_ids, attention_mask=None, **kwargs):
        rows = [sequences[tuple(row[mask.bool()].tolist())] for row, mask in zip(input_ids, attention_mask)]
        width = max(map(len, rows))
        return torch.tensor([ids + [model.config.pad_token_id] * (width - len(ids)) for ids in rows])

    model.generate = scripted_generate
    model.train()
    result = generate_code(data, model, InferenceConfig(tokenizer, device="cpu", batch_size=2, max_new_tokens=24, output_dir=tmp_path / "run"))
    assert result.predictions["question"] == data["question"]
    assert result.codes == ["def solve():\n    return 0\n", "def solve():\n    return 1\n", "def solve():\n    return 2\n"]
    assert result.metrics["num_examples"] == 3
    assert result.predictions["ended_with_eos"] == [True, True, True]
    assert model.training
    saved = [json.loads(line) for line in (tmp_path / "run" / "predictions.jsonl").read_text().splitlines()]
    assert [row["generated_code"] for row in saved] == result.codes
    assert json.loads((tmp_path / "run" / "metrics.json").read_text())["total_seconds"] > 0
    with pytest.raises(ValueError, match="output_dir"):
        generate_code(data, model, InferenceConfig(tokenizer, device="cpu", max_new_tokens=4, output_dir=tmp_path / "run"))


def test_real_model_inference_and_checkpoint_reload(small_model, tmp_path):
    from inference import InferenceConfig, generate_code, load_inference_model

    data, tokenizer, model = small_model
    model.save_checkpoint(tmp_path / "model.pt")
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    restored, restored_tokenizer = load_inference_model(tmp_path, device="cpu")
    result = generate_code(data.select([0, 1]), restored, InferenceConfig(restored_tokenizer, device="cpu", max_new_tokens=4, batch_size=2))
    assert len(result.codes) == 2
    assert all(isinstance(code, str) for code in result.codes)
    assert all(1 <= n <= 4 for n in result.predictions["generated_tokens"])
    assert result.metrics["generation_seconds"] > 0


def test_long_sources_are_reported_and_never_dropped(small_model):
    from inference import InferenceConfig, generate_code

    _, tokenizer, model = small_model
    data = ds.Dataset.from_dict({"question": ["short " * 100, "middle"]})
    result = generate_code(data, model, InferenceConfig(tokenizer, device="cpu", max_source_length=8, max_new_tokens=2))
    assert len(result.codes) == 2
    assert result.predictions["source_truncated"] == [True, False]
    assert result.metrics["source_truncated_rate"] == 0.5
    with pytest.raises(ValueError, match="обрез|длин"):
        generate_code(data, model, InferenceConfig(tokenizer, device="cpu", max_source_length=8, truncate_source=False, max_new_tokens=2))


def test_invalid_input_and_tokenizer_mismatch_fail_before_generation(small_model):
    from inference import InferenceConfig, generate_code

    _, tokenizer, model = small_model
    for inputs in ([None], []):
        with pytest.raises(ValueError):
            generate_code(ds.Dataset.from_dict({"question": inputs}), model, InferenceConfig(tokenizer, device="cpu"))
    other = train_bpe_tokenizer(ds.Dataset.from_dict({"question": ["different vocabulary"]}), 260, columns="question")
    with pytest.raises(ValueError, match="словар"):
        generate_code(ds.Dataset.from_dict({"question": ["short"]}), model, InferenceConfig(other, device="cpu", max_new_tokens=4))


def test_padding_generated_before_eos_still_counts_as_generated_tokens(small_model):
    from inference import InferenceConfig, generate_code

    data, tokenizer, model = small_model
    model.generate = lambda input_ids, **kwargs: torch.tensor([[model.config.bos_token_id] + [model.config.pad_token_id] * 4] * len(input_ids))
    result = generate_code(data, model, InferenceConfig(tokenizer, device="cpu", max_new_tokens=4))
    assert result.predictions["generated_tokens"] == [4, 4, 4]
    assert result.predictions["hit_token_limit"] == [True, True, True]
    assert result.codes == ["", "", ""]


def test_model_mode_is_restored_on_generation_failure(small_model):
    from inference import InferenceConfig, generate_code

    data, tokenizer, model = small_model
    def broken_generate(*args, **kwargs):
        raise RuntimeError("generation failed")
    model.generate = broken_generate
    model.train()
    with pytest.raises(RuntimeError, match="generation failed"):
        generate_code(data, model, InferenceConfig(tokenizer, device="cpu", max_new_tokens=4))
    assert model.training
