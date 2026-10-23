"""Конфигурация encoder–decoder Transformer."""

from dataclasses import asdict, dataclass

from tokenizers import Tokenizer


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    d_model: int = 512
    encoder_layers: int = 6
    decoder_layers: int = 6
    attention_heads: int = 8
    ffn_dim: int = 2048
    max_position_embeddings: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3
    unk_token_id: int | None = 1

    def __post_init__(self) -> None:
        for name in (
            "vocab_size", "d_model", "encoder_layers", "decoder_layers",
            "attention_heads", "ffn_dim", "max_position_embeddings",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} должен быть положительным целым числом.")
        if self.d_model % self.attention_heads:
            raise ValueError("d_model должен делиться на attention_heads.")
        for name in ("dropout", "attention_dropout"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} должен находиться в диапазоне [0, 1).")
        token_ids = (self.pad_token_id, self.bos_token_id, self.eos_token_id)
        if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < self.vocab_size for value in token_ids):
            raise ValueError("ID спецтокенов должны находиться внутри словаря.")
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("PAD, BOS и EOS должны иметь разные ID.")
        if self.unk_token_id is not None and (type(self.unk_token_id) is not int or not 0 <= self.unk_token_id < self.vocab_size):
            raise ValueError("UNK должен находиться внутри словаря или быть None.")

    @classmethod
    def from_tokenizer(cls, tokenizer: Tokenizer, **overrides) -> "TransformerConfig":
        """Взять фактический размер словаря и ID спецтокенов из токенизатора."""
        values = {"vocab_size": tokenizer.get_vocab_size(), "unk_token_id": tokenizer.token_to_id('<unk>')}
        for name, token in (
            ("pad_token_id", "<pad>"),
            ("bos_token_id", "<bos>"),
            ("eos_token_id", "<eos>"),
        ):
            token_id = tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(f"В токенизаторе отсутствует {token}.")
            values[name] = token_id
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)
