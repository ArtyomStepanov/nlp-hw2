"""Базовый encoder–decoder Transformer со случайной инициализацией."""

import torch
from torch import Tensor
from transformers import BartConfig, BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from .base import BaseSeq2SeqModel
from .config import TransformerConfig


class BaselineTransformer(BaseSeq2SeqModel):
    """BART-архитектура для обучения с нуля, без загрузки готовых весов.

    Shared embeddings связывают энкодер, декодер и выходную проекцию.
    forward использует teacher forcing, generate — self/cross-attention KV-кэш.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        bart_config = BartConfig(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            encoder_attention_heads=config.attention_heads,
            decoder_attention_heads=config.attention_heads,
            encoder_ffn_dim=config.ffn_dim,
            decoder_ffn_dim=config.ffn_dim,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_function="gelu",
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            decoder_start_token_id=config.bos_token_id,
            forced_eos_token_id=None,
            tie_word_embeddings=True,
            use_cache=True,
        )
        self.network = BartForConditionalGeneration(bart_config)

    def _check_token_batch(self, tokens: Tensor, name: str, batch_size: int | None = None) -> None:
        if tokens.ndim != 2 or not tokens.shape[0] or not tokens.shape[1]:
            raise ValueError(f"{name} должен иметь непустую форму [batch, length].")
        if tokens.shape[1] > self.config.max_position_embeddings:
            raise ValueError(f"Длина {name} превышает max_position_embeddings.")
        if batch_size is not None and tokens.shape[0] != batch_size:
            raise ValueError(f"Размер батча {name} отличается от input_ids.")

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        decoder_input_ids: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
    ) -> Seq2SeqLMOutput:
        """labels: целевой код с EOS, без начального BOS; PAD игнорируется.

        Используйте prepare_labels(target_ids) для батча от нашего токенизатора.
        Вместо labels можно передать decoder_input_ids для получения logits.
        """
        self._check_token_batch(input_ids, "input_ids")
        if labels is None and decoder_input_ids is None:
            raise ValueError("Передайте labels или decoder_input_ids; для генерации используйте generate.")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        if labels is not None:
            self._check_token_batch(labels, "labels", input_ids.shape[0])
            labels = labels.masked_fill(labels == self.config.pad_token_id, -100)
            if decoder_input_ids is None:
                decoder_input_ids = self.network.prepare_decoder_input_ids_from_labels(labels)
        self._check_token_batch(decoder_input_ids, "decoder_input_ids", input_ids.shape[0])
        if decoder_attention_mask is None:
            decoder_attention_mask = decoder_input_ids.ne(self.config.pad_token_id)
        return self.network(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        max_new_tokens: int = 256,
        num_beams: int = 1,
    ) -> Tensor:
        """Батчевая генерация; num_beams=1 задаёт greedy decoding."""
        self._check_token_batch(input_ids, "input_ids")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens < self.config.max_position_embeddings:
            raise ValueError("max_new_tokens должен быть положительным и меньше max_position_embeddings.")
        if isinstance(num_beams, bool) or not isinstance(num_beams, int) or num_beams < 1:
            raise ValueError("num_beams должен быть положительным целым числом.")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        was_training = self.training
        self.eval()
        try:
            return self.network.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                use_cache=True,
            )
        finally:
            self.train(was_training)
