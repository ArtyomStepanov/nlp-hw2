"""T5-style encoder–decoder from scratch, using our unchanged BPE contract."""
from transformers import T5Config, T5ForConditionalGeneration

from .base import BaseSeq2SeqModel
from .baseline import BaselineTransformer
from .config import TransformerConfig
from .t5_attention import enable_t5_sdpa


class T5Transformer(BaselineTransformer):
    """Reuse the generic HF forward/generate wrapper, replacing its network.

    Pre-RMSNorm, relative attention bias, GEGLU and tied embeddings. Native T5
    dropout applies to attention as well as FFN/residuals (attention_dropout in
    TransformerConfig is a BART-only option). BOS starts decoding so the saved
    BART training data, whose labels exclude BOS, needs no transformation.
    """

    def __init__(self, config: TransformerConfig) -> None:
        BaseSeq2SeqModel.__init__(self, config)
        hf_config = T5Config(
            vocab_size=config.vocab_size, d_model=config.d_model,
            d_kv=config.d_model // config.attention_heads, d_ff=config.ffn_dim,
            num_layers=config.encoder_layers, num_decoder_layers=config.decoder_layers,
            num_heads=config.attention_heads, feed_forward_proj='gated-gelu',
            relative_attention_num_buckets=32, relative_attention_max_distance=128,
            dropout_rate=config.dropout, tie_word_embeddings=True,
            pad_token_id=config.pad_token_id, bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id, decoder_start_token_id=config.bos_token_id,
            use_cache=True,
        )
        self.network = enable_t5_sdpa(T5ForConditionalGeneration(hf_config))
