"""Instance-local SDPA for the eager-only T5 in Transformers 5.10.2.

Projection/cache/bias semantics follow Hugging Face modeling_t5.T5Attention
(Apache-2.0). Standard HF checkpoints remain readable without this adapter.
"""
import torch
from torch.nn import functional as F
from transformers.cache_utils import EncoderDecoderCache
from transformers.models.t5.modeling_t5 import T5Attention


class T5SdpaAttention(T5Attention):
    def forward(self, hidden_states, mask=None, key_value_states=None,
                position_bias=None, past_key_values=None, output_attentions=False, **kwargs):
        if output_attentions:
            return super().forward(hidden_states, mask=mask, key_value_states=key_value_states,
                position_bias=position_bias, past_key_values=past_key_values,
                output_attentions=True, **kwargs)
        input_shape = hidden_states.shape[:-1]
        past_seen = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
        if isinstance(past_seen, torch.Tensor):
            past_seen = past_seen.clone()
        query = self.q(hidden_states).view(*input_shape, -1, self.key_value_proj_dim).transpose(1, 2)
        cross = key_value_states is not None
        updated = False
        cache = past_key_values
        if isinstance(cache, EncoderDecoderCache):
            updated = cache.is_updated.get(self.layer_idx)
            cache = cache.cross_attention_cache if cross else cache.self_attention_cache
        if cross and cache is not None and updated:
            key = cache.layers[self.layer_idx].keys
            value = cache.layers[self.layer_idx].values
        else:
            states = key_value_states if cross else hidden_states
            shape = (*states.shape[:-1], -1, self.key_value_proj_dim)
            key = self.k(states).view(shape).transpose(1, 2)
            value = self.v(states).view(shape).transpose(1, 2)
            if cache is not None:
                key, value = cache.update(key, value, self.layer_idx)
                if cross and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(input_shape[1], key.shape[-2],
                    device=query.device, past_seen_tokens=past_seen)
            else:
                position_bias = torch.zeros((1, query.shape[1], input_shape[1], key.shape[-2]),
                    device=query.device, dtype=query.dtype)
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            if mask is not None:
                position_bias = position_bias + mask[:, :, :, :key.shape[-2]]
        # T5 intentionally has no 1/sqrt(d_k) scaling. The shared bias already
        # includes padding/causality; adding is_causal would apply a second mask
        # and can misalign incremental decoding with a non-square KV cache.
        attended = F.scaled_dot_product_attention(query, key, value,
            attn_mask=position_bias.to(query.dtype),
            dropout_p=self.dropout if self.training else 0.0, scale=1.0)
        attended = attended.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        return self.o(attended), position_bias


def enable_t5_sdpa(network):
    """Change attention dispatch without reinitializing or renaming parameters."""
    for module in network.modules():
        if type(module) is T5Attention:
            module.__class__ = T5SdpaAttention
    return network
