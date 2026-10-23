"""BART with a differentiable pointer-generator, trained on the same code labels."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForSeq2SeqLM, BartConfig, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils.generic import can_return_tuple

from .base import BaseSeq2SeqModel
from .baseline import BaselineTransformer, make_bart_config
from .config import TransformerConfig


class PointerGenerator(nn.Module):
    """Single copy attention; no full layer attention matrices or extra targets.

    Returns FP32 log probabilities. HF softmax/cross_entropy therefore describe
    the same mixture as NLL, while greedy/beam generation can use the usual API.
    """
    def __init__(self, vocab_size: int, hidden_size: int, copy_dim: int, excluded_ids):
        super().__init__()
        self.vocab_size = vocab_size
        self.query = nn.Linear(hidden_size, copy_dim, bias=False)
        self.key = nn.Linear(hidden_size, copy_dim, bias=False)
        self.gate = nn.Linear(2 * hidden_size, 1)
        self.excluded_ids = tuple(excluded_ids)

    def forward(self, vocab_logits, decoder_states, encoder_states, source_ids, source_mask):
        valid = source_mask.bool()
        for token_id in self.excluded_ids:
            valid = valid & source_ids.ne(token_id)
        query, key = self.query(decoder_states), self.key(encoder_states)
        # Explicitly disable AMP for these matmuls; .float() alone is not enough
        # inside autocast. Avoid overflow and preserve tiny mixture probabilities.
        with torch.autocast(device_type=decoder_states.device.type, enabled=False):
            scores = (query.float() @ key.float().transpose(-1,-2)) / math.sqrt(query.shape[-1])
            scores = scores.masked_fill(~valid[:,None,:], -1e9)
            attention = scores.softmax(-1) * valid[:,None,:]
            attention = attention / attention.sum(-1,keepdim=True).clamp_min(1e-20)
            context = attention @ encoder_states.float()
            copy_prob = torch.zeros_like(vocab_logits,dtype=torch.float32).scatter_add(
                -1,source_ids[:,None,:].expand(-1,attention.shape[1],-1),attention)
        gate = self.gate(torch.cat((decoder_states,context.to(decoder_states.dtype)),dim=-1)).float()
        has_copy = valid.any(-1)[:,None,None]
        log_generate = torch.where(has_copy,F.logsigmoid(gate),torch.zeros_like(gate))
        log_copy = torch.where(has_copy,F.logsigmoid(-gate),torch.full_like(gate,-torch.inf))
        # Clamp before log so masked zero-probability branches have finite grads.
        copy_log = copy_prob.clamp_min(torch.finfo(torch.float32).tiny).log()
        copy_log = copy_log.masked_fill(copy_prob.eq(0),-torch.inf)
        return torch.logaddexp(F.log_softmax(vocab_logits.float(),-1)+log_generate,copy_log+log_copy)


class CopyBartConfig(BartConfig):
    model_type = 'copy_bart'

    def __init__(self, copy_attention_dim=128, copy_excluded_ids=None, **kwargs):
        super().__init__(**kwargs)
        self.copy_attention_dim = copy_attention_dim
        self.copy_excluded_ids = list(copy_excluded_ids if copy_excluded_ids is not None else
            {self.pad_token_id,self.bos_token_id,self.eos_token_id})


@dataclass
class CopyEncoderOutput(BaseModelOutput):
    source_input_ids: torch.Tensor | None = None


class CopyBartForConditionalGeneration(BartForConditionalGeneration):
    config_class = CopyBartConfig

    def __init__(self, config):
        super().__init__(config)
        self.copy_head = PointerGenerator(config.vocab_size,config.d_model,
            config.copy_attention_dim,config.copy_excluded_ids)
        self.copy_head.apply(self._init_weights)
        with torch.no_grad():
            self.copy_head.gate.bias.fill_(2.)  # begin close to ordinary BART

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor, model_kwargs,
        model_input_name, generation_config):
        model_kwargs = super()._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor,model_kwargs,model_input_name,generation_config)
        source = inputs_tensor if model_input_name in (None,'input_ids') else model_kwargs.get('source_input_ids')
        if source is None:
            raise ValueError('Copy generation with embeddings needs source_input_ids.')
        model_kwargs['encoder_outputs'] = CopyEncoderOutput(
            **model_kwargs['encoder_outputs'], source_input_ids=source)
        return model_kwargs

    @can_return_tuple
    def forward(self,input_ids=None,attention_mask=None,decoder_input_ids=None,
        decoder_attention_mask=None,encoder_outputs=None,past_key_values=None,
        inputs_embeds=None,decoder_inputs_embeds=None,labels=None,use_cache=None,
        source_input_ids=None,**kwargs):
        if source_input_ids is None:
            source_input_ids = input_ids if input_ids is not None else getattr(encoder_outputs,'source_input_ids',None)
        if source_input_ids is None:
            raise ValueError('Copy model requires source input IDs, including when using encoder_outputs.')
        if attention_mask is None:
            attention_mask = source_input_ids.ne(self.config.pad_token_id)
        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)
        outputs = self.model(input_ids=input_ids,attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,return_dict=True,**kwargs)
        vocab_logits = self.lm_head(outputs.last_hidden_state) + self.final_logits_bias
        logits = self.copy_head(vocab_logits,outputs.last_hidden_state,
            outputs.encoder_last_hidden_state,source_input_ids,attention_mask)
        loss = None if labels is None else F.nll_loss(logits.reshape(-1,self.config.vocab_size),
            labels.to(logits.device).reshape(-1),ignore_index=-100)
        return Seq2SeqLMOutput(loss=loss,logits=logits,past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,encoder_attentions=outputs.encoder_attentions)


class CopyTransformer(BaselineTransformer):
    def __init__(self, config: TransformerConfig):
        BaseSeq2SeqModel.__init__(self,config)
        values = make_bart_config(config).to_dict()
        values.pop('model_type',None)
        excluded = {config.pad_token_id,config.bos_token_id,config.eos_token_id}
        if config.unk_token_id is not None:
            excluded.add(config.unk_token_id)
        hf_config = CopyBartConfig(**values,copy_attention_dim=min(128,config.d_model),
            copy_excluded_ids=sorted(excluded))
        self.network = CopyBartForConditionalGeneration(hf_config)


AutoConfig.register(CopyBartConfig.model_type,CopyBartConfig)
AutoModelForSeq2SeqLM.register(CopyBartConfig,CopyBartForConditionalGeneration)
