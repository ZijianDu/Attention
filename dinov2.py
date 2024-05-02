from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.dinov2.modeling_dinov2 import (
    Dinov2Embeddings,
    Dinov2SelfAttention,
    Dinov2SelfOutput,
    Dinov2Attention,
    Dinov2Encoder,
    Dinov2Layer,
    Dinov2Model,
    Dinov2PreTrainedModel,
    Dinov2LayerScale,
    Dinov2MLP
)
from transformers.modeling_outputs import (
    BackboneOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)
from transformers.models.dinov2.configuration_dinov2 import Dinov2Config
from transformers.modeling_outputs import BaseModelOutput


@dataclass
class BaseModelOutputWithPoolingwAttentionScores(BaseModelOutputWithPooling):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    attention_scores: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass
class BaseModelOutputwAttentionScores(BaseModelOutputWithPooling):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[Tuple[torch.FloatTensor, ...]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    attention_scores: Optional[Tuple[torch.FloatTensor, ...]] = None

class Dinov2SelfAttentionwOutput(Dinov2SelfAttention):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        ## add class member to directly access q, k, v
        self.key_layer = None
        self.mixed_query_layer = None
        self.value_layer = None
        self.query_layer = None

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def forward(self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ):
        self.mixed_query_layer = self.query(hidden_states)
        self.key_layer = self.transpose_for_scores(self.key(hidden_states))
        self.value_layer = self.transpose_for_scores(self.value(hidden_states))
        self.query_layer = self.transpose_for_scores(self.mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(self.query_layer, self.key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, self.value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # add attention scores into the output and propragate to higher class
        ##key_layer, query_layer, value_layer, 
        outputs = (context_layer, attention_probs, attention_scores) if output_attentions else (context_layer,)

        return outputs
    
class Dinov2AttentionwOutput(Dinov2Attention):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)
        self.attention = Dinov2SelfAttentionwOutput(config)
        self.output = Dinov2SelfOutput(config)
        self.pruned_heads = set()

class Dinov2LayerwOutput(Dinov2Layer):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Dinov2AttentionwOutput(config)
        self.layer_scale1 = Dinov2LayerScale(config)
        self.drop_path = Dinov2DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.use_swiglu_ffn:
            self.mlp = Dinov2SwiGLUFFN(config)
        else:
            self.mlp = Dinov2MLP(config)
        self.layer_scale2 = Dinov2LayerScale(config)


class Dinov2EncoderwOutput(Dinov2Encoder):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList([Dinov2LayerwOutput(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = True,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_self_attention_scores = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_self_attention_scores = all_self_attention_scores + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions, all_self_attention_scores] if v is not None)
        return BaseModelOutputwAttentionScores(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            attention_scores=all_self_attention_scores,
        )

        
class Dinov2ModelwOutput(Dinov2PreTrainedModel):
    def __init__(self, config: Dinov2Config):
        super().__init__(config)
        self.config = config

        self.embeddings = Dinov2Embeddings(config)
        self.encoder = Dinov2EncoderwOutput(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

        # class member to store key, query and value for all iterations
        self.key = None
        self.query = None
        self.value = None
        
        # pick certain channel of q, k, v
        self.picked_channel = 0

    def getqkv(self):
        return (self.query, self.key, self.value)
    
    def getkey(self):
        assert self.key is not None
        return self.key

    def forward(self, layer_idx,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None, 
        return_dict: Optional[bool] = True,) -> Union[Tuple, BaseModelOutputWithPoolingwAttentionScores]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos).cuda()
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            return head_outputs + encoder_outputs[1:]

        ## set qkv for each iteration
        ## CLS token w.r.t all other tokens
        self.key = self.encoder.layer[layer_idx].attention.attention.key_layer[:, :, 1:, :]
        torch.cuda.empty_cache()

        return BaseModelOutputWithPoolingwAttentionScores(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            attention_scores=encoder_outputs.attention_scores,
        )




