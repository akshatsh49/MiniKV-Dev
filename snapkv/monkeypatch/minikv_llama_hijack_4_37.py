import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import warnings
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import (
    logging,
)
from snapkv.monkeypatch.snapkv_utils import init_snapkv

import math
from quant.new_pack import triton_quantize_and_pack_along_last_dim
from quant.matmul import cuda_bmm_fA_qB_outer

logger = logging.get_logger(__name__)

# https://github.com/huggingface/transformers/blob/v4.37-release/src/transformers/models/llama/modeling_llama.py
def sparsity_llama_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # [SnapKV] register kv_cluster
    init_snapkv(self)
    # LlamaFlashAttention2 attention does not support output_attentions
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): #[SnapKV] add kv_seq_len
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [SnapKV] move to ahead
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        # print('kv_seq_len:', kv_seq_len)
        # print('key_states.shape:', key_states.shape)
        if key_states.shape[-2] == kv_seq_len: # [SnapKV] add kv_cluster
            self.kv_seq_len = kv_seq_len # [SnapKV] register kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups)
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = self._flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def sparsity_prepare_inputs_for_generation_llama(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is None: # [SnapKV]
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            # cache_length = past_length = past_key_values[0][0].shape[2]
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None
        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

##############
class MKVCache(DynamicCache):
    '''
    The kv cache here is (key_code, key_scale, key_mn, key_unq, value_code, value_scale, value_mn, value_unq)
    '''
    def __init__(self, quant_bits:int = 2, group_size:int = 16, residual_length:int = 128, num_layers = 32):
        super().__init__()
        self.quant_bits = quant_bits
        self.group_size = group_size
        self.residual_length = residual_length
        self.num_layers = num_layers
        assert self.residual_length % self.group_size == 0, f"residual_length should be divisible by group_size, {self.residual_length = } % {self.group_size = } != 0"
        
        self.kv_seq_len = 0
        self._seen_tokens = 0
        self.quantized_cache = [{} for i in range(self.num_layers)] # stores quantized/unq KV values
        # print(f"[INFO] Using MKVCache with quant_bits: {quant_bits}, group_size: {group_size}, residual_length: {residual_length}, num_layers: {num_layers}")
        
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: Optional[dict] = None
    ):
        '''
        should return key_code, key_scale, key_mn, key_unq, value_code, value_scale, value_mn, value_unq
        '''
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        
        # if pre-fill
        # do quantization of the passed key_states and value_states obtained from pre-fill after eviction
        if key_states.shape[-2] != 1: # the pre-fill phase
            num_keys = key_states.shape[-2]
            if num_keys < self.group_size:
                key_code, key_scale, key_mn = None, None, None
            else :
                if num_keys % self.group_size != 0:
                    key_unq = key_states[:, :, -(num_keys % self.group_size):, :]
                    key_states = key_states[:, :, :-(num_keys % self.group_size), :].transpose(2,3).contiguous()
                    key_code, key_scale, key_mn = triton_quantize_and_pack_along_last_dim(key_states, self.group_size, self.quant_bits)
                else :
                    key_code, key_scale, key_mn = triton_quantize_and_pack_along_last_dim(key_states.transpose(2,3).contiguous(), self.group_size, self.quant_bits)
                    key_unq = None
                    
            num_values = value_states.shape[-2]
            if num_values < self.group_size:
                value_code, value_scale, value_mn = None, None, None
            else :
                if num_values % self.group_size != 0:
                    value_unq = value_states[:, :, -(num_values % self.group_size):, :]
                    value_states = value_states[:, :, :-(num_values % self.group_size), :].contiguous()
                    value_code, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states, self.group_size, self.quant_bits)
                else :
                    value_code, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states.contiguous(), self.group_size, self.quant_bits)
                    value_unq = None
                    
            # update cache
            self.quantized_cache[layer_idx] = {
                "key_code": key_code, "key_scale": key_scale, "key_mn": key_mn, "key_unq": key_unq,
                "value_code": value_code, "value_scale": value_scale, "value_mn": value_mn, "value_unq": value_unq
            }
        
        else :  # the generation phase
            key_code, key_scale, key_mn, key_unq, value_code, value_scale, value_mn, value_unq = self.quantized_cache[layer_idx].values()
            
            if key_unq is not None:
                key_unq = torch.cat([key_unq, key_states], dim = 2)
            else:
                key_unq = key_states
            
            if key_unq.shape[-2] >= self.residual_length:
                key_code_new, key_scale_new, key_mn_new = triton_quantize_and_pack_along_last_dim(key_unq.transpose(2,3).contiguous(), self.group_size, self.quant_bits)
                key_unq = None
                if key_code is not None:
                    key_code = torch.cat([key_code, key_code_new], dim = 3)
                    key_scale = torch.cat([key_scale, key_scale_new], dim = 3)
                    key_mn = torch.cat([key_mn, key_mn_new], dim = 3)
                else:
                    key_code, key_scale, key_mn = key_code_new, key_scale_new, key_mn_new
                    
            if value_unq is not None:
                value_unq = torch.cat([value_unq, value_states], dim = 2)
            else:
                value_unq = value_states
                
            if value_unq.shape[-2] >= self.residual_length:
                value_code_new, value_scale_new, value_mn_new = triton_quantize_and_pack_along_last_dim(value_unq.contiguous(), self.group_size, self.quant_bits)
                value_unq = None
                if value_code is not None:
                    value_code = torch.cat([value_code, value_code_new], dim = 2)
                    value_scale = torch.cat([value_scale, value_scale_new], dim = 2)
                    value_mn = torch.cat([value_mn, value_mn_new], dim = 2)
                else:
                    value_code, value_scale, value_mn = value_code_new, value_scale_new, value_mn_new

            # update cache
            self.quantized_cache[layer_idx] = {
                "key_code": key_code, "key_scale": key_scale, "key_mn": key_mn, "key_unq": key_unq,
                "value_code": value_code, "value_scale": value_scale, "value_mn": value_mn, "value_unq": value_unq
            }
            
        return key_code, key_scale, key_mn, key_unq, value_code, value_scale, value_mn, value_unq

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        # return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1
        return 0
    
    def reset(self):
        # self._seen_tokens = 0
        # self.quantized_cache = [{} for i in range(self.num_layers)]
        raise RuntimeError(f"Shouldn't be calling reset explicitly on MKVCache")
        

def minikv_llama_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # [SnapKV] register kv_cluster
    init_snapkv(self)
    
    if isinstance(past_key_value, DynamicCache) and not isinstance(past_key_value, MKVCache):
        raise RuntimeError(f"past_key_value should be of type MKVCache, got {type(past_key_value)}")
        
    # LlamaFlashAttention2 attention does not support output_attentions
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

        # overwrite attention_mask with padding_mask
        attention_mask = kwargs.pop("padding_mask")

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): #[SnapKV] add kv_seq_len
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    
    # [SnapKV] move to ahead
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        if key_states.shape[-2] == kv_seq_len: # [SnapKV] add kv_cluster
            self.kv_seq_len = kv_seq_len # [SnapKV] register kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups)   # only eviction here
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs) # quantization happens here
            
            # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
            # to be able to avoid many of these transpose/reshape/view.
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            dropout_rate = self.attention_dropout if self.training else 0.0

            # In PEFT, usually we cast the layer norms in float32 for training stability reasons
            # therefore the input hidden states gets silently casted in float32. Hence, we need
            # cast them back in the correct dtype just to be sure everything works as expected.
            # This might slowdown training & inference so it is recommended to not cast the LayerNorms
            # in fp32. (LlamaRMSNorm handles it correctly)

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                # Handle the case where the model is quantized
                elif hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)

            attn_output = self._flash_attention_forward(
                query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
            )

        else:
            self.kv_seq_len += q_len
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            key_code, key_scale, key_mn, key_unq, value_code, value_scale, value_mn, value_unq = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            
            # call mkv kernels here to get attn map + attn output
            if key_code is not None:
                att_qkquant = cuda_bmm_fA_qB_outer(self.config.group_size, query_states, key_code, key_scale, key_mn, self.config.quant_bits)
            else:
                att_qkquant = None
                
            if key_unq is not None:
                att_qkfull = torch.matmul(query_states, key_unq.transpose(2,3))
            else :
                att_qkfull = None
            
            attn_list = []
            if att_qkquant is not None:
                attn_list.append(att_qkquant)
            if att_qkfull is not None:
                attn_list.append(att_qkfull)
            attn_weights = torch.cat(attn_list, dim=-1) / math.sqrt(self.head_dim)
            
            # if att_qkquant is not None:
            #     attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            # else:
            #     attn_weights = att_qkfull / math.sqrt(self.head_dim)
            
            # these tests are wrong as kv_seq_len is not the same as number of tokens in the cache, it is the number of tokens + number of evicted tokens, i.e. tokens seen so far
            # if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            #     raise ValueError(
            #         f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            #         f" {attn_weights.size()}"
            #     )
            
            if attention_mask is not None:
                # if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                #     raise ValueError(
                #         f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                #     )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )
            
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            value_full_length = value_unq.shape[-2] if value_unq is not None else 0
            if value_code is None:
                attn_output = torch.matmul(attn_weights, value_unq)
            else:
                if value_full_length != 0:
                    attn_output = cuda_bmm_fA_qB_outer(self.config.group_size, attn_weights[:, :, :, :-value_full_length], value_code, value_scale, value_mn, self.config.quant_bits)
                    attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], value_unq)
                else :  # list[:-0] does not work
                    attn_output = cuda_bmm_fA_qB_outer(self.config.group_size, attn_weights, value_code, value_scale, value_mn, self.config.quant_bits)
                
            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            attn_output = attn_output.transpose(1, 2)   ## to match flash attn output shape
            
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def minikv_prepare_inputs_for_generation_llama(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is None: # [SnapKV]
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0
        # make mkvcache instance here
        past_key_values = MKVCache(quant_bits = self.config.quant_bits, group_size = self.config.group_size, residual_length = self.config.residual_length, num_layers = self.config.num_hidden_layers)
        
    elif past_key_values is not None:
        # if isinstance(past_key_values, Cache):
        #     cache_length = past_key_values.get_seq_length()
        #     past_length = past_key_values.seen_tokens
        #     max_cache_length = past_key_values.get_max_length()
        # else:
        #     # cache_length = past_length = past_key_values[0][0].shape[2]
        #     cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
        #     max_cache_length = None
            
        cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
        max_cache_length = None
            
            
            
        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values is not None:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
