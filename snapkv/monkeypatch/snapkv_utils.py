
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

class SparsityKVCluster():
    def __init__(self, window_size = 64, prompt_sparsity_ratio = 0.25, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.prompt_sparsity_ratio = prompt_sparsity_ratio
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, prompt_sparsity_ratio = 0.25, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.prompt_sparsity_ratio = prompt_sparsity_ratio
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if int(self.prompt_sparsity_ratio * q_len) < self.window_size:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(int(self.prompt_sparsity_ratio * q_len) - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

class Snap_MiniKVCluster():
    def __init__(self, window_size = 64, prompt_sparsity_ratio = 0.25, kernel_size = 5, pooling = 'avgpool', quant_bits = 2, group_size = 16, residual_length = 128):
        self.window_size = window_size
        self.prompt_sparsity_ratio = prompt_sparsity_ratio
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.quant_bits = quant_bits
        self.group_size = group_size
        self.residual_length = residual_length
        print(f"[Snap_MKV CLUSTER] window_size : {window_size}, prompt_sparsity_ratio : {prompt_sparsity_ratio}, kernel_size : {kernel_size}, pooling : {pooling}, quant_bits : {quant_bits}, group_size : {group_size}, residual_length : {residual_length}")

    def reset(self, window_size = 64, prompt_sparsity_ratio = 0.25, kernel_size = 5, pooling = 'avgpool'):
        raise NotImplementedError

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if int(self.prompt_sparsity_ratio * q_len) < self.window_size:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(int(self.prompt_sparsity_ratio * q_len) - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            return key_states, value_states

class MiniKVCluster():
    def __init__(self, heavy_ratio = 0.25, recent_ratio = 0.25, quant_bits = 2, group_size = 16, residual_length = 128):
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio
        self.quant_bits = quant_bits
        self.group_size = group_size
        self.residual_length = residual_length
        print(f"[MKV CLUSTER] heavy_ratio : {heavy_ratio}, recent_ratio : {recent_ratio}, quant_bits : {quant_bits}, group_size : {group_size}, residual_length : {residual_length}")

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def update_kv(self, key_states, query_states, value_states, cumulative_attn_map):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        num_hh = int(q_len * self.heavy_ratio)
        num_rw = int(q_len * self.recent_ratio)
        
        if num_hh + num_rw >= q_len:
            return key_states, value_states
        else:
            hh_score = cumulative_attn_map

            select_hh_scores = hh_score[:, :, :q_len - num_rw]
            _, keep_topk = torch.topk(select_hh_scores, num_hh, dim=-1)
            keep_topk = keep_topk.sort().values

            keep_recent = torch.arange(q_len - num_rw, q_len, device=keep_topk.device).repeat(keep_topk.shape[0], keep_topk.shape[1], 1)
            keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)

            mask = torch.zeros(hh_score.shape, dtype=torch.bool).to(keep_topk.device)
            mask = mask.scatter(-1, keep_idx, 1)
            
            key_states_compress = key_states[mask].view(bsz, num_heads, -1, head_dim)
            value_states_compress = value_states[mask].view(bsz, num_heads, -1, head_dim)
            
            return key_states_compress, value_states_compress
    
def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not self.config.use_snap:
            assert not self.config.use_snap and hasattr(self.config, 'heavy_ratio')
            print(f"[INFO] Loading MiniKVCluster")
            self.kv_cluster = MiniKVCluster(
                heavy_ratio = self.config.heavy_ratio,
                recent_ratio = self.config.recent_ratio,
                quant_bits = self.config.quant_bits,
                group_size = self.config.group_size,
                residual_length = self.config.residual_length,
                )
        else:
            if self.config.quant_bits == 16:
                print(f"[INFO] Loading SparsityKVCluster")
                self.kv_cluster = SparsityKVCluster(
                    window_size = self.config.window_size, 
                    prompt_sparsity_ratio = self.config.prompt_sparsity_ratio, 
                    kernel_size = self.config.kernel_size,
                    pooling = self.config.pooling
                    )
            else :
                print(f"[INFO] Loading Snap_MiniKVCluster")
                self.kv_cluster = Snap_MiniKVCluster(
                    window_size = self.config.window_size, 
                    prompt_sparsity_ratio = self.config.prompt_sparsity_ratio, 
                    kernel_size = self.config.kernel_size,
                    pooling = self.config.pooling,
                    quant_bits = self.config.quant_bits,
                    group_size = self.config.group_size,
                    residual_length = self.config.residual_length,
                    )
                
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
