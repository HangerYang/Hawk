# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from huggingface_hub import snapshot_download
from torch import nn
from transformers import LlamaConfig
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from ...data.data_utils import process_token_dict_to_mappings
from ..model_utils import apply_rotary_pos_emb, apply_rotary_pos_emb_mrope, repeat_kv
from .base_model import Eagle3BaseDraftModel
from .draft_model_factory import DraftModelFactory

logger = logging.getLogger(__name__)

# Two injection modes ship here. ``fused_fc`` is stock EAGLE-3: three target
# layers hard-concatenated into the nH->H fusion FC. ``banded_mix_fc`` keeps
# that shape -- 1 draft layer, fusion FC, fc_norm/norm_output -- and only
# replaces each raw FC input stream with a learned softmax mix over a
# contiguous band of target layers.
BANDED_MIX_MODES = frozenset({"banded_mix_fc"})


def _num_aux_bands(config) -> int:
    """Band count for the banded modes; 0 when the config has no bands."""
    bands = getattr(config, "eagle_aux_layer_bands", None)
    return len(bands) if bands else 0


def infer_target_layer_weight_prefix(embed_weight_key: str) -> str:
    """Derive target decoder layer prefix from the embedding weight key."""
    suffix = "embed_tokens.weight"
    if not embed_weight_key.endswith(suffix):
        raise ValueError(
            f"Cannot infer layer prefix from embed_weight_key={embed_weight_key!r}; "
            "expected a key ending with 'embed_tokens.weight', or pass "
            "target_layer_weight_prefix explicitly."
        )
    return embed_weight_key[: -len(suffix)] + "layers"


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation
        # in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # Callers index the result by absolute position_ids, not by a local
        # 0..seq_len-1 range -- true whenever a caller (e.g. vistoken row
        # compression) keeps original absolute positions with gaps. `seq_len`
        # here is just a lower bound used to decide whether to grow the
        # cache, so the returned table must stay uncut at its full cached
        # width rather than truncated to `:seq_len`.
        return (
            self.cos_cached.to(dtype=x.dtype),
            self.sin_cached.to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """
    LlamaRotaryEmbedding extended with linear scaling.
    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation
        # in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling.
    Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation
        # in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class MRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type", "default")
            )
            # mrope is actually a special case of default RoPE that uses the same inv_freq
            # calculation but with different layout in forward.
            if self.rope_type == "mrope":
                self.rope_type = "default"
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        if self.rope_type == "default":
            # Standard RoPE: no scaling, equivalent to _compute_default_rope_parameters
            inv_freq, self.attention_scaling = self.compute_default_rope_parameters(config, device)
        else:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

    @staticmethod
    def compute_default_rope_parameters(config, device=None):
        """mrope reuses default RoPE's inv_freq (only the forward layout differs).

        Also satisfies transformers' generic post-load re-init path, which
        matches any *RotaryEmbedding class with an `original_inv_freq` buffer
        and, for rope_type == "default", calls `module.compute_default_rope_parameters`
        to recompute it -- without this method that lookup raises AttributeError.
        In transformers>=5.x, rope_theta is merged into rope_scaling/rope_parameters
        dict, so we need to check there first, then fallback to config attribute.
        """
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        base = rope_scaling.get("rope_theta", None) or getattr(config, "rope_theta", 10000.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
                / dim
            )
        )
        return inv_freq, 1.0

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, x, position_ids, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            # expand (batch, seq_len) to (3, batch, seq_len), match MRoPE T/H/W layout
            position_ids = position_ids[None].expand(3, position_ids.shape[0], -1)

        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = (
            x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        # Eagle layer 0 concatenates the token embedding with the fused target
        # stream, so its QKV is 2H. Deeper draft layers are a stock H block.
        qkv_streams = 2 if layer_idx == 0 else 1
        qkv_in = self.hidden_size * qkv_streams
        self.q_proj = nn.Linear(qkv_in, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            qkv_in, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            qkv_in, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK-norm (Qwen3 / Gemma-style): per-head RMSNorm on Q and K before
        # RoPE. Weights init to ones, so an untrained draft starts identical to
        # the no-QK-norm model and nothing needs copying from the target.
        self.use_qk_norm = bool(getattr(config, "qk_norm", False))
        if self.use_qk_norm:
            self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None
        self._init_rope()

    def _init_rope(self):
        self.rope_apply_func = apply_rotary_pos_emb
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings
            )
        else:
            scaling_type = self.config.rope_scaling.get(
                "type", self.config.rope_scaling.get("rope_type", "default")
            )
            if scaling_type == "mrope" or self.config.rope_scaling.get("mrope_interleaved", False):
                self.rotary_emb = MRotaryEmbedding(self.config)
                self.rope_apply_func = apply_rotary_pos_emb_mrope
            elif scaling_type == "default":
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings
                )
            elif scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=self.config.rope_scaling["factor"],
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=self.config.rope_scaling["factor"],
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        lck = len(cache_hidden[0])

        # cache_k = [self.k_proj(hidden) for hidden in cache_hidden]
        # cache_v = [self.v_proj(hidden) for hidden in cache_hidden]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # RMSNorm over head_dim (last dim), applied pre-RoPE.
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        cos, sin = self.rotary_emb(
            query_states, seq_len=q_len + lck, position_ids=position_ids + lck
        )
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        query_states, key_states = self.rope_apply_func(
            query_states, key_states, cos, sin, position_ids + lck
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Avoid modify hidden cache inplace which will cause in-place
        # modification error when enable gradient checkpoint.

        # Return the updated hidden cache instead.
        if cache_hidden is None:
            local_cache_k = []
            local_cache_v = []
        else:
            local_cache_k = list(cache_hidden[0])
            local_cache_v = list(cache_hidden[1])

        local_cache_k.append(key_states)
        local_cache_v.append(value_states)

        cache_k = local_cache_k
        cache_v = local_cache_v

        k0 = cache_k[0]
        v0 = cache_v[0]

        lck = len(cache_k)

        if lck == 1:
            # cuDNN SDPA backend (PyTorch 2.7+) does not support arbitrary 4D
            # float attention masks and will fail during backward with:
            #   "Expected mha_graph->execute(...).is_good() to be true"
            # Disable cuDNN backend so that Flash Attention / math backends
            # handle the 4D causal mask correctly.
            _sdpa_backends = [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
            with torch.nn.attention.sdpa_kernel(_sdpa_backends):
                attn_output = torch.nn.functional.scaled_dot_product_attention(
                    query_states, k0, v0, attn_mask=attention_mask
                )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)
            new_past_key_value = [local_cache_k, local_cache_v]
            return attn_output, new_past_key_value

        attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(self.head_dim)

        attn_weights = attn_weights + attention_mask

        for i in range(1, lck):
            ki = cache_k[i]

            qi = query_states
            kiq = ki

            attn_weightsi = (qi * kiq).sum(-1) / math.sqrt(self.head_dim)
            attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        # k0 length, not q_len: they are equal in TTT training (the whole
        # sequence is re-fed each step) but differ during incremental decode,
        # where q_len == 1 while cache_hidden[0] holds the whole prefill.
        k0_len = k0.shape[-2]
        attn_weights0 = attn_weights[..., :k0_len]

        attn_output = torch.matmul(attn_weights0, v0)

        for i in range(1, lck):
            vi = cache_v[i]
            attn_weightsi = attn_weights[..., k0_len + i - 1]
            attn_outputi = attn_weightsi[..., None] * vi
            attn_output = attn_output + attn_outputi

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        # Return the updated hidden cache.
        new_past_key_value = [local_cache_k, local_cache_v]
        return attn_output, new_past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayeremb(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        # Layer 0 takes the dual-norm Eagle path; deeper layers are H only.
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): residual stream `[B,S,H]`.
        """

        residual = hidden_states

        if self.layer_idx == 0:
            if input_emb is None:
                raise ValueError("Eagle layer 0 requires input_emb (token embeds).")
            attn_input = torch.cat(
                (self.input_layernorm(input_emb), self.hidden_norm(hidden_states)),
                dim=-1,
            )
            return_hidden = attn_input
        else:
            attn_input = self.input_layernorm(hidden_states)
            return_hidden = attn_input

        # Self Attention
        hidden_states, latest_hidden_cache = self.self_attn(
            cache_hidden=cache_hidden,
            hidden_states=attn_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, return_hidden)

        return outputs, latest_hidden_cache


@DraftModelFactory.register
class Eagle3LlamaForCausalLM(Eagle3BaseDraftModel):
    config_class = LlamaConfig

    def __init__(self, config):
        super().__init__(config)
        num_layers = getattr(config, "num_hidden_layers", 1)
        mode = getattr(config, "eagle_aux_injection_mode", "fused_fc")
        if mode not in ("fused_fc", "banded_mix_fc"):
            raise ValueError(
                "eagle_aux_injection_mode must be 'fused_fc' or 'banded_mix_fc', "
                f"got {mode!r}"
            )
        # Band mix in front of the stock fused_fc path.
        self.banded_mix_fc = mode in BANDED_MIX_MODES
        self.layers = nn.ModuleList(
            [LlamaDecoderLayeremb(config, layer_idx=i) for i in range(num_layers)]
        )

        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.padding_idx = config.pad_token_id
        self.hidden_size = config.hidden_size
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # EAGLE 3.1: per-stream RMSNorm before FC, and post-norm next-step HS.
        # Both default off so existing EAGLE 3 checkpoints stay valid.
        self.norm_output = bool(getattr(config, "norm_output", False))
        use_fc_norm = bool(getattr(config, "fc_norm", False))
        aux_ids = getattr(config, "aux_hidden_states_layer_ids", None)
        self.num_aux_hidden_states = len(aux_ids) if aux_ids else 3
        self.aux_layer_bands: Tuple[Tuple[int, ...], ...] = tuple()
        self.aux_mix_norms: Optional[nn.ModuleList] = None
        if self.banded_mix_fc:
            raw_bands = getattr(config, "eagle_aux_layer_bands", None)
            if not raw_bands:
                raise ValueError("banded_mix_fc requires eagle_aux_layer_bands")
            self.aux_layer_bands = tuple(
                tuple(int(layer_id) for layer_id in band) for band in raw_bands
            )
            # The bands are FC input streams, so their count is free of the
            # draft's layer count.
            if any(not band for band in self.aux_layer_bands):
                raise ValueError(
                    f"{mode} requires non-empty aux bands; got {self.aux_layer_bands}"
                )
            flat_band_ids = [layer_id for band in self.aux_layer_bands for layer_id in band]
            if aux_ids is None or flat_band_ids != [int(layer_id) for layer_id in aux_ids]:
                raise ValueError(
                    "aux_hidden_states_layer_ids must equal the flattened "
                    f"eagle_aux_layer_bands; got {aux_ids} vs {flat_band_ids}"
                )
            init_layer_ids = list(
                getattr(
                    config,
                    "eagle_aux_band_init_layer_ids",
                    [band[0] for band in self.aux_layer_bands],
                )
            )
            if len(init_layer_ids) != len(self.aux_layer_bands):
                raise ValueError(
                    "eagle_aux_band_init_layer_ids must have one id per aux band"
                )
            for band_idx, band in enumerate(self.aux_layer_bands):
                # Uniform init: all-zeros logits → equal softmax weight across all
                # layers in the band. Lets training discover the best mix freely.
                logits = torch.zeros(len(band), dtype=torch.float32)
                self.register_parameter(f"band{band_idx}_mix_logits", nn.Parameter(logits))
            # When the mixed streams feed the fusion FC and fc_norm is on,
            # fc_norm already normalises them (and the draft outs at step 1+),
            # so a second RMSNorm here is redundant. Skip building it — unused
            # parameters would also trip DDP.
            if bool(getattr(config, "fc_norm", False)) and self.banded_mix_fc:
                self.aux_mix_norms = None
            else:
                self.aux_mix_norms = nn.ModuleList(
                    [
                        LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                        for _ in self.aux_layer_bands
                    ]
                )
        # Streams reaching the fusion FC. banded_mix_fc collapses the raw aux
        # layers into one mixed stream per band, so the fc_norm count follows
        # the band count.
        self.num_fusion_streams = (
            len(self.aux_layer_bands) if self.banded_mix_fc else self.num_aux_hidden_states
        )
        self.fc_norm: Optional[nn.ModuleList] = None
        # Stock Eagle3: early fc nH->H, then the 2H Eagle layer 0.
        self.fc = nn.Linear(
            self.hidden_size * self.num_fusion_streams,
            self.hidden_size,
            bias=False,
        )
        if use_fc_norm:
            self.fc_norm = nn.ModuleList(
                [
                    LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                    for _ in range(self.num_fusion_streams)
                ]
            )
        # Aux streams carried between speculative steps (shifted per step).
        self._aux_inject: Optional[Tuple[torch.Tensor, ...]] = None
        self._last_layer_outs: Optional[List[torch.Tensor]] = None
        if self.norm_output or self.fc_norm is not None:
            logger.info(
                "EAGLE 3.1 enabled: fc_norm=%s norm_output=%s",
                self.fc_norm is not None,
                self.norm_output,
            )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # create vocab buffers
        t2d = torch.zeros(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

        # Required by new transformers gradient checkpointing format
        self.gradient_checkpointing = False

        # transformers>=5 sets all_tied_weights_keys here; from_pretrained needs it.
        self.post_init()

    @property
    def midlayer(self):
        """Legacy alias for the first draft layer (Eagle 2H block)."""
        return self.layers[0]

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Target HS are often bf16 while --draft_model_dtype float32 keeps draft
        # params in fp32 (for FP32 Adam moments under plain DDP). Match dtypes.
        draft_dtype = next(self.parameters()).dtype
        if hidden_states.dtype != draft_dtype:
            hidden_states = hidden_states.to(dtype=draft_dtype)
        if self.banded_mix_fc:
            expected = self.hidden_size * self.num_aux_hidden_states
            if hidden_states.shape[-1] != expected:
                raise ValueError(
                    "banded_mix_fc expects concat aux HS of size "
                    f"{expected}, got {hidden_states.shape[-1]}"
                )
            source_chunks = hidden_states.split(self.hidden_size, dim=-1)
            mixed_chunks = []
            offset = 0
            _mix_norms = self.aux_mix_norms or [None] * len(self.aux_layer_bands)
            for band_idx, (band, norm) in enumerate(
                zip(self.aux_layer_bands, _mix_norms)
            ):
                band_chunks = source_chunks[offset : offset + len(band)]
                band_stack = torch.stack(band_chunks, dim=-1)
                logits = getattr(self, f"band{band_idx}_mix_logits")
                weights = torch.softmax(logits.float(), dim=0).to(
                    device=band_stack.device, dtype=band_stack.dtype
                )
                _mixed = torch.matmul(band_stack, weights)
                mixed_chunks.append(norm(_mixed) if norm is not None else _mixed)
                offset += len(band)
            # Band mix is the front end: hand the mixed streams to the stock
            # nH->H fusion FC below, so the bands look like ordinary aux streams.
            hidden_states = torch.cat(mixed_chunks, dim=-1)
        if self.fc_norm is not None:
            chunks = hidden_states.split(self.hidden_size, dim=-1)
            if len(chunks) != len(self.fc_norm):
                raise ValueError(
                    f"fc_norm expects {len(self.fc_norm)} aux streams, "
                    f"got {len(chunks)} (last dim={hidden_states.shape[-1]})"
                )
            hidden_states = torch.cat(
                [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)],
                dim=-1,
            )
        return self.fc(hidden_states)

    def next_hidden_from_encode(self, prenorm: torch.Tensor) -> torch.Tensor:
        """Hidden states to feed the next speculative draft step.

        EAGLE 3.1 ``norm_output`` uses the final RMSNorm so residual magnitude
        does not grow across draft steps. EAGLE 3 keeps the pre-norm residual.
        """
        return self.norm(prenorm) if self.norm_output else prenorm

    def shift_aux_inject(self, left: bool = False):
        """Pad/shift the stored aux injects by one position."""
        if self._aux_inject is None:
            return
        from angelslim.compressor.speculative.utils import padding as pad_fn

        self._aux_inject = tuple(pad_fn(t, left=left) for t in self._aux_inject)

    def get_banded_aux_mix_weights(self) -> Dict[str, Dict[str, float]]:
        """Return layer-labelled softmax weights for checkpoint diagnostics."""
        if not self.banded_mix_fc:
            return {}
        result: Dict[str, Dict[str, float]] = {}
        for band_idx, band in enumerate(self.aux_layer_bands):
            logits = getattr(self, f"band{band_idx}_mix_logits")
            weights = torch.softmax(logits.detach().float(), dim=0).cpu().tolist()
            result[f"band{band_idx}"] = {
                str(layer_id): weight for layer_id, weight in zip(band, weights)
            }
        return result

    def init_cache_hidden(self):
        """Per-draft-layer KV cache containers for the speculative train loop."""
        return [[[], []] for _ in range(len(self.layers))]

    def encode_layers(
        self,
        inputs_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_cache: bool,
    ):
        # Accept legacy single-layer cache [[ks], [vs]] when num_hidden_layers==1.
        if cache_hidden is None:
            cache_hidden = self.init_cache_hidden()
        elif (
            len(self.layers) == 1
            and len(cache_hidden) == 2
            and isinstance(cache_hidden[0], list)
            and (len(cache_hidden[0]) == 0 or torch.is_tensor(cache_hidden[0][0]))
        ):
            cache_hidden = [cache_hidden]
        elif len(cache_hidden) != len(self.layers):
            raise ValueError(
                f"cache_hidden length {len(cache_hidden)} != "
                f"num draft layers {len(self.layers)}"
            )

        for layer_idx, layer in enumerate(self.layers):
            layer_outputs, cache_hidden[layer_idx] = layer(
                inputs_embeds if layer_idx == 0 else None,
                hidden_states,
                cache_hidden[layer_idx],
                attention_mask,
                position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
        return hidden_states, cache_hidden

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_hidden_states = self.norm(hidden_states)
        logits = self.lm_head(norm_hidden_states)
        return logits.float()

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Remap legacy `midlayer.*` keys to `layers.0.*` for older checkpoints.

        Checkpoints written before the gist-conditioning path was removed carry
        a `layers.*.gist_norm.weight` that no forward ever read. Drop it rather
        than fail a strict load.
        """
        state_dict = {k: v for k, v in state_dict.items() if ".gist_norm." not in k}
        if any(k.startswith("midlayer.") for k in state_dict):
            remapped = {}
            for k, v in state_dict.items():
                if k.startswith("midlayer."):
                    remapped["layers.0." + k[len("midlayer.") :]] = v
                else:
                    remapped[k] = v
            state_dict = remapped
        return super().load_state_dict(state_dict, strict=strict)

    def load_layer_weights_from_target(
        self,
        target_model_name_or_path: str,
        layer_ids: List[int],
        embed_weight_key: Optional[str] = None,
        target_layer_weight_prefix: Optional[str] = None,
    ):
        """
        Initialize draft layers from selected target decoder layers.

        Stock (fused_fc):
          Layer 0 (2H QKV, dual norm):
            - emb / left path ← random (not copied)
            - HS / right path ← ``layer_ids[0]``
          Later layers: full H→H copy from ``layer_ids[i]``.

        """
        if layer_ids is None:
            return
        if len(layer_ids) != len(self.layers):
            raise ValueError(
                f"draft_layer_init_from_target length ({len(layer_ids)}) must equal "
                f"num_hidden_layers ({len(self.layers)})"
            )

        if not os.path.exists(target_model_name_or_path):
            target_model_name_or_path = snapshot_download(repo_id=target_model_name_or_path)

        if target_layer_weight_prefix is None:
            if embed_weight_key is None:
                raise ValueError(
                    "Need embed_weight_key or target_layer_weight_prefix to locate "
                    "target layer weights."
                )
            target_layer_weight_prefix = infer_target_layer_weight_prefix(embed_weight_key)

        def _layer_name_map(target_idx: int) -> Dict[str, str]:
            prefix = f"{target_layer_weight_prefix}.{target_idx}"
            return {
                "self_attn.q_proj.weight": f"{prefix}.self_attn.q_proj.weight",
                "self_attn.k_proj.weight": f"{prefix}.self_attn.k_proj.weight",
                "self_attn.v_proj.weight": f"{prefix}.self_attn.v_proj.weight",
                "self_attn.o_proj.weight": f"{prefix}.self_attn.o_proj.weight",
                "mlp.gate_proj.weight": f"{prefix}.mlp.gate_proj.weight",
                "mlp.up_proj.weight": f"{prefix}.mlp.up_proj.weight",
                "mlp.down_proj.weight": f"{prefix}.mlp.down_proj.weight",
                "input_layernorm.weight": f"{prefix}.input_layernorm.weight",
                "post_attention_layernorm.weight": f"{prefix}.post_attention_layernorm.weight",
            }

        # Collect keys we need, then load once.
        needed: List[str] = []
        plan: List[Tuple[int, int, Dict[str, str]]] = []  # draft_idx, target_idx, name_map
        # L0's left (embedding) half stays random.
        left_src_ids: List[Optional[int]] = []
        for draft_idx, target_idx in enumerate(layer_ids):
            name_map = _layer_name_map(target_idx)
            needed.extend(name_map.values())
            plan.append((draft_idx, target_idx, name_map))
            left_src_ids.append(None if draft_idx == 0 else int(target_idx))

        for left_id in left_src_ids:
            if left_id is not None:
                needed.extend(_layer_name_map(left_id).values())

        tensors = self._load_weight_tensors(target_model_name_or_path, needed)
        missing = [k for k in needed if k not in tensors]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} target layer weight(s), e.g. {missing[0]}. "
                f"Check target_layer_weight_prefix={target_layer_weight_prefix!r}."
            )

        h = self.hidden_size
        with torch.no_grad():
            for draft_idx, target_idx, name_map in plan:
                layer = self.layers[draft_idx]
                src = {k: tensors[v] for k, v in name_map.items()}
                left_id = left_src_ids[draft_idx]
                left_src = (
                    {k: tensors[v] for k, v in _layer_name_map(left_id).items()}
                    if left_id is not None
                    else None
                )

                def _copy(param: torch.nn.Parameter, tensor: torch.Tensor, what: str = ""):
                    if param.shape != tensor.shape:
                        raise ValueError(
                            f"Shape mismatch when init draft layer {draft_idx} "
                            f"({what}) from target layer {target_idx}: "
                            f"{tuple(param.shape)} vs {tuple(tensor.shape)}"
                        )
                    param.copy_(tensor.to(dtype=param.dtype, device=param.device))

                # Residual-stream body always from the mapped HS source layer.
                _copy(layer.mlp.gate_proj.weight, src["mlp.gate_proj.weight"], "mlp.gate")
                _copy(layer.mlp.up_proj.weight, src["mlp.up_proj.weight"], "mlp.up")
                _copy(layer.mlp.down_proj.weight, src["mlp.down_proj.weight"], "mlp.down")
                _copy(
                    layer.post_attention_layernorm.weight,
                    src["post_attention_layernorm.weight"],
                    "post_attn_norm",
                )
                _copy(layer.self_attn.o_proj.weight, src["self_attn.o_proj.weight"], "o_proj")

                use_2h = draft_idx == 0
                if use_2h:
                    # HS / right path ← consumer layer; L0 left stays random.
                    _copy(
                        layer.hidden_norm.weight,
                        src["input_layernorm.weight"],
                        "hidden_norm",
                    )
                    if left_src is not None:
                        _copy(
                            layer.input_layernorm.weight,
                            left_src["input_layernorm.weight"],
                            "left input_layernorm",
                        )
                    for proj_name in ("q_proj", "k_proj", "v_proj"):
                        draft_w = getattr(layer.self_attn, proj_name).weight
                        hs_w = src[f"self_attn.{proj_name}.weight"]
                        if draft_w.shape[-1] != 2 * h or hs_w.shape[-1] != h:
                            raise ValueError(
                                f"Expected draft layer{draft_idx} {proj_name} "
                                f"in_features=2H and target in_features=H; got "
                                f"{draft_w.shape} / {hs_w.shape}"
                            )
                        draft_w[:, h:].copy_(
                            hs_w.to(dtype=draft_w.dtype, device=draft_w.device)
                        )
                        if left_src is not None:
                            emb_w = left_src[f"self_attn.{proj_name}.weight"]
                            if emb_w.shape[-1] != h:
                                raise ValueError(
                                    f"Expected left-source {proj_name} in_features=H; "
                                    f"got {emb_w.shape}"
                                )
                            draft_w[:, :h].copy_(
                                emb_w.to(dtype=draft_w.dtype, device=draft_w.device)
                            )
                else:
                    _copy(layer.input_layernorm.weight, src["input_layernorm.weight"])
                    _copy(layer.self_attn.q_proj.weight, src["self_attn.q_proj.weight"])
                    _copy(layer.self_attn.k_proj.weight, src["self_attn.k_proj.weight"])
                    _copy(layer.self_attn.v_proj.weight, src["self_attn.v_proj.weight"])

        print(
            f"Initialized draft layers from target layers {layer_ids} "
            f"(left sources={left_src_ids}; "
            f"prefix={target_layer_weight_prefix})"
        )

    def forward(
        self,
        hidden_states,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        loss_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # This forward function is not used actually

        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
            hidden_states.requires_grad = True

        if hidden_states.shape[-1] != self.hidden_size:
            hidden_states = self.combine_hidden_states(hidden_states)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )

        attention_mask = self.prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        cache_hidden = self.init_cache_hidden()

        inputs_embeds = self.embed_tokens(input_ids)
        if self.training and self.gradient_checkpointing and not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad = True
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        hidden_states, cache_hidden = self.encode_layers(
            inputs_embeds=inputs_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=bool(use_cache),
        )

        hidden_states_out = self.norm(hidden_states)

        logits = self.lm_head(hidden_states_out)
        logits = logits.float()
        return hidden_states, logits
