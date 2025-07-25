# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
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
# Adapted from vllm/model_executor/models/qwen3_moe.py
# This file is a part of the vllm-ascend project.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_npu
import vllm
import vllm.envs as envs
from torch import nn
from transformers import PretrainedConfig
from vllm.attention import AttentionMetadata
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              get_tp_group, tensor_model_parallel_all_gather)
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (ReplicatedLinear,
                                               RowParallelLinear)
from vllm.config import CacheConfig, VllmConfig
                                               
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_ep_group
from vllm_ascend.ops.fused_moe import AscendFusedMoE

from vllm.model_executor.models.utils import (AutoWeightsLoader, extract_layer_index, maybe_prefix)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeAttention, Qwen3MoeMLP, Qwen3MoeModel, Qwen3MoeSparseMoeBlock
from transformers import PretrainedConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.compilation.decorators import support_torch_compile
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.sampling_metadata import SamplingMetadata
from collections.abc import Iterable
import torch.nn.functional as F

VLLM_ASCEND_ENABLE_FLASHCOMM = 2

def pad(tensor, x):
    length = tensor.size(0)
    pad_size = (x - (length % x)) % x
    if pad_size > 0:
        return F.pad(tensor, (0, 0, 0, pad_size)), pad_size
    return tensor, pad_size

def unpad(tensor, pad_size):
    if pad_size > 0:
        return tensor[:-pad_size, :]
    return tensor

class CustomQwen3MoeAttention(Qwen3MoeAttention):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(hidden_size=hidden_size,
                         num_heads=num_heads,
                         num_kv_heads=num_kv_heads,
                         max_position_embeddings=max_position_embeddings,
                         head_dim=head_dim,
                         rms_norm_eps=rms_norm_eps,
                         qkv_bias=qkv_bias,
                         rope_theta=rope_theta,
                         cache_config=cache_config,
                         quant_config=quant_config,
                         rope_scaling=rope_scaling,
                         prefix=prefix)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.enable_fc = VLLM_ASCEND_ENABLE_FLASHCOMM
        if self.enable_fc == 2:
            self.o_proj = ReplicatedLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )
        else:
            self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                            hidden_size,
                                            bias=False,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.o_proj")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        #print(f"positions.shape = {positions.shape} | q.shape = {q.shape} | k.shape = {k.shape}")
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        #print(f"attn_output --- attn_output.shape = {attn_output.shape}")
        pad_size = 0
        if self.enable_fc == 2:
            # pad input because AllGather requires token_num to be divisible by tp_size
            attn_output, pad_size = pad(attn_output, self.tp_size)
            output = torch.empty(attn_output.shape,
                                 dtype=attn_output.dtype,
                                 device=attn_output.device)
            dist.all_to_all_single(output,
                                   attn_output,
                                   group=get_tp_group().device_group)
            #print(f"all_to_all_single --- attn_output.shape = {attn_output.shape}, output.shape = {output.shape}")
            attn_output = output.reshape(self.tp_size, -1, output.size(-1)) \
                                .transpose(0, 1) \
                                .reshape(-1, output.size(-1)*self.tp_size)
            #print(f"all_to_all_single reshape --- attn_output.shape = {attn_output.shape}, output.shape = {output.shape}")
        #print(f"o_proj --- self.total_num_heads * self.head_dim = {self.total_num_heads * self.head_dim}")
        output, _ = self.o_proj(attn_output)
        if self.enable_fc == 2:
            output = tensor_model_parallel_all_gather(output, 0)
            output = unpad(output, pad_size)        
        return output

class CustomQwen3MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.self_attn = CustomQwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = AscendQwen3MoeSparseMoeBlock(config=config,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.mlp")
        else:
            self.mlp = Qwen3MoeMLP(hidden_size=config.hidden_size,
                                   intermediate_size=config.intermediate_size,
                                   hidden_act=config.hidden_act,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class AscendQwen3MoeSparseMoeBlock(nn.Module):
    
    top_k: int

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")

        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        self.enable_multistream_moe = \
            ascend_config.torchair_graph_config.enable_multistream_moe

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.num_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")

        self.experts = AscendFusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts")

        
        self.top_k = config.num_experts_per_tok

        self.dp_size = get_dp_group().world_size

        self.tp_group = get_tp_group().device_group
        self.tp_rank = get_tp_group().rank_in_group
        self.ep_group = get_ep_group()

        self.params_dtype = torch.get_default_dtype()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        if attn_metadata is None:
            attn_metadata = get_forward_context().attn_metadata
        # when profile runs, force experts to load balanced tokens
        # to avoid high memory consumption on a single rank.
        # TODO: need a better flag to indicate whether in profile run or not.
        if attn_metadata is None:
            # for profile run
            is_prefill = True
            enable_force_load_balance = True
        else:
            # is_prefill = attn_metadata.num_prefills > 0 is_prefill or
            enable_force_load_balance = False
            if hasattr(attn_metadata, 'with_prefill_across_dp'):
                is_prefill = attn_metadata.with_prefill_across_dp

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            is_prefill=is_prefill,
            top_k=self.top_k,
            enable_force_load_balance=enable_force_load_balance,
            shared_experts=None,
        )

        return hidden_states

from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM
class CustomQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
    }


vllm.model_executor.models.qwen3_moe.Qwen3MoeDecoderLayer = CustomQwen3MoeDecoderLayer