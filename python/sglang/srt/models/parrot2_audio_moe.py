# coding=utf-8
# Adapted from
# https://github.com/MonadKai/transformers/tree/v4.55.0-bairong/src/transformers/models/parrot2_audio/modeling_parrot2_audio.py
# Copyright 2025 The bairong-inc team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Parrot2-Audio-Moe model compatible with HuggingFace weights."""
import logging
import math
from functools import lru_cache, partial
from typing import Any, Iterable, List, Optional, Tuple, Type, TypedDict, Dict, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoTokenizer, Qwen3MoeConfig
from transformers.activations import ACT2FN
from transformers.models.parrot2_audio_moe.configuration_parrot2_audio_moe import Parrot2AudioMoeConfig
from transformers.models.parrot_sensevoice.configuration_parrot_sensevoice import ParrotSenseVoiceConfig
from transformers.models.parrot_sensevoice.modeling_parrot_sensevoice import ParrotSenseVoiceEncoder
from transformers.models.parrot2_audio_moe.modeling_parrot2_audio_moe import Parrot2AudioMoeMultiModalProjector

from sglang.srt.hf_transformers_utils import get_processor
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.layers.activation import QuickGELU
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Parrot2AudioMoeForConditionalGeneration(nn.Module):
    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Parrot2AudioMoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        if getattr(self.config, "audio_config", None) is None:
            self.config.audio_config = ParrotSenseVoiceConfig(
                self.config._name_or_path
            )

        if getattr(self.config, "text_config", None) is None:
            self.config.text_config = Qwen3MoeConfig(self.config._name_or_path)

        self.audio_tower_dtype = config.audio_config.torch_dtype
        with set_default_torch_dtype(self.audio_tower_dtype):
            self.audio_tower = ParrotSenseVoiceEncoder(
                config.audio_config,
            )
        self.projector_dtype = config.audio_config.torch_dtype
        with set_default_torch_dtype(self.projector_dtype):
            self.multi_modal_projector = Parrot2AudioMoeMultiModalProjector(config)
        self.language_model_dtype = config.text_config.torch_dtype
        with set_default_torch_dtype(self.language_model_dtype):
            self.language_model = Qwen3MoeForCausalLM(
                config.text_config, quant_config, prefix=add_prefix("model", prefix)
            )

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        # Get all special token IDs for audio
        audio_token_id: int = getattr(
            mm_inputs, "audio_token_id", mm_inputs.im_token_id
        )

        pattern = MultiModalityDataPaddingPatternMultimodalTokens([audio_token_id])
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # Extract audio features from input items
        input_features = torch.cat([item.feature for item in items], dim=0).type(
            self.audio_tower.dtype
        )

        audio_embeds = self.audio_tower(input_features).last_hidden_state
        audio_embeds = self.multi_modal_projector(audio_embeds)

        audio_feature_lens = torch.cat([item.audio_feature_lens for item in items])
        new_embeds = []
        for i, d in zip(audio_feature_lens, audio_embeds):
            new_embeds.append(d[: i.item()])

        return torch.cat(new_embeds, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            data_embedding_funcs={
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            if self.config.text_config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "audio_tower" in name:
                    continue
                name_tmp = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models.
                if name_tmp.endswith(".bias") and name_tmp not in params_dict:
                    continue
                param = params_dict[name_tmp]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                try:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                except KeyError:
                    print(params_dict.keys())
                    raise

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = Parrot2AudioMoeForConditionalGeneration
