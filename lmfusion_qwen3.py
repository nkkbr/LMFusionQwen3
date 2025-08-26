# Copyright 2025 nkkbr

# This work is developed based on the following open-source projects:

# - Qwen3: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
#   Original copyright:
#     Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.

# - Stable Diffusion: https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion
#   Original copyright:
#     Copyright 2025 The HuggingFace Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Callable, Optional, Union, List, Tuple, Any
import PIL.Image
import torch
import torch.nn as nn
from dataclasses import dataclass
from lmfusion_qwen3_config import LMFusionQwen3Config
from transformers.utils import logging, can_return_tuple, ModelOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from unet import LMFusionDownsampler, LMFusionUpsampler
from diffusers.models import AutoencoderKL
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from diffusers import PNDMScheduler
from tqdm.auto import tqdm
import inspect
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.modeling_outputs import AutoencoderKLOutput
import random

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# Copied from https://github.com/huggingface/diffusers/blob/7bc0a07b1947bcbe0e84bbe9ecf8ae2d234382c4/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

# Copied from https://github.com/huggingface/diffusers/blob/50dea89dc6036e71a00bc3d57ac062a80206d9eb/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L86-L96
def retrieve_latents(
    encoder_output: Union[torch.Tensor, AutoencoderKLOutput], 
    generator: Optional[torch.Generator] = None, 
    sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class LMFusionQwen3MLP(nn.Module):
    """
    A modality-specific MLP (Multi-Layer Perceptron) block for the LMFusion model.

    This module contains two separate MLP networks: one specialized for processing
    text representations and another for image representations. It dynamically routes
    the hidden states through the appropriate MLP based on the `modality_ids`
    provided for each token. This allows the model to learn distinct transformations
    for different modalities within a unified architecture.

    The MLP computation is applied token-wise, meaning the output for each token
    depends only on its own input hidden state.

    Args:
        config (LMFusionQwen3Config): The model configuration object.
    """
    def __init__(
        self, 
        config: LMFusionQwen3Config
    ):
        super().__init__()
        self.text_mlp = Qwen3MLP(config)
        self.image_mlp = Qwen3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor, 
        modality_ids: torch.LongTensor,
    ):
        # Fast path for single-modality inputs to avoid unnecessary computation.
        if torch.all(modality_ids == 0):
            return self.text_mlp(hidden_states)
        
        if torch.all(modality_ids == 1):
            return self.image_mlp(hidden_states)
        
        output_text = self.text_mlp(hidden_states)
        output_image = self.image_mlp(hidden_states)

        # test_mask: torch.BoolTensor (batch_size, sequence_length,1)
        text_mask = (modality_ids == 0).unsqueeze(-1)

        if hidden_states.shape[1] == 1: # check if it is the decode phase
            text_mask = text_mask[:,-1:]
        output = torch.where(text_mask, output_text, output_image)

        return output


class LMFusionQwen3Attention(nn.Module):

    def __init__(
        self,
        config: LMFusionQwen3Config,
        layer_idx: int
    ):
        # The design of this constructor is based on the Qwen3Attention implementation.

        # The layer_idx is used to enable layer-specific behaviors, such as applying
        # sliding window attention. For a model like Qwen3-8B, its value would range
        # from 0 to 35.
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj_text = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj_text = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj_text = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj_text = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_proj_image = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj_image = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj_image = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj_image = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)

        self.q_norm_text = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_norm_image = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_text = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_image = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_ids: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache]=None,
        cache_position: Optional[torch.LongTensor]=None,
        **kwargs
    ):
        
        # In the type hints for Qwen3Attention's forward function, there is a line like **kwargs: Unpack[FlashAttentionKwargs],
        # but we are not using FlashAttention.

        # hidden_states
        # Prefill phase: (batch_size, seq_len, hidden_size)
        # Decode phase: (batch_size, 1, hidden_size)

        # modality_ids
        # Regardless of whether it is in the initial generation (prefill phase) or the subsequent generation (decode phase), its shape is always (batch_size, sequence_length).
        # Here, 0 is for text, 1 is for image.
        
        # In this class, we have a QK-Norm (reportedly from this paper https://arxiv.org/abs/2302.05442, needs verification).

        # input_shape: (batch_size, sequence_length)
        input_shape = hidden_states.shape[:-1]

        # hidden_shape: (batch_size, sequence_length,-1 head_dim=128)
        # This shape is used for the subsequent reshape. Because we use GQA (Grouped-Query Attention, https://arxiv.org/abs/2305.13245),
        # the shapes of Q and KV will be different. But here we cleverly use -1 to circumvent this issue.
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        # The QKV projections like self.{q,k,v}_proj_{text,image} use a single matrix to project for multiple heads,
        # which is more efficient. After projection, we use .view() to reshape these tensors to get the multi-headed versions.

        # q_text_proj: (batch_size, sequence_length,num_attention_heads=32, head_dim=128)
        # k_text_proj: (batch_size, sequence_length,num_key_value_heads=8, head_dim=128)
        # v_text_proj: (batch_size, sequence_length,num_key_value_heads=8, head_dim=128)
        q_text_proj = self.q_proj_text(hidden_states).view(hidden_shape)
        k_text_proj = self.k_proj_text(hidden_states).view(hidden_shape)
        v_text_proj = self.v_proj_text(hidden_states).view(hidden_shape)

        q_text_proj_norm = self.q_norm_text(q_text_proj)
        k_text_proj_norm = self.k_norm_text(k_text_proj)

        # q_image_proj: (batch_size, sequence_length,num_attention_heads=32, head_dim=128)
        # k_image_proj: (batch_size, sequence_length,num_key_value_heads=8, head_dim=128)
        # v_image_proj: (batch_size, sequence_length,num_key_value_heads=8, head_dim=128)
        q_image_proj = self.q_proj_image(hidden_states).view(hidden_shape)
        k_image_proj = self.k_proj_image(hidden_states).view(hidden_shape)
        v_image_proj = self.v_proj_image(hidden_states).view(hidden_shape)

        q_image_proj_norm = self.q_norm_image(q_image_proj)
        k_image_proj_norm = self.k_norm_image(k_image_proj)

        text_mask = (modality_ids == 0).unsqueeze(-1).unsqueeze(-1)

        # Decode phase
        # Although it's the same object, it's named past_key_value in LMFusionQwen3Attention and LMFusionQwen3DecoderLayer,
        # but past_key_values in LMFusionQwen3Model and LMFusionQwen3ForCausalLM.
        # During the prefill phase, the list past_key_value.key_cache will continuously grow until it reaches 36,
        # which is the number of layers in Qwen3.
        if hidden_states.shape[1] == 1: # check if it is the decode phase
            text_mask = text_mask[:,-1:]
        
        # .transpose(1, 2) is a standard operation. Since matrix operations like attention act on the last two dimensions
        # of the tensor, transposing allows the matrix operation to be performed on (sequence_length, head_dim).
        # query_states: (batch_size, num_attention_heads=32, sequence_length, head_dim=128)
        # key_states: (batch_size, num_attention_heads=8, sequence_length, head_dim=128)
        # value_states: (batch_size, num_attention_heads=8, sequence_length, head_dim=128)
        query_states = torch.where(text_mask,q_text_proj_norm,q_image_proj_norm).transpose(1, 2)
        key_states = torch.where(text_mask,k_text_proj_norm,k_image_proj_norm).transpose(1, 2)
        value_states = torch.where(text_mask,v_text_proj,v_image_proj).transpose(1, 2)

        # Rotary Position Embedding (RoPE, https://arxiv.org/abs/2104.09864)
        # cos, sin : (batch_size, sequence_length, head_dim=128)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # In the decode phase, update past_key_value. This code is from qwen3.
        # Here, past_key_value is a DynamicCache object (from transformers.cache_utils).
        # For a DynamicCache object, cache_kwargs is actually discarded directly.
        # However, this code is designed to also work with StaticCache, so this parameter is kept.
        # Back to our DynamicCache, it defines three attributes in its __init__ function:
        #   self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        #   self.key_cache: List[torch.Tensor] = []
        #   self.value_cache: List[torch.Tensor] = []

        # Let's look at an example:
        # _seen_tokens: 23
        # len(past_key_value.key_cache): 36
        # torch.Size([2, 8, 23, 128])
        # len(past_key_value.value_cache): 36
        # torch.Size([2, 8, 23, 128])

        # This shows that we have currently seen 23 tokens, which is of course consistent with the 23 in torch.Size([2, 8, 23, 128]).
        # len(past_key_value.key_cache) and len(past_key_value.value_cache) indicate that this large language model has 36 layers.
        # If you print this value for each layer, you will see it gradually increase from 1 to 36 during the prefill phase
        # (because initially, the later layers do not have this value). In the decode phase, this value is always 36.
        # And torch.Size([2, 8, 23, 128]) is the cached K or V for a specific layer, with the shape
        # (batch_size, num_key_value_heads=8, sequence_length, head_dim=128).

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # At this point, we have prepared Q, K, and V, and we can perform the attention calculation. In the paper,
        # these two modalities are represented by two formulas (9 and 10), but the attention is calculated together.
        # They are only separated when using the o_proj projection.
        
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        assert self.config._attn_implementation == 'sdpa', "Attention implementation must be 'sdpa'"
        # To be honest, I haven't figured out when self.config._attn_implementation is assigned the value 'sdpa',
        # but in fact, it is 'sdpa'. We add an assert to ensure this.

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window, # In our current setup, this is always None
            **kwargs,
        )

        # attn_output: (batch_size, sequence_length, num_attenton_heads=32, head_dim=128) -> (batch_size, sequence_length, hidden_dim=4096)
        attn_output = attn_output.reshape(*input_shape,-1).contiguous()
        attn_output_text = self.o_proj_text(attn_output)
        attn_output_image = self.o_proj_image(attn_output)

        text_mask = (modality_ids == 0).unsqueeze(-1)

        # Decode phase
        # Although it's the same object, it's named past_key_value in LMFusionQwen3Attention and LMFusionQwen3DecoderLayer,
        # but past_key_values in LMFusionQwen3Model and LMFusionQwen3ForCausalLM.
        if hidden_states.shape[1] == 1: # check if it is the decode phase
            text_mask = text_mask[:,-1:]
        
        attn_output = torch.where(text_mask, attn_output_text, attn_output_image)
        # attn_output:(batch_size, sequence_length, hidden_dim=4096)
        
        return attn_output, attn_weights
    

class LMFusionQwen3DecoderLayer(GradientCheckpointingLayer):
    """
    This class will instantiate and connect the LMFusionQwen3Attention and LMFusionQwen3MLP modules from above.
    It implements the concepts mentioned in the paper:
    (page 5) The pre-attention layer normalization is also modality-specific and is folded into the QKV functions.
    (page 5) The pre-FFN layer normalization is also modality-specific and is folded in the FFN functions.
    """
    def __init__(
        self,
        config: LMFusionQwen3Config,
        layer_idx: int
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LMFusionQwen3Attention(config=config,layer_idx=layer_idx)
        self.mlp = LMFusionQwen3MLP(config=config)
        self.input_layernorm_text = Qwen3RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_image = Qwen3RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_text = Qwen3RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_image = Qwen3RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]  # Currently, our setting is always full_attention.

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache:Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
        # In the Qwen3 code, the type hint for **kwargs is Unpack[FlashAttentionKwargs], but we are not using FlashAttention here.
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states

        # Pre-calculate the text_mask.
        # It remains constant for the modality-specific pre-attention layer normalization 
        # and modality-specific pre-FFN layer normalization.
        text_mask = (modality_ids == 0).unsqueeze(-1)
        if hidden_states.shape[1] == 1: # check if it is the decode phase
            text_mask = text_mask[:,-1:]

        # modality-specific pre-attention layer normalization
        if torch.all(modality_ids == 0):
            hidden_states = self.input_layernorm_text(hidden_states)
        
        elif torch.all(modality_ids == 1):
            hidden_states = self.input_layernorm_image(hidden_states)
        
        else:
            hidden_states_text = self.input_layernorm_text(hidden_states)
            hidden_states_image = self.input_layernorm_image(hidden_states)
            hidden_states = torch.where(text_mask, hidden_states_text, hidden_states_image)
        
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            modality_ids=modality_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )
        
        hidden_states = residual + hidden_states
        
        residual = hidden_states

        # modality-specific pre-FFN layer normalization
        if torch.all(modality_ids == 0):
            hidden_states = self.post_attention_layernorm_text(hidden_states)
        
        elif torch.all(modality_ids == 1):
            hidden_states = self.post_attention_layernorm_image(hidden_states)
        
        else:
            hidden_states_text = self.post_attention_layernorm_text(hidden_states)
            hidden_states_image = self.post_attention_layernorm_image(hidden_states)
            hidden_states = torch.where(text_mask, hidden_states_text, hidden_states_image)
        
        hidden_states = self.mlp(
            hidden_states=hidden_states,
            modality_ids=modality_ids,
        )
        
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        return outputs


class LMFusionQwen3PreTrainedModel(PreTrainedModel):
    config_class = LMFusionQwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LMFusionQwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True
    # In fact, I don't have a deep understanding of how these variables work.
    # They seem to tell the transformers library which forms of attention and cache our model supports,
    # so it can switch to the most efficient one available.
    # Our model, as of now, does not actually support flash attention, nor have we considered supporting flex attention.
    # From a conservative perspective, one could say it's self-limiting, and we could consider setting:
    # _supports_flash_attn_2 = False
    # _supports_flex_attn = False
    # _supports_attention_backend = False
    # However, the most important thing is whether it can run correctly and accurately when we actually use sdpa.


    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3RMSNorm):
            module.weight.data.fill_(1.0)

def get_image_token_spans(modality_ids: torch.Tensor) -> List[List[List[int]]]:
    """
    Returns the start and end position intervals [start, end) of image tokens (value 1) in each batch, processed row by row.
    """
    results = []
    for row in modality_ids:
        spans = []
        start = None
        for i, token in enumerate(row):
            if token == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    spans.append([start, i])
                    start = None
        if start is not None:
            spans.append([start, len(row)])
        results.append(spans)
    return results

def create_image_bidirectional_mask_function(
    modality_ids: torch.LongTensor
) -> Callable:
    """
    This is a higher-order function (a factory function) that receives a modality_ids tensor,
    and returns a concrete, vmap-compatible mask_function that can be used for or_masks.

    The logic implemented by this function is: to allow an image token to see all other tokens
    within its own contiguous image block.
    
    Args:
        modality_ids (torch.LongTensor): A tensor of shape (batch_size, sequence_length),
                                         where a value of 1 indicates an image token, and 0 indicates a text token.
    
    Returns:
        Callable: A function that follows the (batch_idx, head_idx, q_idx, kv_idx) -> bool signature.
                  The function returns True when q and kv belong to the same image block.
    """
    
    # 1. Find out which tokens are image tokens
    is_image = (modality_ids == 1)

    # 2. Find the starting position of each image block
    #    The start of an image block is where the current token is an image token (True), and the previous one is a text token (False).
    #    We achieve this by comparing is_image with a version of it shifted one position to the right.
    padded_is_image = torch.nn.functional.pad(is_image, (1, 0), value=False)  # Pad with False on the left side of the sequence dimension
    is_start_of_block = is_image & ~padded_is_image[:, :-1]

    # 3. Use cumsum to assign a unique ID to each image block
    #    Perform a cumulative sum on is_start_of_block; the ID increments at the start of each new block.
    block_ids_raw = torch.cumsum(is_start_of_block.long(), dim=1)
    
    # 4. We only care about the group IDs of image tokens; set the group IDs of text tokens to 0.
    #    This way, if a token's group ID is not 0, we know it is an image token.
    image_group_ids = block_ids_raw * is_image.long()

    # image_group_ids is now a tensor of shape (batch_size, sequence_length)
    # For example, if modality_ids = [0, 1, 1, 0, 1, 1, 1]
    # then, image_group_ids = [0, 1, 1, 0, 2, 2, 2]

    def image_bidirectional_mask(
        batch_idx: torch.Tensor, head_idx: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        This inner function contains the actual masking logic and is fully vmap-compatible.
        It checks if q_idx and kv_idx belong to the same image group.
        """
        # Get the group IDs for q and kv from the pre-computed image_group_ids
        q_group_id = image_group_ids[batch_idx, q_idx]
        kv_group_id = image_group_ids[batch_idx, kv_idx]
        
        # Conditions for the additional rule:
        # 1. The query token q must be an image token (its group ID > 0).
        # 2. The group IDs of q and k must be the same (they belong to the same image block).
        #
        # Note: (q_group_id > 0) ensures that we only apply this bidirectional rule for image query tokens.
        # For text query tokens, q_group_id is 0, so this expression will always be False, and they will not
        # gain extra attention permissions.
        return (q_group_id > 0) & (q_group_id == kv_group_id)
        
    return image_bidirectional_mask

################################################################################################
# ----------- How to use our create_image_bidirectional_mask_function -----------
# # In the model's forward method
# # batch_image_groups: List[List[List[int]]] # This is the new parameter you need to prepare and pass into forward
# # ...

# # 1. Create a "batch-aware" custom mask function for the entire batch
# batch_custom_mask_func = create_batch_aware_image_mask_function(batch_image_groups)

# # 2. Pass this function into create_causal_mask
# # The mask_interface (e.g., _create_eager_mask) will iterate or broadcast over all coordinates when generating the 4D mask internally,
# # for each coordinate point (b, h, q, k), it will call the function we passed in.
# # Our function can now correctly look up the corresponding rules based on b (batch_idx).
# final_attention_mask = create_causal_mask(
#     config=self.config,
#     input_embeds=inputs_embeds,
#     attention_mask=attention_mask,
#     cache_position=cache_position,
#     past_key_values=past_key_values,
#     or_mask_function=batch_custom_mask_func, # Pass in this function that can handle batches
# )

# # final_attention_mask is now the 4D tensor we want, where each sample has different masking rules.
################################################################################################

class LMFusionQwen3Model(LMFusionQwen3PreTrainedModel):

    def __init__(
        self,
        config: LMFusionQwen3Config
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LMFusionQwen3DecoderLayer(config=config,layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        vision_config = getattr(config, "vision_config", {})
        self.unet_downsampler = LMFusionDownsampler(
            in_channels=vision_config.get("in_channels",4), 
            temb_channels=vision_config.get("temb_channels",1280), 
            hidden_size=vision_config.get("hidden_size",4096)
        )
        self.unet_upsampler = LMFusionUpsampler(
            out_channels=vision_config.get("out_channels",4), 
            temb_channels=vision_config.get("temb_channels",1280), 
            hidden_size=vision_config.get("hidden_size",4096)
        )
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds:Optional[torch.FloatTensor] = None,
        modality_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")
        
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # An example of cache_position and position_ids:
        # prefill phase
        # cache_position: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        #         18, 19, 20, 21], device='cuda:2')
        # position_ids: tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        #         18, 19, 20, 21],
        #         [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  1,  1,
        #         1,  1,  1,  1]], device='cuda:2')
        # decode phase
        # cache_position: tensor([22], device='cuda:2')
        # position_ids: tensor([[22],
        #         [16]], device='cuda:2')

        # The following causal_mask_mapping is a very important step in our reproduction work.

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):

            custom_mask_func = None
            if modality_ids is not None:
                custom_mask_func = create_image_bidirectional_mask_function(modality_ids)
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "or_mask_function": custom_mask_func, 
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                modality_ids=modality_ids,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

################################################################################################
# ----------- Regarding our special attention mask -----------
# Our code requires a custom mask.
# The mask is a dict, its key is the attention type, e.g., "full_attention" or "sliding_attention".
# Currently, in our research, we uniformly use "full_attention".
# The value of the mask is a torch tensor.
# In the prefill phase, its shape is (B, 1, N, N).
# In the decode phase, its shape is (B, 1, 1, N).
# We also need to be aware that when B > 1, the sequence_length of different batches can be different,
# so outside of a square in the top-left corner, it might be all 0s (False).
################################################################################################


@dataclass
class LMFusionCausalLMOutput(CausalLMOutputWithPast):
    """
    The output class for the LMFusion model, which extends the standard CausalLMOutput with support for predicted noise.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None # Logits for the text part
    predicted_noise: Optional[torch.FloatTensor] = None # Predicted noise for the image part
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class LMFusionQwen3ForCausalLM(LMFusionQwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(
        self,
        config:LMFusionQwen3Config
    ):
        super().__init__(config)
        self.model = LMFusionQwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        vision_config =getattr(config, 'vision_config', {})
        time_embed_dim = vision_config.get('temb_channels', 1280)
        self.time_proj = Timesteps(num_channels=320, flip_sin_to_cos=True, downscale_freq_shift=0) # The numbers here come from the model = "CompVis/stable-diffusion-v1-4"
        self.time_embedding = TimestepEmbedding(in_channels=320, time_embed_dim=time_embed_dim)

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        # We encapsulate the tokenizer within our class. Although generally, large language models
        # in the transformer ecosystem don't do this. But our model is special, our four inference functions
        # directly handle text and images, and we have only encapsulated it in this class
        # without creating a dedicated pipeline.
        
        # The vae_scale_factor might need to be determined by code like the following:
        # self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # But here we directly provide the answer, which is 8.
        self.image_processor = VaeImageProcessor(vae_scale_factor=8)
        self.scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder='scheduler')
        self.do_classifier_free_guidance = True

        # Store some input_ids for concatenation, to be computed after the tokenizer is loaded.
        self.prefix_ids = None
        self.suffix_ids = None
        self.boi_ids = None
        self.eoi_ids = None
        self.eos_ids = None
        self.pad_ids = None

        self.post_init()
        
        if hasattr(config, "vae_model_name_or_path"):
            self.build_vae(
                vae_model_name_or_path=config.vae_model_name_or_path,
                delay_load=True
            )
    def _precompute_special_token_ids(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been loaded. Cannot initialize special token IDs.")

        # Store some input_ids for concatenation.
        prefix = "<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_ids = self.tokenizer.encode(prefix, return_tensors='pt') # tensor([[151644,    872,    198]])
        self.suffix_ids = self.tokenizer.encode(suffix, return_tensors='pt') # tensor([[151645,    198, 151644,  77091,    198, 151667,    271, 151668,    271]])
        self.boi_ids = self.tokenizer.encode("<BOI>", return_tensors='pt') # tensor([[151669]])
        self.eoi_ids = self.tokenizer.encode("<EOI>", return_tensors='pt') # tensor([[151670]])
        self.eos_ids = self.tokenizer.encode("<|im_end|>" , return_tensors='pt') # tensor([[151645]])
        self.pad_ids = self.tokenizer.encode('<|endoftext|>', return_tensors='pt') # tensor([[151643]])

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        
         # We override this method mainly to load the tokenizer.
        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
            model.tokenizer = tokenizer
        except OSError:
            logger.warning("Tokenizer not found. You will need to load it manually.")
        
        # After loading the tokenizer, execute precompute.
        model._precompute_special_token_ids()
        
        return model

    def build_vae(
            self, 
            vae_model_name_or_path,
            delay_load=False
        ):
        if not delay_load:
            logger.info(f'Loading {vae_model_name_or_path}')
            self.vae = AutoencoderKL.from_pretrained(vae_model_name_or_path)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def _pretrain_or_finetune(
        self,
        modality_ids:torch.LongTensor
    ):
        if modality_ids[0].sum().item() == 256.:
            return "PRETRAIN_PHASE"
        elif modality_ids[0].sum().item() == 512.:
            return "FINETUNE_PHASE"
        raise ValueError("Could not determine task from the provided inputs.")
    
    def get_diffusion_loss(
        self,
        hidden_states:torch.Tensor,
        target_noise:torch.Tensor,
        image_res_samples:torch.Tensor,
        image_temb:torch.Tensor,
        modality_ids:torch.Tensor,
        pretrain_or_finetune,
        modality_mask:Optional[torch.Tensor]=None
    ):
        
        if pretrain_or_finetune == "PRETRAIN_PHASE":
            mask = modality_ids.bool()
        elif pretrain_or_finetune == "FINETUNE_PHASE":
            mask = modality_mask.bool()
        else:
            raise ValueError(f'pretrain_or_finetune should be "PRETRAIN_PHASE" or "FINETUNE_PHASE" but {pretrain_or_finetune}')
        
        B,_,D = hidden_states.shape

        # selected_hidden_states (B, 256, 4096)
        selected_hidden_states = hidden_states[mask].reshape(B, -1, D)
        image_hidden_states = selected_hidden_states.permute(0, 2, 1).reshape(-1, 4096, 16, 16)

        # noise_pred (B,4,32,32)
        noise_pred = self.model.unet_upsampler(
            hidden_states=image_hidden_states,
            res_samples=image_res_samples,
            temb=image_temb
        )

        return nn.MSELoss()(noise_pred,target_noise)
    
    def get_timesteps_for_finetune(
        self,
        batch_size,
        device,
        timestep_sampling_strategy
    ):
        if timestep_sampling_strategy == 'uniform':
            return torch.randint(
                    0,
                    self.scheduler.num_train_timesteps,
                    (batch_size,),
                    device=device,
                    dtype=torch.long
                )
        elif timestep_sampling_strategy == 'cosine':
            # To be implemented later.
            pass
        else:
            raise ValueError(f'timestep_sampling_strategy should be "uniform" or "cosine" but {timestep_sampling_strategy}')

    def get_model_inputs_for_image_first(
        self,
        prefix_embeds:torch.Tensor,
        boi_embeds:torch.Tensor,
        eoi_embeds:torch.Tensor,
        suffix_embeds:torch.Tensor,
        eos_embeds:torch.Tensor,
        pad_embeds:torch.Tensor,
        input_ids_list:List[torch.Tensor], # The items in the list are tensors of shape [batch_size=1, seq_len], indicating the valid input_ids for each sample.
        sample_embed:torch.Tensor,
    ):
        device = sample_embed.device

        seqlen = 0
        for item in input_ids_list:
            seqlen = max(seqlen, item.shape[1])

        batch_size = sample_embed.shape[0]

        inputs_embeds_list = [[] for _ in range(batch_size)]
        attention_mask_list = [[] for _ in range(batch_size)]
        modality_ids_list = [[] for _ in range(batch_size)]
        labels_list = [[] for _ in range(batch_size)]

        for idx in range(batch_size):

            # 1. Add prefix
            inputs_embeds_list[idx].append(prefix_embeds)
            attention_mask_list[idx].append(torch.ones(1,3, device=device))
            modality_ids_list[idx].append(torch.zeros(1,3, device=device))
            labels_list[idx].append(self.prefix_ids.to(device))

            # 2. Add <BOI>
            inputs_embeds_list[idx].append(boi_embeds)
            attention_mask_list[idx].append(torch.ones(1,1, device=device))
            modality_ids_list[idx].append(torch.zeros(1,1, device=device))
            labels_list[idx].append(self.boi_ids.to(device))

            # 3. Add image embedding
            inputs_embeds_list[idx].append(sample_embed[idx:idx+1])
            attention_mask_list[idx].append(torch.ones(1,256, device=device))
            modality_ids_list[idx].append(torch.ones(1,256, device=device))
            labels_list[idx].append(torch.full((1, 256), -100, device=device))

            # 4. Add <EOI>
            inputs_embeds_list[idx].append(eoi_embeds)
            attention_mask_list[idx].append(torch.ones(1,1, device=device))
            modality_ids_list[idx].append(torch.zeros(1,1, device=device))
            labels_list[idx].append(torch.full((1, 1), -100, device=device))

            # 5. Add suffix
            inputs_embeds_list[idx].append(suffix_embeds)
            attention_mask_list[idx].append(torch.ones(1,9, device=device))
            modality_ids_list[idx].append(torch.zeros(1,9, device=device))
            labels_list[idx].append(self.suffix_ids.to(device))

            # 6. Add the final text, eos, and padding
            current_text_embedding = self.model.embed_tokens(input_ids_list[idx])
            padding_length = seqlen-input_ids_list[idx].shape[1]

            inputs_embeds_list[idx].append(current_text_embedding)
            inputs_embeds_list[idx].append(eos_embeds)
            inputs_embeds_list[idx].append(pad_embeds.repeat(1, padding_length,1))

            labels_list[idx].append(input_ids_list[idx])
            labels_list[idx].append(self.eos_ids.to(device))
            labels_list[idx].append(torch.full((1, padding_length), -100, device=device))

            attention_mask_list[idx].append(torch.ones(1,input_ids_list[idx].shape[1]+1, device=device))
            attention_mask_list[idx].append(torch.zeros(1,padding_length, device=device))

            modality_ids_list[idx].append(torch.zeros(1,seqlen+1, device=device))

        # 7. Merge 1
        inputs_embeds_list = [torch.cat(item, dim=1) for item in inputs_embeds_list]
        attention_mask_list = [torch.cat(item, dim=1) for item in attention_mask_list]
        modality_ids_list = [torch.cat(item, dim=1) for item in modality_ids_list]
        labels_list = [torch.cat(item, dim=1) for item in labels_list]

        # 8. Merge 2
        inputs_embeds = torch.cat(inputs_embeds_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        modality_ids = torch.cat(modality_ids_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        return inputs_embeds, attention_mask, position_ids, modality_ids, labels

    def get_model_inputs_for_text_first(
        self,
        prefix_embeds:torch.Tensor,
        boi_embeds:torch.Tensor,
        eoi_embeds:torch.Tensor,
        suffix_embeds:torch.Tensor,
        eos_embeds:torch.Tensor,
        pad_embeds:torch.Tensor,
        input_ids_list:List[torch.Tensor], # The items in the list are tensors of shape [batch_size=1, seq_len], indicating the valid input_ids for each sample.
        sample_embed:torch.Tensor,
    ):
        device = sample_embed.device

        seqlen = 0
        for item in input_ids_list:
            seqlen = max(seqlen, item.shape[1])

        batch_size = sample_embed.shape[0]

        inputs_embeds_list = [[] for _ in range(batch_size)]
        attention_mask_list = [[] for _ in range(batch_size)]
        modality_ids_list = [[] for _ in range(batch_size)]
        labels_list = [[] for _ in range(batch_size)]

        for idx in range(batch_size):
            
            # 1. Add prefix
            inputs_embeds_list[idx].append(prefix_embeds)
            attention_mask_list[idx].append(torch.ones(1,3, device=device))
            modality_ids_list[idx].append(torch.zeros(1,3, device=device))
            labels_list[idx].append(self.prefix_ids.to(device))

            # 2. Add text, suffix, pad, and <BOI>
            current_text_embedding = self.model.embed_tokens(input_ids_list[idx])
            padding_length = seqlen-input_ids_list[idx].shape[1]

            inputs_embeds_list[idx].append(current_text_embedding)
            inputs_embeds_list[idx].append(suffix_embeds)
            inputs_embeds_list[idx].append(pad_embeds.repeat(1, padding_length,1))
            inputs_embeds_list[idx].append(boi_embeds)

            attention_mask_list[idx].append(torch.ones(1,input_ids_list[idx].shape[1], device=device))
            attention_mask_list[idx].append(torch.ones(1,suffix_embeds.shape[1], device=device))
            attention_mask_list[idx].append(torch.zeros(1,padding_length, device=device))
            attention_mask_list[idx].append(torch.ones(1,1, device=device))

            modality_ids_list[idx].append(torch.zeros(1,seqlen+1+9, device=device)) # seqlen is for text and padding, 1 is for <BOI>, 9 is for suffix

            labels_list[idx].append(input_ids_list[idx])
            labels_list[idx].append(self.suffix_ids.to(device))
            labels_list[idx].append(torch.full((1, padding_length), -100, device=device))
            labels_list[idx].append(self.boi_ids.to(device))

            # 3. Add image embedding
            inputs_embeds_list[idx].append(sample_embed[idx:idx+1])
            attention_mask_list[idx].append(torch.ones(1,256, device=device))
            modality_ids_list[idx].append(torch.ones(1,256, device=device))
            labels_list[idx].append(torch.full((1, 256), -100, device=device))

            # 4. Add <EOI>
            inputs_embeds_list[idx].append(eoi_embeds)
            attention_mask_list[idx].append(torch.ones(1,1, device=device))
            modality_ids_list[idx].append(torch.zeros(1,1, device=device))
            labels_list[idx].append(torch.full((1, 1), -100, device=device))

             # 5. Add eos
            inputs_embeds_list[idx].append(eos_embeds)
            attention_mask_list[idx].append(torch.ones(1,1, device=device))
            modality_ids_list[idx].append(torch.zeros(1,1, device=device))
            labels_list[idx].append(torch.full((1, 1), -100, device=device))

        # 6. Merge 1
        inputs_embeds_list = [torch.cat(item, dim=1) for item in inputs_embeds_list]
        attention_mask_list = [torch.cat(item, dim=1) for item in attention_mask_list]
        modality_ids_list = [torch.cat(item, dim=1) for item in modality_ids_list]
        labels_list = [torch.cat(item, dim=1) for item in labels_list]

        # 7. Merge 2
        inputs_embeds = torch.cat(inputs_embeds_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        modality_ids = torch.cat(modality_ids_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        # 8. Calculate position_ids
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        return inputs_embeds, attention_mask, position_ids, modality_ids, labels

    def get_model_inputs_for_finetune(
        self,
        prefix_embeds:torch.Tensor,
        boi_embeds:torch.Tensor,
        eoi_embeds:torch.Tensor,
        suffix_embeds:torch.Tensor,
        eos_embeds:torch.Tensor,
        pad_embeds:torch.Tensor,
        input_ids_list:List[torch.Tensor],
        sample_for_source_image_embed:torch.Tensor,
        sample_for_target_image_embed:torch.Tensor,
    ):
        device = sample_for_source_image_embed.device

        seqlen = 0
        for item in input_ids_list:
            seqlen = max(seqlen, item.shape[1])

        batch_size = sample_for_source_image_embed.shape[0]

        inputs_embeds_list = [[] for _ in range(batch_size)]
        attention_mask_list = [[] for _ in range(batch_size)]
        modality_ids_list = [[] for _ in range(batch_size)]
        modality_mask_list = [[] for _ in range(batch_size)]
        labels_list = [[] for _ in range(batch_size)]

        for idx in range(batch_size):

            # 1. Add prefix
            inputs_embeds_list[idx].append(prefix_embeds)
            attention_mask_list[idx].append(torch.ones(1,3, device=device))
            modality_ids_list[idx].append(torch.zeros(1,3, device=device))
            modality_mask_list[idx].append(torch.zeros(1,3, device=device))
            labels_list[idx].append(self.prefix_ids.to(device))

            # 2. Add <BOI>
            inputs_embeds_list[idx].append(boi_embeds)
            attention_mask_list[idx].append(torch.ones(1,1, device=device))
            modality_ids_list[idx].append(torch.zeros(1,1, device=device))
            modality_mask_list[idx].append(torch.zeros(1,1, device=device))
            labels_list[idx].append(self.boi_ids.to(device))
            
            # 3. Add source_image's embedding
            inputs_embeds_list[idx].append(sample_for_source_image_embed[idx:idx+1])
            attention_mask_list[idx].append(torch.ones(1,256, device=device))
            modality_ids_list[idx].append(torch.ones(1,256, device=device))
            modality_mask_list[idx].append(torch.zeros(1,256, device=device)) # This is the key point, the only difference between modality_ids_list and modality_mask_list is here.
            labels_list[idx].append(torch.full((1, 256), -100, device=device))

            # 4. Add <EOI>
            inputs_embeds_list[idx].append(eoi_embeds)
            attention_mask_list[idx].append(torch.ones(1,1, device=device))
            modality_ids_list[idx].append(torch.zeros(1,1, device=device))
            modality_mask_list[idx].append(torch.zeros(1,1, device=device))
            labels_list[idx].append(torch.full((1, 1), -100, device=device))

            # 5. Add text, suffix, padding
            current_text_embedding = self.model.embed_tokens(input_ids_list[idx])
            padding_length = seqlen-input_ids_list[idx].shape[1]

            inputs_embeds_list[idx].append(current_text_embedding)
            inputs_embeds_list[idx].append(suffix_embeds)
            inputs_embeds_list[idx].append(pad_embeds.repeat(1, padding_length,1))

            labels_list[idx].append(input_ids_list[idx])
            labels_list[idx].append(self.suffix_ids.to(device))
            labels_list[idx].append(torch.full((1, padding_length), -100, device=device))

            attention_mask_list[idx].append(torch.ones(1,input_ids_list[idx].shape[1]+9, device=device))
            attention_mask_list[idx].append(torch.zeros(1,padding_length, device=device))

            modality_ids_list[idx].append(torch.zeros(1,seqlen+9, device=device))
            modality_mask_list[idx].append(torch.zeros(1,seqlen+9, device=device))

            # 6. Add <BOI>
            inputs_embeds_list[idx].append(boi_embeds)
            attention_mask_list[idx].append(torch.ones(1,1, device=device))
            modality_ids_list[idx].append(torch.zeros(1,1, device=device))
            modality_mask_list[idx].append(torch.zeros(1,1, device=device))
            labels_list[idx].append(self.boi_ids.to(device))

            # 7. Add target_image's embedding
            inputs_embeds_list[idx].append(sample_for_target_image_embed[idx:idx+1])
            attention_mask_list[idx].append(torch.ones(1,256, device=device))
            modality_ids_list[idx].append(torch.ones(1,256, device=device))
            modality_mask_list[idx].append(torch.ones(1,256, device=device))
            labels_list[idx].append(torch.full((1, 256), -100, device=device))

            # 8. Add <EOI>
            inputs_embeds_list[idx].append(eoi_embeds)
            attention_mask_list[idx].append(torch.ones(1,1, device=device))
            modality_ids_list[idx].append(torch.zeros(1,1, device=device))
            modality_mask_list[idx].append(torch.zeros(1,1, device=device))
            labels_list[idx].append(torch.full((1, 1), -100, device=device))

            # 9. Add eos
            inputs_embeds_list[idx].append(eos_embeds)
            attention_mask_list[idx].append(torch.ones(1,1, device=device))
            modality_ids_list[idx].append(torch.zeros(1,1, device=device))
            modality_mask_list[idx].append(torch.zeros(1,1, device=device))
            labels_list[idx].append(torch.full((1, 1), -100, device=device))

        # 10. Merge 1
        inputs_embeds_list = [torch.cat(item, dim=1) for item in inputs_embeds_list]
        attention_mask_list = [torch.cat(item, dim=1) for item in attention_mask_list]
        modality_ids_list = [torch.cat(item, dim=1) for item in modality_ids_list]
        modality_mask_list = [torch.cat(item, dim=1) for item in modality_mask_list]
        labels_list = [torch.cat(item, dim=1) for item in labels_list]

        # 11. Merge 2
        inputs_embeds = torch.cat(inputs_embeds_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        modality_ids = torch.cat(modality_ids_list, dim=0)
        modality_mask = torch.cat(modality_mask_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        return inputs_embeds, attention_mask, position_ids, modality_ids, labels, modality_mask

    # Our forward method, during PRETRAIN_PHASE/FINETUNE_PHASE training, is mainly to return the loss. These two losses,
    # are different. The former is a combination of text CE and noise MSE (lambda=5), the latter is only the MSE between predicted noise and real noise.
    # 
    # During inference:
    # 
    # Pure text generation (Task 1) forward method needs to return logits and past_key_values.
    # Image-to-text (Task 3) forward method needs to return logits and past_key_values.
    # 
    # Text-to-image (Task 2) forward method needs to return last_hidden_state (i.e., hidden_states), in fact we only need the hidden_states of image tokens.
    # Image editing (Task 4) forward method needs to return last_hidden_state (i.e., hidden_states), in fact we only need the hidden_states of image tokens.
    # 
    # However, for the two training phases and the four inference tasks, the return of our forward function is the same.
    # We return everything that is needed. Let the specific inference function choose for itself.
    
    # In addition, the forward method has many parameters. In fact, in general large language models, they are not prepared by ourselves.
    # For example, during inference, they are prepared by GenerationMixin's prepare_inputs_for_generation function.
    # During training, they are prepared by DataCollator.

    @can_return_tuple
    def forward(
        self,

        # ------------ Core Input Parameters ------------
        input_ids: Optional[torch.LongTensor] = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,

        # ------------ Standard HF Parameters ------------
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,

        # ------------ Direct Embedding Inputs ------------
        inputs_embeds: Optional[torch.FloatTensor] = None,

        # ------------ Modality Identifiers ------------
        modality_ids: Optional[torch.LongTensor] = None, 
        # modality_mask is used to mark which tokens require image loss calculation.
        # During pretraining, it is equal to modality_ids, with 256 ones per row. During finetuning, modality_ids has
        # two blocks of 256 ones per row, where the first block of 256 is set to 0.
        # We place the process of calculating modality_mask from modality_ids in the Dataset.
        # modality_mask is not passed during pretraining; it is None. It is pre-calculated only for finetuning.

        # ------------ Inference Optimization ------------
        logits_to_keep: Union[int, torch.Tensor] = 0,

        # ------------ For Training Only ------------
        # From the DataLoader, will be converted into the required format in the `if self.training:` branch.
        input_ids_list: List[torch.Tensor]=None,

        # During pretrain phase
        clean_latents: torch.FloatTensor=None,

        # During finetune phase
        clean_latents_for_source_image: torch.FloatTensor=None,
        clean_latents_for_target_image: torch.FloatTensor=None,

        **kwargs,
    ) -> LMFusionCausalLMOutput:
        
        # The input_ids passed by inference functions are already packaged and ready to use.
        # During training, input_ids is not passed; input_ids_list is passed instead.
        training_phase = getattr(self.config, 'training_phase', None)

        if self.training:
            if training_phase == "pretrain":
                # Task
                # (1) Text needs to be converted to embeddings
                # (2) Image needs to be processed
                # (3) Text and image data are combined into inputs_embeds
                # Inputs
                # input_ids_list
                # clean_latents

                batch_size = clean_latents.shape[0]
                device = clean_latents.device

                # Generate random timesteps for adding noise.
                timesteps = torch.randint(
                    0,
                    self.scheduler.num_train_timesteps,
                    (batch_size,),
                    device=device,
                    dtype=torch.long
                )

                 # Generate time embeddings.
                t_emb = self.get_time_embed(sample=clean_latents, timestep=0)  # This is just to get the time embedding; it is independent of whether noise has been added to clean_latents.
                emb = self.time_embedding(t_emb)

                # Generate random noise.
                noise = torch.randn(
                    clean_latents.shape,
                    device=device,
                    dtype=clean_latents.dtype
                )

                # Add random noise to the image based on the timestep.
                latents = self.scheduler.add_noise(clean_latents, noise, timesteps)

                sample, res_sample = self.model.unet_downsampler(
                    hidden_states=latents,
                    temb=emb
                )

                # sample:[B,4096,16,16] -> sample_embed: [B, 256, 4096] obtained the embeddings for the image part.
                sample_embed = sample.reshape(*sample.shape[:-2],-1).permute(0,2,1)

                # Randomly decide whether the image or the text comes first.
                is_image_first = random.random() < 0.8

                # Prepare some fixed embeddings.
                prefix_embeds = self.model.embed_tokens(self.prefix_ids.to(device))
                boi_embeds = self.model.embed_tokens(self.boi_ids.to(device))
                eoi_embeds = self.model.embed_tokens(self.eoi_ids.to(device))
                suffix_embeds = self.model.embed_tokens(self.suffix_ids.to(device))
                eos_embeds = self.model.embed_tokens(self.eos_ids.to(device))
                pad_embeds = self.model.embed_tokens(self.pad_ids.to(device))

                if is_image_first:
                    # Case where image comes first, text comes second.
                    inputs_embeds, attention_mask, position_ids, modality_ids,labels = self.get_model_inputs_for_image_first(
                        prefix_embeds=prefix_embeds,
                        boi_embeds=boi_embeds,
                        eoi_embeds=eoi_embeds,
                        suffix_embeds=suffix_embeds,
                        eos_embeds=eos_embeds,
                        pad_embeds=pad_embeds,
                        input_ids_list=input_ids_list,
                        sample_embed=sample_embed
                    )
                else:
                    # Case where text comes first, image comes second.
                    inputs_embeds, attention_mask, position_ids, modality_ids,labels = self.get_model_inputs_for_text_first(
                        prefix_embeds=prefix_embeds,
                        boi_embeds=boi_embeds,
                        eoi_embeds=eoi_embeds,
                        suffix_embeds=suffix_embeds,
                        eos_embeds=eos_embeds,
                        pad_embeds=pad_embeds,
                        input_ids_list=input_ids_list,
                        sample_embed=sample_embed
                    )

            elif training_phase == "finetune":
                # Inputs
                # input_ids_list
                # clean_latents_for_source_image (Batch_size, 4, 32, 32)
                # clean_latents_for_target_image (Batch_size, 4, 32, 32)
                # model.config.timestep_sampling_strategy

                batch_size = clean_latents_for_source_image.shape[0]
                device = clean_latents.device

                # Generate random timesteps for adding noise. The sampling method can be adjusted. 
                # (The closer to 0, the higher the probability of being sampled, which may be more conducive to generation).
                timesteps = self.get_timesteps_for_finetune(
                    batch_size=batch_size,
                    device=device,
                    timestep_sampling_strategy = getattr(self.model.config, "timestep_sampling_strategy", "uniform")
                )

                # Generate time embeddings.
                t_emb = self.get_time_embed(sample=clean_latents, timestep=0) # This is just to get the time embedding; it is independent of whether noise has been added to clean_latents.
                emb = self.time_embedding(t_emb)

                # Generate random noise.
                noise = torch.randn(
                    clean_latents.shape,
                    device=device,
                    dtype=clean_latents.dtype
                )

                # Add random noise to the image based on the timestep.
                latents_for_target_image = self.scheduler.add_noise(clean_latents_for_target_image, noise, timesteps)

                # The noisy target_image passes through the U-Net's downsampler to get a residual, which needs to be passed
                # to the upsampler and then used for loss calculation, so it must be kept.
                # In contrast, for the source_image.
                sample_for_target_image, res_sample = self.model.unet_downsampler(
                    hidden_states=latents_for_target_image,
                    temb=emb
                )

                sample_for_source_image, _ = self.model.unet_downsampler(
                    hidden_states=clean_latents_for_source_image,
                    temb=emb
                )

                sample_for_target_image_embed = sample_for_target_image.reshape(*sample_for_target_image.shape[:-2],-1).permute(0,2,1)
                sample_for_source_image_embed = sample_for_source_image.reshape(*sample_for_source_image.shape[:-2],-1).permute(0,2,1)

                # Prepare some fixed embeddings.
                prefix_embeds = self.model.embed_tokens(self.prefix_ids.to(device))
                boi_embeds = self.model.embed_tokens(self.boi_ids.to(device))
                eoi_embeds = self.model.embed_tokens(self.eoi_ids.to(device))
                suffix_embeds = self.model.embed_tokens(self.suffix_ids.to(device))
                eos_embeds = self.model.embed_tokens(self.eos_ids.to(device))
                pad_embeds = self.model.embed_tokens(self.pad_ids.to(device))

                # Similar to the function during pretraining, but here we will return a modality_mask. 
                # In modality_ids, there are two consecutive blocks of 256 ones, and modality_mask will set the first block entirely to 0.
                inputs_embeds, attention_mask, position_ids, modality_ids,labels, modality_mask = self.get_model_inputs_for_finetune(
                        prefix_embeds=prefix_embeds,
                        boi_embeds=boi_embeds,
                        eoi_embeds=eoi_embeds,
                        suffix_embeds=suffix_embeds,
                        eos_embeds=eos_embeds,
                        pad_embeds=pad_embeds,
                        input_ids_list=input_ids_list,
                        sample_for_source_image_embed=sample_for_source_image_embed,
                        sample_for_target_image_embed=sample_for_target_image_embed
                    )
            else:
                raise ValueError(f'training_phase should be "pretrain" or "finetune" but {training_phase}')

            target_noise = noise            # shape: [B, C_latent, H_latent, W_latent], the actual added noise
            image_res_samples = res_sample       # Skip-connection features from the Downsampler
            image_temb = emb             # Time embeddings
            # modality_mask = None # Provided only during "finetune"
        
        # We are a research project, and we directly pass in the image and process it into source_image_pixels.
        # In fact, it might be more efficient to process the images with a VAE beforehand and save them as data files.

        # The parameters passed into this forward method are pre-processed either in (1) the Dataset during training
        # or (2) in the four inference functions during inference. They are used directly in this forward method.
        
        # The code for calculating logits and hidden_states is the same for all tasks.

        # The code for calculating the loss is different for "PRETRAIN_PHASE" and "FINETUNE_PHASE".
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            modality_ids=modality_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])


        # Now it's time to calculate the loss.
        # In Qwen3, to determine whether it is training or inference, it uses:
        # if labels is not None:
        # This line of code can still calculate the loss during training but while in eval mode.
        # Our project does not perform evaluation during the training process, so we took a "shortcut" and directly use self.training to determine if we are training.
        loss = None
        if labels is not None:
            # Training is divided into two phases.
            if training_phase == "pretrain":
                ce_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
                mse_loss = self.get_diffusion_loss(
                    hidden_states = hidden_states,
                    target_noise=target_noise,
                    image_res_samples=image_res_samples,
                    image_temb=image_temb,
                    modality_ids=modality_ids,
                    pretrain_or_finetune="PRETRAIN_PHASE"
                )
                loss = ce_loss + self.config.loss_lambda * mse_loss
            elif training_phase == "finetune":
                mse_loss = self.get_diffusion_loss(
                    hidden_states = hidden_states,
                    target_noise=target_noise,
                    image_res_samples=image_res_samples,
                    image_temb=image_temb,
                    modality_ids=modality_ids,
                    modality_mask=modality_mask,
                    pretrain_or_finetune="FINETUNE_PHASE"
                )
                loss = mse_loss
            else:
                raise ValueError(f'training_phase should be "pretrain" or "finetune" but {training_phase}')

        return LMFusionCausalLMOutput(
            loss=loss,
            logits=logits,
            predicted_noise=None, # TODO
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )
        
        modality_id_to_add = torch.zeros(
            (model_kwargs["modality_ids"].shape[0],1),
            device=model_kwargs["modality_ids"].device,
            dtype=model_kwargs["modality_ids"].dtype
        )
        model_kwargs["modality_ids"] = torch.cat([model_kwargs["modality_ids"],modality_id_to_add],dim=1)

        return model_kwargs


    # Some external utility tools
    # from https://github.com/huggingface/diffusers/blob/7392c8ff5a2a3ea2b059a9e1fdc5164759844976/src/diffusers/models/unets/unet_2d_condition.py#L907-L932
    def get_time_embed(
        self,
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int]
    ) -> Optional[torch.Tensor]:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            is_npu = sample.device.type == "npu"
            if isinstance(timestep, float):
                dtype = torch.float32 if (is_mps or is_npu) else torch.float64
            else:
                dtype = torch.int32 if (is_mps or is_npu) else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)
        return t_emb

    # Some utility functions used by the four inference functions

    # From https://github.com/huggingface/diffusers/blob/50dea89dc6036e71a00bc3d57ac062a80206d9eb/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L724-L733
    # Used to return based on the specified strength
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)


        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)


        return timesteps, num_inference_steps - t_start

    def format_chat_prompts(
        self,
        tokenizer:PreTrainedTokenizerBase,
        input_text: List[str]
    ):
        """
        Input
        tokenizer:PreTrainedTokenizerBase: The tokenizer
        input_text: List[str]: A list of texts to be tokenized

        Output:
        A dictionary, including 'input_ids' and 'attention_mask'
        """
        messages = [
            [{"role": "user", "content": prompt}] for prompt in input_text
        ]
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False
        )

        return model_inputs
    
    def format_prompt(
        self,
        input_text: List[str],
        tokenizer:PreTrainedTokenizerBase,
        device: torch.device
    ):
        """
        Formats the prompts with text information and without text information, and returns their respective input_ids and attention_mask.

        For example, a return example for formatted_prompt is:
        {'input_ids': tensor([[151644,    872,    198,     32,  67487,   6686,    304,  12095, 151645,
            198, 151644,  77091,    198, 151667,    271, 151668,    271, 151643,
         151643, 151643, 151643, 151643, 151643, 151643],
        [151644,    872,    198,   2082,  13513,  72077,    367,    315,  11097,
           5748,   5956,    304,   3478,    323,  22526, 151645,    198, 151644,
          77091,    198, 151667,    271, 151668,    271]], device='cuda:2'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
       device='cuda:2')}
        """
        formatted_inputs = self.format_chat_prompts(
            tokenizer = tokenizer,
            input_text=input_text
        ).to(device)
        formatted_negative_inputs = self.format_chat_prompts(
            tokenizer = tokenizer,
            input_text=["" for _ in range(len(input_text))]
        ).to(device)
        return formatted_inputs, formatted_negative_inputs

    def get_mask_from_modality_ids_with_both_source_and_edit_images(
        self,
        modality_ids:torch.Tensor
    ):
        """
        This function processes modality_ids that have two consecutive blocks of 256 ones.
        The first block is for the source_image, the second is for the edit_image.
        We change the 256 ones of the first block to 0.
        """

        # The starting position of our first block of 1s is always 4.
        modality_ids[:,4:4+256]=0
        return modality_ids.to(torch.bool)
    
    def prepare_inputs_for_text_to_image(
        self,
        formatted_inputs: torch.Tensor,
        formatted_negative_inputs: torch.Tensor,
        image_embeddings: torch.Tensor,
        tokenizer:PreTrainedTokenizerBase,
    ):
        """
        Used for Task 2: Text-to-Image Generation
        Input: formatted_inputs and formatted_negative_inputs are the results after wrapping with a chat template,
               with and without text, respectively. They are dictionaries containing input_ids and attention_mask.
               Of course, the seq_len of formatted_negative_inputs will never be greater than that of formatted_inputs.
               Their shapes are both (batch_size, seq_len).
               image_embeddings are the image embeddings for the current computation step. Its shape is (B, 256, 4096).

        Process: (1) Append <BOI> after formatted_inputs and formatted_negative_inputs respectively,
                    then add <EOI> and <|im_end|> at the end.
                 (2) Convert the above input_ids into embeddings, with shape (batch_size, seq_len, hidden_dim=4096).
                 (3) Insert image_embeddings between <BOI> and <EOI>.
                 (4) negative_prompt_embeds is always shorter than prompt_embeds, we need to pad negative_prompt_embeds at the end.
                 (5) The above content forms inputs_embeds. We also need to construct attention_mask, position_ids, and modality_ids
                     based on input_ids.

        Output: inputs is a dictionary with keys: inputs_embeds, attention_mask, position_ids, and modality_ids.
                negative_inputs is the same.
                The values in inputs and negative_inputs are all tensors, and their seq_len needs to be the same.
                This is necessary so they can be fed into the model's forward pass in the same batch.
        """

        # Variable naming convention
        # inputs_inputs_embeds, inputs_attention_mask, inputs_position_ids, inputs_modality_ids are respectively
        # the 'inputs_embeds', 'attention_mask', 'position_ids', 'modality_ids' of the returned dictionary `inputs`.
        # The same applies to negative_inputs.

        device = formatted_inputs['input_ids'].device
        #####################################
        ## First, process formatted_inputs
        #####################################
        batch_size = formatted_inputs['input_ids'].shape[0]

        boi_tokens = tokenizer.encode("<BOI>",return_tensors='pt').repeat(batch_size, 1).to(device)
        final_tokens = tokenizer.encode("<EOI><|im_end|>",return_tensors='pt').repeat(batch_size, 1).to(device)

        inputs_input_ids = torch.cat([formatted_inputs['input_ids'],boi_tokens],dim=1)
        # inputs_embeds : (B, SEQ_LEN+1, 4096)
        inputs_inputs_embeds = self.model.embed_tokens(inputs_input_ids)
        # inputs_embeds : (B, SEQ_LEN+1, 4096) -> (B, SEQ_LEN+1+256, 4096)
        inputs_inputs_embeds = torch.cat([inputs_inputs_embeds, image_embeddings], dim=1)

        final_tokens_embeds = self.model.embed_tokens(final_tokens)
        # inputs_embeds : (B, SEQ_LEN+1+256, 4096) -> (B, SEQ_LEN+1+256+2, 4096)
        inputs_inputs_embeds = torch.cat([inputs_inputs_embeds, final_tokens_embeds], dim=1)

        # Next, process the attention_mask

        # The newly added <BOI> + Image token + <EOI> + <|im_end|>
        num_token_difference = 1 + 256 + 2

        # The newly added part has no padding, so all attention values are true.
        new_attention_mask = torch.ones((batch_size,num_token_difference)).to(device)
        inputs_attention_mask = torch.cat([formatted_inputs['attention_mask'],new_attention_mask],dim=1)
        
        # Set position_ids
        inputs_position_ids = inputs_attention_mask.long().cumsum(-1) - 1
        inputs_position_ids.masked_fill_(inputs_attention_mask == 0, 1)

        # Set modality_ids
        inputs_modality_ids = torch.zeros((batch_size,inputs_inputs_embeds.shape[1]),device=device)
        num_prefix_token = formatted_inputs['input_ids'].shape[1]+1 # Should we add 1? Because <BOI> is after it. # Yes, it should be, added.
        # The tokens at the fixed 256 positions are images.
        inputs_modality_ids[:,num_prefix_token:num_prefix_token+256] = 1

        # Organize inputs
        inputs = {
            'inputs_embeds' : inputs_inputs_embeds,
            'attention_mask' : inputs_attention_mask,
            'position_ids' : inputs_position_ids,
            'modality_ids' : inputs_modality_ids
        }

        #####################################
        ## Then, process negative_inputs
        #####################################

        # Same logic as with inputs
        negative_inputs_input_ids = torch.cat([formatted_negative_inputs['input_ids'],boi_tokens],dim=1)
        negative_inputs_inputs_embeds = self.model.embed_tokens(negative_inputs_input_ids)
        negative_inputs_inputs_embeds = torch.cat([negative_inputs_inputs_embeds, image_embeddings], dim=1)
        negative_inputs_inputs_embeds = torch.cat([negative_inputs_inputs_embeds, final_tokens_embeds], dim=1)

        negative_inputs_attention_mask = torch.cat([formatted_negative_inputs['attention_mask'],new_attention_mask],dim=1)

        negative_inputs_position_ids = negative_inputs_attention_mask.long().cumsum(-1) - 1
        negative_inputs_position_ids.masked_fill_(negative_inputs_attention_mask == 0, 1)

        negative_inputs_modality_ids = torch.zeros((batch_size,negative_inputs_inputs_embeds.shape[1]),device=device)
        num_prefix_token = formatted_negative_inputs['input_ids'].shape[1]+1
        negative_inputs_modality_ids[:,num_prefix_token:num_prefix_token+256] = 1

        # However, later on, depending on the situation, i.e., when using CFG, we might concatenate inputs and negative_inputs
        # and feed them into the large language model at once.
        # Therefore, we need the shapes of all items in negative_inputs to be the same, especially the SEQ_LEN.
        # The SEQ_LEN of negative_inputs will certainly not be greater than that of inputs. So we append to the items in negative_inputs.

        batch_size,seq_len,hidden_dim = negative_inputs_inputs_embeds.shape
        num_token_diff = inputs_attention_mask.shape[1]-negative_inputs_attention_mask.shape[1]

        # As long as attention_mask is 0, the other values do not matter.
        additional_inputs_embeds = torch.zeros((batch_size,num_token_diff,hidden_dim),device=device,dtype=negative_inputs_inputs_embeds.dtype)
        additional_attention_mask = torch.zeros(batch_size,num_token_diff,device=device)
        additional_positional_ids = torch.ones(batch_size,num_token_diff,device=device)
        additional_modality_ids = torch.zeros(batch_size,num_token_diff,device=device)

        # Append
        negative_inputs_inputs_embeds = torch.cat([negative_inputs_inputs_embeds,additional_inputs_embeds],dim=1)
        negative_inputs_attention_mask = torch.cat([negative_inputs_attention_mask,additional_attention_mask],dim=1)
        negative_inputs_position_ids = torch.cat([negative_inputs_position_ids,additional_positional_ids],dim=1)
        negative_inputs_modality_ids = torch.cat([negative_inputs_modality_ids,additional_modality_ids],dim=1)

        # Organize negative_inputs
        negative_inputs = {
            'inputs_embeds' : negative_inputs_inputs_embeds,
            'attention_mask' : negative_inputs_attention_mask,
            'position_ids' : negative_inputs_position_ids,
            'modality_ids' : negative_inputs_modality_ids
        }

        return inputs, negative_inputs
    
    def prepare_embeds_for_model_forward(
        self,
        image_embeds: torch.Tensor,
        tokenizer:PreTrainedTokenizerBase,
        device:torch.device
    ):
        """
        Used for Task 3, Image-to-Text generation.
        Uses an empty input template to wrap the image embedding, assembles it into inputs_embeds,
        and then uses model.generate to generate text.
        """
        formatted_inputs = self.format_chat_prompts(
            tokenizer = tokenizer,
            input_text=["" for _ in range(len(image_embeds))]
        ).to(device)
        input_ids = formatted_inputs['input_ids']
        batch_size = input_ids.shape[0]
        boi_tokens = tokenizer.encode("<BOI>",return_tensors='pt').repeat(batch_size, 1).to(device)
        eoi_tokens = tokenizer.encode("<EOI>",return_tensors='pt').repeat(batch_size, 1).to(device)

        # inputs_embeds: (B,seq_len=12,4096)
        inputs_embeds = self.model.embed_tokens(input_ids)
        # boi_tokens_embeds, eoi_tokens_embeds: (B,1,4096)
        boi_tokens_embeds = self.model.embed_tokens(boi_tokens)
        eoi_tokens_embeds = self.model.embed_tokens(eoi_tokens)


        # According to the Qwen3 template, we need to insert the image embedding after the third token,
        # and also wrap it with <BOI> and <EOI>, which is:
        # 151644 '<|im_start|>'
        # 872 'user'      
        # 198 '\n'
        # 151669 '<BOI>'  
        # ...      
        # [256 image embedding]
        # ...
        # 151670 '<EOI>'
        # 151645 '<|im_end|>'
        # 198 '\n'        
        # 151644 '<|im_start|>'
        # 77091 'assistant' 
        # 198 '\n'        
        # 151667 '<think>'   
        # 271 '\n\n'      
        # 151668 '</think>'  
        # 271 '\n\n'

        # image_embeds is the same except for the image tokens. image_embeds are inserted from the fourth token position.
        image_embeds = torch.cat([inputs_embeds[:,:3],boi_tokens_embeds,image_embeds,eoi_tokens_embeds,inputs_embeds[:,3:]],dim=1) 
        attention_mask = torch.ones(image_embeds.shape[:-1],device=device,dtype=torch.long) # attention is all 1s
        position_ids = attention_mask.long().cumsum(-1) - 1
        modality_ids = torch.zeros(image_embeds.shape[:-1],device=device,dtype=torch.long)
        modality_ids[:,4:4+256] = 1
        inputs = {
            'inputs_embeds' : image_embeds,
            'attention_mask' : attention_mask,
            'position_ids' : position_ids,
            'modality_ids' : modality_ids
        }

        return inputs   
    
    def prepare_inputs_for_edit_image(
        self,
        formatted_inputs: torch.Tensor,
        formatted_negative_inputs: torch.Tensor,
        source_image_embedding: torch.Tensor,
        edit_image_embeddings: torch.Tensor,
        tokenizer:PreTrainedTokenizerBase,
    ):
        """
        Used for Task 4: Image Editing

        Similar to prepare_inputs_for_text_to_image
        """

        device = formatted_inputs['input_ids'].device
        #####################################
        ## First, process formatted_inputs
        #####################################
        batch_size = formatted_inputs['input_ids'].shape[0]

        boi_tokens = tokenizer.encode("<BOI>",return_tensors='pt').repeat(batch_size, 1).to(device)
        eoi_tokens = tokenizer.encode("<EOI>",return_tensors='pt').repeat(batch_size, 1).to(device)
        final_tokens = tokenizer.encode("<EOI><|im_end|>",return_tensors='pt').repeat(batch_size, 1).to(device)

        # First, we add <BOI> <EOI> after
        # 151644	<|im_start|>
        # 872	user
        # 198	\n
        inputs_input_ids = torch.cat([formatted_inputs['input_ids'][:,:3],boi_tokens,eoi_tokens,formatted_inputs['input_ids'][:,3:]],dim=1)
        # inputs_embeds : (B, SEQ_LEN+2, 4096)
        inputs_inputs_embeds = self.model.embed_tokens(inputs_input_ids)
        # source_image_embedding is inserted after the fourth token, i.e., after the following tokens:
        # 151644	<|im_start|>
        # 872	user
        # 198	\n
        # 待定	<BOI>
        # inputs_embeds : (B, SEQ_LEN+2, 4096) -> (B, SEQ_LEN+2+256, 4096)
        inputs_inputs_embeds = torch.cat([inputs_inputs_embeds[:,:4], source_image_embedding,inputs_inputs_embeds[:,4:]], dim=1)
        
        boi_tokens_embeds = self.model.embed_tokens(boi_tokens)
        final_tokens_embeds = self.model.embed_tokens(final_tokens)

        # Then, first add a <BOI>,
        # and then append the edit_image_embeddings and the embeddings of the final two tokens at the end.
        # TBD	<EOI>
        # 151645	<|im_end|>
        inputs_inputs_embeds = torch.cat([inputs_inputs_embeds, boi_tokens_embeds, edit_image_embeddings,final_tokens_embeds], dim=1)

        # Then we need to create the attention_mask.
        # We should note that the only part of the attention_mask that could be False is the padded part in
        # formatted_inputs['attention_mask'] due to different text lengths after tokenization.
        # This part with False has already been calculated, so we can just use it.

        # First, for the newly added <BOI> + source_image token + <EOI>
        num_token_difference = 1 + 256 + 1
        new_attention_mask = torch.ones((batch_size,num_token_difference)).to(device)
        # Insert starting from the third position; the attention_mask for <BOI> is also newly added by us.
        inputs_attention_mask = torch.cat([formatted_inputs['attention_mask'][:,:3],new_attention_mask,formatted_inputs['attention_mask'][:,3:]],dim=1)
        # Then add the final <BOI> + edit_image token + <EOI> + <|im_end|>
        num_token_difference = 1 + 256 + 2
        new_attention_mask = torch.ones((batch_size,num_token_difference)).to(device)
        inputs_attention_mask = torch.cat([inputs_attention_mask,new_attention_mask],dim=1)

        # Set position_ids, which are generated entirely from inputs_attention_mask.
        inputs_position_ids = inputs_attention_mask.long().cumsum(-1) - 1
        inputs_position_ids.masked_fill_(inputs_attention_mask == 0, 1)

        # Set modality_ids
        # We first generate a tensor of all zeros.
        # Then we have two consecutive blocks of 256 tensors that are image tokens, their modality_ids value is 1.
        # What we need to do is to accurately calculate the starting positions of these two consecutive blocks of 256 tensors, which is very important.
        inputs_modality_ids = torch.zeros((batch_size,inputs_inputs_embeds.shape[1]),device=device)
        # The first position is fixed, starting from 4.
        num_prefix_token = 4
        inputs_modality_ids[:,num_prefix_token:num_prefix_token+256] = 1
        # The second position is the number of tokens in formatted_inputs['input_ids'] plus
        # the source_image's <BOI> + 256 + <EOI> plus the starting <BOI> of the edit_image, totaling 1+256+1+1=259.
        num_prefix_token = formatted_inputs['input_ids'].shape[1]+259
        inputs_modality_ids[:,num_prefix_token:num_prefix_token+256] = 1
        # Organize inputs
        inputs = {
            'inputs_embeds' : inputs_inputs_embeds,
            'attention_mask' : inputs_attention_mask,
            'position_ids' : inputs_position_ids,
            'modality_ids' : inputs_modality_ids
        }

        #####################################
        ## Then, process negative_inputs
        #####################################

        # The ['input_ids'] for negative_inputs are all the same, with no padding, so the attention_mask has no False values either.
        negative_inputs_input_ids = torch.cat([formatted_negative_inputs['input_ids'][:,:3],boi_tokens,eoi_tokens,formatted_negative_inputs['input_ids'][:,3:]],dim=1)
        negative_inputs_inputs_embeds = self.model.embed_tokens(negative_inputs_input_ids)
        negative_inputs_inputs_embeds = torch.cat([
            negative_inputs_inputs_embeds[:,:4], 
            source_image_embedding,
            negative_inputs_inputs_embeds[:,4:],
            boi_tokens_embeds,
            edit_image_embeddings,
            final_tokens_embeds
        ], dim=1)

        # attention_mask
        num_token_difference = 1 + 256 + 1
        new_attention_mask = torch.ones((batch_size,num_token_difference)).to(device)
        # Insert starting from the third position; the attention_mask for <BOI> is also newly added by us.
        negative_inputs_attention_mask = torch.cat([formatted_negative_inputs['attention_mask'][:,:3],new_attention_mask,formatted_negative_inputs['attention_mask'][:,3:]],dim=1)
        # Then add the final <BOI> + edit_image token + <EOI> + <|im_end|>
        num_token_difference = 1 + 256 + 2
        new_attention_mask = torch.ones((batch_size,num_token_difference)).to(device)
        negative_inputs_attention_mask = torch.cat([negative_inputs_attention_mask,new_attention_mask],dim=1)

        # position_ids
        negative_inputs_position_ids = negative_inputs_attention_mask.long().cumsum(-1) - 1
        negative_inputs_position_ids.masked_fill_(negative_inputs_attention_mask == 0, 1)

        # modality_ids
        negative_inputs_modality_ids = torch.zeros((batch_size,negative_inputs_inputs_embeds.shape[1]),device=device)
        # The first position is fixed, starting from 4.
        num_prefix_token = 4
        negative_inputs_modality_ids[:,num_prefix_token:num_prefix_token+256] = 1
        # The second position is the number of tokens in formatted_negative_inputs['input_ids'] plus
        # the source_image's <BOI> + 256 + <EOI> plus the starting <BOI> of the edit_image, totaling 1+256+1+1=259.
        num_prefix_token = formatted_negative_inputs['input_ids'].shape[1]+259
        negative_inputs_modality_ids[:,num_prefix_token:num_prefix_token+256] = 1

        batch_size,seq_len,hidden_dim = negative_inputs_inputs_embeds.shape
        num_token_diff = inputs_attention_mask.shape[1]-negative_inputs_attention_mask.shape[1]

        # As long as attention_mask is 0, the other values do not matter.
        additional_inputs_embeds = torch.zeros((batch_size,num_token_diff,hidden_dim),device=device,dtype=negative_inputs_inputs_embeds.dtype)
        additional_attention_mask = torch.zeros(batch_size,num_token_diff,device=device)
        additional_positional_ids = torch.ones(batch_size,num_token_diff,device=device)
        additional_modality_ids = torch.zeros(batch_size,num_token_diff,device=device)

        # Append
        negative_inputs_inputs_embeds = torch.cat([negative_inputs_inputs_embeds,additional_inputs_embeds],dim=1)
        negative_inputs_attention_mask = torch.cat([negative_inputs_attention_mask,additional_attention_mask],dim=1)
        negative_inputs_position_ids = torch.cat([negative_inputs_position_ids,additional_positional_ids],dim=1)
        negative_inputs_modality_ids = torch.cat([negative_inputs_modality_ids,additional_modality_ids],dim=1)

        # Organize negative_inputs
        negative_inputs = {
            'inputs_embeds' : negative_inputs_inputs_embeds,
            'attention_mask' : negative_inputs_attention_mask,
            'position_ids' : negative_inputs_position_ids,
            'modality_ids' : negative_inputs_modality_ids
        }

        return inputs, negative_inputs
        pass



    # The four inference functions all use type hints like Union[List[str], str] and Union[Image.Image, List[Image.Image]].
    # This approach allows handling both a batch of data and a single data point. When processing a batch,
    # it returns a batch, and when processing a single data point, it returns a single data point.
    # The internal processing is all batch-based, but when a single data input is encountered, it will be wrapped into a list.
    # When the output list contains only a single item, it will be "unpacked".

    @torch.inference_mode()
    def generate_text(
        self,
        input_text: Union[List[str],str],
        max_new_tokens: int = 256,
        **kwargs
    ) -> Union[List[str],str]:
        # Task 1: Pure Text Generation
        start_device = self.model.device

        if isinstance(input_text,str):
            input_text = [input_text]

        messages = [
            [{"role": "user", "content": prompt}] for prompt in input_text
        ]

        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        ).to(start_device)

        model_inputs['modality_ids'] = torch.zeros_like(model_inputs['input_ids'])
        
        generated_ids = self.generate(
            **model_inputs,
            max_new_tokens = max_new_tokens
        )
        output_ids = generated_ids[:,len(model_inputs.input_ids[0]):].tolist()

        generated_text = [self.tokenizer.decode(item, skip_special_tokens=True).strip("\n") for item in output_ids]

        if len(generated_text)==1:
            generated_text=generated_text[0]

        return generated_text

    
    # Managing which device tensors are on may still require attention or adjustments
    @torch.inference_mode()
    def text_to_image(
        self,
        input_text: Union[List[str],str],
        num_inference_steps: int = 250, # Following the setting from transfusion
        guidance_scale: float = 1.55, # The default value is set to the value in the paper. When this value is 1, there is no need to calculate pred_unconditional.
                                      # final_prediction = pred_unconditional + guidance_scale * (pred_conditional - pred_unconditional)
        output_type: Optional[str] = "pil",
        **kwargs
    ) -> Union[Image.Image,List[Image.Image]]:
        # Task 2: Text-to-Image Generation
        # For the text-to-image process, we referred to the diffusers code.
        # https://github.com/huggingface/diffusers/blob/7bc0a07b1947bcbe0e84bbe9ecf8ae2d234382c4/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py

        # The device where the model's parameters are located may change as the model's forward function progresses
        # (it might be that a single card cannot load the entire model).
        # What we need to manage is, for some tensors that we create ourselves and that do not belong to any nn.Module,
        # which device they are on when they are first created.
        start_device = self.model.device

        # Determine whether to use CFG
        if guidance_scale == 1.:
            self.do_classifier_free_guidance = False
        else:
            self.do_classifier_free_guidance = True

        # Prepare the text
        if isinstance(input_text,str):
            input_text = [input_text]

        formatted_inputs, formatted_negative_inputs = self.format_prompt(
            input_text = input_text,
            tokenizer = self.tokenizer,
            device=start_device
        )

        # if self.do_classifier_free_guidance:
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        
        # Prepare the timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, 
            num_inference_steps, 
            device = start_device, # This should probably be set
            timesteps = None, 
            sigmas = None
        )

        # Prepare a random image in latent space to serve as the starting point for the denoising process
        # num_channels_latents = 4 # In the paper, it is 4

        # Note, the batch_size of latents at initialization, even in the case of CFG, is not 2*len(input_text).
        # For the conditional and unconditional versions of the same prompt, their initial values are the same.
        batch_size = len(input_text)
        latents = torch.randn(
            (batch_size,4,32,32), # (-1,8,32,32) is specified in the paper, but the VAE from the paper does not seem to be public.
                                  # The VAE link in the paper points to `stabilityai/sd-vae-ft-mse`, so we use that, and its channel is 4.
            device=start_device,
            dtype=self.dtype
        )
        latents = latents * self.scheduler.init_noise_sigma # In fact, in our experiments, self.scheduler.init_noise_sigma = 1

        # Denoising loop
        for t in tqdm(timesteps):

            # 1. Prepare the time embeddings required by the U-Net
            t_emb = self.get_time_embed(sample=latents, timestep=t)
            emb = self.time_embedding(t_emb)

            # 2. Obtain the embedding through the Downsampler
            sample, res_sample = self.model.unet_downsampler(
                hidden_states=latents,
                temb=emb
            )
            # In our research, the shape of latents is (B, 4096, 16, 16), we need to convert it to (B, 256, 4096)
            image_embeddings = sample.view((*sample.shape[:2],-1)).permute(0,2,1)            

            # Prepare the dictionary to be fed into the model
            inputs, negative_inputs = self.prepare_inputs_for_text_to_image(
                formatted_inputs=formatted_inputs,
                formatted_negative_inputs=formatted_negative_inputs,
                image_embeddings=image_embeddings,
                tokenizer = self.tokenizer,
            )

            # If we use CFG
            # 1. We need to calculate the noise for both conditional (with text) and unconditional (without text) cases
            # 2. We need two identical copies of emb
            # 3. We also need two identical copies of res_sample
            if self.do_classifier_free_guidance:
                for key in inputs.keys():
                    inputs[key] = torch.cat([inputs[key],negative_inputs[key]],dim=0)
                
                emb = emb.repeat((2,1))
                res_sample = tuple(torch.cat([item, item],dim=0) for item in res_sample)

            # 3. Feed into the model to get the noise prediction
            transformer_outputs = self.model(
                **inputs,
                use_cache=False
            )

            mask = inputs['modality_ids'].to(torch.bool)
            # We actually only need the hidden states of the image tokens.
            # For each data sample, inputs['modality_ids'] always has 256 ones, so we can reshape with confidence.
            # What we want to get from the model is this image_hidden_states. The model is in fact equivalent to a part of the U-Net in a traditional diffusion model.
            image_hidden_states = transformer_outputs.last_hidden_state[mask].reshape(-1,256,4096)

            # Convert back to the shape accepted by the U-Net's Upsampler
            image_hidden_states = image_hidden_states.permute(0, 2, 1).reshape(-1, 4096, 16, 16)

            # Then send it back to the Upsampler
            # If we do CFG, the shape of noise_pred is (2*batch_size, 8, 32, 32), otherwise it is (batch_size, 8, 32, 32)
            noise_pred = self.model.unet_upsampler(
                hidden_states=image_hidden_states,
                res_samples=res_sample,
                temb=emb
            )

            if self.do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2) # We put the conditional one first
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Denoise
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Reconstruct the image
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        do_denormalize = [True] * image.shape[0] # We do not perform an nsfw check
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Ensure that when there is only one image in the return value, the Image.Image object is taken out of the list
        if len(image) == 1:
            image = image[0]
        return image

    @torch.inference_mode()
    def generate_caption(
        self,
        image: Union[Image.Image,List[Image.Image]],
        sample_mode = "sample", # "sample" or "argmax" # Used to specify whether to sample a value from the distribution or take a deterministic one.
        max_new_tokens: int = 256,
        **kwargs
    ) -> Union[List[str],str]:
        # Task 3: Image-to-Text Generation (Captioning)
        # The process of encoding an image is referenced from https://github.com/huggingface/diffusers/blob/50dea89dc6036e71a00bc3d57ac062a80206d9eb/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py

        start_device = self.model.device

        if not isinstance(image,List):
            image = [image]

        image = self.image_processor.preprocess(image).to(
            dtype=self.dtype, 
            device=self.device
        )

        init_latents = retrieve_latents(
            encoder_output=self.vae.encode(image),
            sample_mode=sample_mode
        )

        # In `stabilityai/sd-vae-ft-mse`, scaling_factor is not set. However, since we load it
        # using the from_pretrained method of AutoencoderKL, this value is set to 0.18215 by default.
        init_latents = self.vae.scaling_factor * init_latents
        t_emb = self.get_time_embed(sample=init_latents, timestep=0)
        emb = self.time_embedding(t_emb)
        
        # In Task 3: Image-to-Text, we only need sample and not res_sample.
        sample, _ = self.model.unet_downsampler(
            hidden_states=init_latents,
            temb=emb
        )

        # sample_embed: (B, 256, 4096)
        sample_embed = sample.reshape(*sample.shape[:-2],-1).permute(0,2,1)

        model_inputs = self.prepare_embeds_for_model_forward(
            image_embeds=sample_embed,
            tokenizer=self.tokenizer,
            device=start_device
        )
        
        generated_ids = self.generate(
            **model_inputs,
            max_new_tokens = max_new_tokens
        )

        generated_text = [self.tokenizer.decode(item.tolist(), skip_special_tokens=True).strip("\n") for item in generated_ids]

        if len(generated_text)==1:
            generated_text=generated_text[0]

        return generated_text

    @torch.inference_mode()
    def edit_image(
        self,
        source_image: Union[Image.Image,List[Image.Image]],
        edit_prompt: Union[List[str],str],
        strength: float = 1.0, # Between (0, 1]. 1 means starting the generation of the edited image from random noise. The closer to 0, the less noise the initial image of the denoising process contains (the initial image is obtained by adding noise to the pre-edit image).
        sample_mode = "sample", # "sample" or "argmax" # Used to specify whether to sample a value from the distribution or take a deterministic one.
        guidance_scale: float = 1.55, # The paper does not actually mention whether CFG was used, or what this value is.
                                      # final_prediction = pred_unconditional + guidance_scale * (pred_conditional - pred_unconditional)
        output_type: Optional[str] = "pil",
        num_inference_steps: Optional[int] = 50,
        **kwargs
    ) -> Union[Image.Image,List[Image.Image]]:
        # Task 4: Image Editing
        # This task can only be used with fine-tuned weights.
        # https://github.com/huggingface/diffusers/blob/7bc0a07b1947bcbe0e84bbe9ecf8ae2d234382c4/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
        # https://github.com/huggingface/diffusers/blob/50dea89dc6036e71a00bc3d57ac062a80206d9eb/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py

        if strength <= 0 or strength > 1:
            raise ValueError(f"The value of strength should in (0.0, 1.0] but is {strength}")

        start_device = self.model.device

        # Determine whether to use CFG
        if guidance_scale == 1.:
            self.do_classifier_free_guidance = False
        else:
            self.do_classifier_free_guidance = True

        if isinstance(edit_prompt,str):
            edit_prompt = [edit_prompt]

        if not isinstance(source_image,List):
            source_image = [source_image]

        if len(edit_prompt) != len(source_image):
            raise ValueError(f"Expected equal lengths but got {len(edit_prompt)} and {len(source_image)}")

        formatted_inputs, formatted_negative_inputs = self.format_prompt(
            input_text = edit_prompt,
            tokenizer = self.tokenizer,
            device=start_device
        )

        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, 
            num_inference_steps, 
            device = start_device, # This should probably be set
            timesteps = None, 
            sigmas = None
        )

        # Adjust timesteps and num_inference_steps based on strength
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, 
            strength, 
            device=start_device
        )

        # For example, the return value of the above retrieve_timesteps:
        # timesteps : tensor([981, 961, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,
        # 721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461,
        # 441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181,
        # 161, 141, 121, 101,  81,  61,  41,  21,   1], device='cuda:0') torch.Size([51])
        # num_inference_steps : 50

        # After setting strength=0.75 and processing with self.get_timesteps:
        # timesteps : tensor([741, 721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481,
        # 461, 441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201,
        # 181, 161, 141, 121, 101,  81,  61,  41,  21,   1], device='cuda:0') torch.Size([38])
        # num_inference_steps : 37

        image = self.image_processor.preprocess(source_image).to(
            dtype=self.dtype, 
            device=self.device
        )

        ## Original image
        ##
        init_latents = retrieve_latents(
            encoder_output=self.vae.encode(image),
            sample_mode=sample_mode
        )
        # In `stabilityai/sd-vae-ft-mse`, scaling_factor is not set. However, since we load it
        # using the from_pretrained method of AutoencoderKL, this value is set to 0.18215 by default.
        init_latents = self.vae.scaling_factor * init_latents 

        t_emb_for_source_image = self.get_time_embed(sample=init_latents, timestep=0)
        emb_for_source_image  = self.time_embedding(t_emb_for_source_image)

        sample_for_source_image, _ = self.model.unet_downsampler(
            hidden_states=init_latents,
            temb=emb_for_source_image
        )

        # sample_for_source_image_embed: (B, 256, 4096)
        sample_for_source_image_embed = sample_for_source_image.reshape(*sample_for_source_image.shape[:-2],-1).permute(0,2,1)

        ## Noisy version of the original image, as the starting point for the denoising process
        ##
        noise = torch.randn(
            init_latents.shape,
            device=start_device,
            dtype=init_latents.dtype
        )
        # init_latents.shape[0] is the batch_size
        latent_timestep = timesteps[:1].repeat(init_latents.shape[0])
        # The noisy image serves as the starting point for the diffusion process. num_inference_steps determines the level of noise.
        init_latents_for_edited_image = self.scheduler.add_noise(init_latents, noise, latent_timestep)

        latents = init_latents_for_edited_image # The starting point of the diffusion process

        for t in tqdm(timesteps):
            t_emb_for_edited_image = self.get_time_embed(sample=latents, timestep=t)
            emb_for_edited_image  = self.time_embedding(t_emb_for_edited_image)

            sample_for_edited_image, res_sample_for_edited_image = self.model.unet_downsampler(
                hidden_states=latents,
                temb=emb_for_edited_image
            )

            sample_for_edited_image_embed = sample_for_edited_image.reshape(*sample_for_edited_image.shape[:-2],-1).permute(0,2,1)

            inputs, negative_inputs = self.prepare_inputs_for_edit_image(
                formatted_inputs=formatted_inputs,
                formatted_negative_inputs=formatted_negative_inputs,
                source_image_embedding = sample_for_source_image_embed,
                edit_image_embeddings=sample_for_edited_image_embed,
                tokenizer = self.tokenizer,
            )

            # If we use CFG
            # 1. We need to calculate the noise for both conditional (with text) and unconditional (without text) cases
            # 2. We need two identical copies of emb
            # 3. We also need two identical copies of res_sample
            if self.do_classifier_free_guidance:
                for key in inputs.keys():
                    inputs[key] = torch.cat([inputs[key],negative_inputs[key]],dim=0)
                
                emb_for_edited_image = emb_for_edited_image.repeat((2,1))
                res_sample_for_edited_image = tuple(torch.cat([item, item],dim=0) for item in res_sample_for_edited_image)

            # 3. Feed into the model to get the noise prediction
            transformer_outputs = self.model(
                **inputs,
                use_cache=False
            )

            mask = self.get_mask_from_modality_ids_with_both_source_and_edit_images(inputs['modality_ids'])
            # We actually only need the hidden states of the image tokens.
            # For each data sample, inputs['modality_ids'] always has 256 ones, so we can reshape with confidence.
            # What we want to get from the model is this image_hidden_states. The model is in fact equivalent to a part of the U-Net in a traditional diffusion model.
            image_hidden_states = transformer_outputs.last_hidden_state[mask].reshape(-1,256,4096)

            # Convert back to the shape accepted by the U-Net's Upsampler
            image_hidden_states = image_hidden_states.permute(0, 2, 1).reshape(-1, 4096, 16, 16)

            # Then send it back to the Upsampler
            # If we do CFG, the shape of noise_pred is (2*batch_size, 8, 32, 32), otherwise it is (batch_size, 8, 32, 32)
            noise_pred = self.model.unet_upsampler(
                hidden_states=image_hidden_states,
                res_samples=res_sample_for_edited_image,
                temb=emb_for_edited_image
            )

            if self.do_classifier_free_guidance:
                # We put the conditional one first
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Denoise
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Reconstruct the image
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

        do_denormalize = [True] * image.shape[0] # We do not perform an nsfw check
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Ensure that when there is only one image in the return value, the Image.Image object is taken out of the list
        if len(image) == 1:
            image = image[0]
        return image

# In fact, we need to know some details, for example, whether the image was center-cropped, whether it was resized,
# and if there was normalization, what are the parameters for normalization? (Some use 0.5 for everything, others use imagenet values).
# These details are also very important. We need to ensure that it is consistent with the stabilityai/sd-vae-ft-mse specified in the paper.