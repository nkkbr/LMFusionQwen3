import torch
from torch import nn
from typing import Optional, Tuple, List
from diffusers.utils import logging, deprecate

from diffusers.models.unets.unet_2d_blocks import get_down_block, ResnetBlock2D, Attention, Upsample2D
logger = logging.get_logger(__name__) 

class LMFusionDownsampler(nn.Module):
    """
    A 2-block U-Net Downsampler as described in the LMFusion paper.
    It converts a VAE latent of shape (B, 8, 32, 32) into a feature map of shape (B, 4096, 16, 16).
    It utilizes tools from the diffusers library, such as get_down_block.
    """
    def __init__(
        self,
        in_channels: int = 4,
        temb_channels: int = 1280,
        num_layers_per_block: int = 2,
        hidden_size: int = 4096
    ):
        super().__init__()
        self.channels = [320, 1280, hidden_size]
        resnet_eps = 1e-6
        resnet_act_fn = "silu"
        
        # 1. Input convolutional layer
        # (B, 8, 32, 32) -> (B, 320, 32, 32)
        self.conv_in = nn.Conv2d(in_channels, self.channels[0], kernel_size=3, padding=1)
        
        # 2. First downsampling block (with self-attention and spatial reduction)
        # (B, 320, 32, 32) -> (B, 1280, 16, 16)
        self.down_block_1 = get_down_block(
            down_block_type="AttnDownBlock2D",
            num_layers=num_layers_per_block,
            in_channels=self.channels[0],
            out_channels=self.channels[1],
            temb_channels=temb_channels,
            add_downsample=True,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=32,
            attention_head_dim=self.channels[1] // 8, # Assuming 8 heads
            downsample_padding=(1, 1)
        )

        # 3. Second downsampling block (with self-attention, but no spatial reduction)
        # (B, 1280, 16, 16) -> (B, 4096, 16, 16)
        self.down_block_2 = get_down_block(
            down_block_type="AttnDownBlock2D",
            num_layers=num_layers_per_block,
            in_channels=self.channels[1],
            out_channels=self.channels[2],
            temb_channels=temb_channels,
            add_downsample=False, 
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=32,
            attention_head_dim=self.channels[2] // 16 # Assuming 16 heads
        )

    def forward(self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        
        # 1. Input convolution
        sample = self.conv_in(hidden_states)
        
        # Save the first skip connection (the output from conv_in)
        res_samples = (sample,)
        
        # 2. First downsampling block
        sample, down_block_1_res_samples = self.down_block_1(hidden_states=sample, temb=temb)
        res_samples += down_block_1_res_samples
        
        # 3. Second downsampling block
        sample, down_block_2_res_samples = self.down_block_2(hidden_states=sample, temb=temb)
        res_samples += down_block_2_res_samples
        
        # sample.shape is (B, 4096, 16, 16)
        # res_samples is a tuple containing features from all intermediate layers, to be used by the Upsampler.
        return sample, res_samples


class LMFusionUpsampler(nn.Module):
    """
    The U-Net Upsampler corresponding to the LMFusionDownsampler.
    It converts the (B, 4096, 16, 16) Transformer output and skip connections back to (B, 8, 32, 32).
    Tools from the diffusers library are not used here.
    This is mainly because AttnUpBlock2D from diffusers does not seem to be usable correctly in our setup.
    """
    def __init__(
        self,
        out_channels: int = 4,
        temb_channels: int = 1280,
        num_layers_per_block: int = 3,
        hidden_size: int = 4096, 
    ):
        super().__init__()
        self.channels = [320, 1280, hidden_size]
        resnet_eps = 1e-6
        self.num_layers_per_block = num_layers_per_block
        resnet_act_fn = "silu"
        
        # 1. First upsampling block (corresponds to Downsampler's block_2, no spatial up-dimensioning)
        # Input: (B, 4096, 16, 16), Output: (B, 1280, 16, 16)
        self.up_block_1 = AttnUpBlock2D_Adapted_For_LMFusion(
            res_in_channels=[4096+4096, 4096+4096, 4096+1280],
            res_out_channels=[4096, 4096, 1280],
            temb_channels=temb_channels,
            num_layers=num_layers_per_block,
            transformer_hidden_dim=[4096, 4096, 1280],
            transformer_heads=[32, 32, 8],
            upsample_type="conv",
        )
        
        # 2. Second upsampling block (corresponds to Downsampler's block_1, with spatial up-dimensioning)
        # Input: (B, 1280, 16, 16), Output: (B, 320, 32, 32)
        self.up_block_2 = AttnUpBlock2D_Adapted_For_LMFusion(
            res_in_channels=[1280+1280, 1280+1280, 1280+320],
            res_out_channels=[1280, 1280, 320],
            temb_channels=temb_channels,
            num_layers=num_layers_per_block,
            transformer_hidden_dim=[1280, 1280, 320],
            transformer_heads=[8,8,2],
            upsample_type=None
        )

        # 3. Output convolutional layer
        self.conv_norm_out = nn.GroupNorm(num_channels=self.channels[0], num_groups=32, eps=resnet_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(self.channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor, res_samples: Tuple[torch.Tensor, ...], temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        sample = hidden_states

        # 1. First upsampling block
        # It needs to consume the res_samples produced by the Downsampler's down_block_2 and the downsampler from down_block_1.
        # down_block_2 has num_layers(2) resnet outputs + 0 downsampler outputs = 2.
        # down_block_1's downsampler output = 1.
        # A total of 2+1 = 3 are needed, which matches num_up_layers=3.
        num_res_to_pop = self.num_layers_per_block
        
        sample = self.up_block_1(
            hidden_states=sample,
            res_hidden_states_tuple=res_samples[-num_res_to_pop:],
            temb=temb,
        )
        res_samples = res_samples[:-num_res_to_pop]
        
        # 2. Second upsampling block
        # It needs to consume the res_samples produced by the Downsampler's down_block_1 and conv_in.
        # down_block_1 has num_layers(2) resnet outputs.
        # conv_in has 1 resnet output.
        # A total of 2+1 = 3 are needed.
        sample = self.up_block_2(
            hidden_states=sample,
            res_hidden_states_tuple=res_samples, # consume the rest
            temb=temb,
        )
        
        # 3. Output convolution
        
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        
        return sample
    


class AttnUpBlock2D_Adapted_For_LMFusion(nn.Module):
    """
    This is our own modified class.
    The implementation is arguably not very elegant.
    """
    def __init__(
        self,
        res_in_channels: List = None, # In our setup, block1 [4096+4096, 4096+4096, 4096+1280], block2 [1280+1280, 1280+1280, 1280+320]
        res_out_channels: List = None, # In our setup, block1 [4096, 4096, 1280], block2 [1280, 1280, 320]
        temb_channels: int = 1280,
        resolution_idx: int = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        transformer_hidden_dim: List = None, # In our setup, block1 [4096, 4096, 1280], block2 [1280, 1280, 320]
        transformer_heads: List = None, # In our setup, block1  [32, 32, 8], block2 [8,8,2]
        output_scale_factor: float = 1.0,
        upsample_type: str = "conv",
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.upsample_type = upsample_type

        # if attention_head_dim is None:
        #     logger.warning(
        #         f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {out_channels}."
        #     )
        #     attention_head_dim = out_channels

        out_channels = res_out_channels[-1]

        for i in range(num_layers):

            resnets.append(
                ResnetBlock2D(
                    in_channels=res_in_channels[i],
                    out_channels=res_out_channels[i],
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                Attention(
                    transformer_hidden_dim[i],
                    heads=transformer_heads[i],
                    dim_head=transformer_hidden_dim[i]//transformer_heads[i],
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if upsample_type == "conv":
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        elif upsample_type == "resnet":
            self.upsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        up=True,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states, temb)
                hidden_states = attn(hidden_states)
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                if self.upsample_type == "resnet":
                    hidden_states = upsampler(hidden_states, temb=temb)
                else:
                    hidden_states = upsampler(hidden_states)

        return hidden_states
    

# ############################################################################################################
# Usage Example

# from typing import Union, Optional

# # From https://github.com/huggingface/diffusers/blob/7392c8ff5a2a3ea2b059a9e1fdc5164759844976/src/diffusers/models/unets/unet_2d_condition.py#L907-L932
# def get_time_embed(
#     sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]
# ) -> Optional[torch.Tensor]:
#     timesteps = timestep
#     if not torch.is_tensor(timesteps):
#         # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
#         # This would be a good case for the `match` statement (Python 3.10+)
#         is_mps = sample.device.type == "mps"
#         is_npu = sample.device.type == "npu"
#         if isinstance(timestep, float):
#             dtype = torch.float32 if (is_mps or is_npu) else torch.float64
#         else:
#             dtype = torch.int32 if (is_mps or is_npu) else torch.int64
#         timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
#     elif len(timesteps.shape) == 0:
#         timesteps = timesteps[None].to(sample.device)

#     # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
#     timesteps = timesteps.expand(sample.shape[0])
#     t_emb = time_proj(timesteps)
#     # `Timesteps` does not contain any weights and will always return f32 tensors
#     # but time_embedding might actually be running in fp16. so we need to cast here.
#     # there might be better ways to encapsulate this.
#     t_emb = t_emb.to(dtype=sample.dtype)
#     return t_emb

# from diffusers.models.embeddings import TimestepEmbedding, Timesteps
# time_proj = Timesteps(
#     num_channels=320, 
#     flip_sin_to_cos=True, 
#     downscale_freq_shift=0
# ) # The numbers here come from model = "CompVis/stable-diffusion-v1-4"
# time_embedding = TimestepEmbedding(
#     in_channels=320, 
#     time_embed_dim=1280
# )

# unet_downsampler = LMFusionDownsampler(
#             in_channels=8,
#             temb_channels=1280,
#             hidden_size=4096
#         )

# unet_upsampler = LMFusionUpsampler(
#     out_channels=8,
#     temb_channels=1280,
#     num_layers_per_block=3,
#     hidden_size=4096
# )

# sample = torch.randn(2,8,32,32)
# timestep = 999

# # 1.time

# t_emb = get_time_embed(sample=sample, timestep=timestep)
# emb = time_embedding(t_emb)

# # 2. down
# sample, res_samples = unet_downsampler(
#     hidden_states=sample,
#     temb=emb
# )

# # 3. up

# sample = unet_upsampler(
#     hidden_states=sample,
#     res_samples=res_samples,
#     temb=emb
# )
# ############################################################################################################