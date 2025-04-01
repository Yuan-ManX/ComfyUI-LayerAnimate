from typing import Any, Dict, List, Optional, Tuple, Union
from einops import rearrange, repeat
import numpy as np
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from .unet import TimestepEmbedSequential, ResBlock, Downsample, Upsample, TemporalConvBlock
from ..basics import zero_module, conv_nd
from ..modules.attention import SpatialTransformer, TemporalTransformer
from ..common import checkpoint

from diffusers import __version__
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.model_loading_utils import load_state_dict
from diffusers.utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    logging,
    _get_model_file,
    _add_variant
)
from omegaconf import ListConfig, DictConfig, OmegaConf


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class LayerControlNet(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        context_dim=None,
        use_scale_shift_norm=False,
        resblock_updown=False,
        num_heads=-1,
        num_head_channels=-1,
        transformer_depth=1,
        use_linear=False,
        use_checkpoint=False,
        temporal_conv=False,
        tempspatial_aware=False,
        temporal_attention=True,
        use_relative_position=True,
        use_causal_attention=False,
        temporal_length=None,
        addition_attention=False,
        temporal_selfatt_only=True,
        image_cross_attention=False,
        image_cross_attention_scale_learnable=False,
        default_fps=4,
        fps_condition=False,
        ignore_noisy_latents=True,
        condition_channels={},
        control_injection_mode='add',
        use_vae_for_trajectory=False,
    ):
        super().__init__()
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'
        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.temporal_attention = temporal_attention
        time_embed_dim = model_channels * 4
        self.use_checkpoint = use_checkpoint
        temporal_self_att_only = True
        self.addition_attention = addition_attention
        self.temporal_length = temporal_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        self.default_fps = default_fps
        self.fps_condition = fps_condition
        self.ignore_noisy_latents = ignore_noisy_latents
        assert len(condition_channels) > 0, 'Condition types must be specified'
        self.condition_channels = condition_channels
        self.control_injection_mode = control_injection_mode
        self.use_vae_for_trajectory = use_vae_for_trajectory

        ## Time embedding blocks
        self.time_proj = Timesteps(model_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embed = TimestepEmbedding(model_channels, time_embed_dim)

        if fps_condition:
            self.fps_embedding = TimestepEmbedding(model_channels, time_embed_dim)
            nn.init.zeros_(self.fps_embedding.linear_2.weight)
            nn.init.zeros_(self.fps_embedding.linear_2.bias)

        if "motion_score" in condition_channels:
            if control_injection_mode == 'add':
                self.motion_embedding = zero_module(conv_nd(dims, condition_channels["motion_score"], model_channels, 3, padding=1))
            elif control_injection_mode == 'concat':
                self.motion_embedding = zero_module(conv_nd(dims, condition_channels["motion_score"], condition_channels["motion_score"], 3, padding=1))
            else:
                raise ValueError(f"control_injection_mode {control_injection_mode} is not supported, use 'add' or 'concat'")
        if "sketch" in condition_channels:
            if control_injection_mode == 'add':
                self.sketch_embedding = zero_module(conv_nd(dims, condition_channels["sketch"], model_channels, 3, padding=1))
            elif control_injection_mode == 'concat':
                self.sketch_embedding = zero_module(conv_nd(dims, condition_channels["sketch"], condition_channels["sketch"], 3, padding=1))
            else:
                raise ValueError(f"control_injection_mode {control_injection_mode} is not supported, use 'add' or 'concat'")
        if "trajectory" in condition_channels:
            if control_injection_mode == 'add':
                if use_vae_for_trajectory:
                    self.trajectory_embedding = zero_module(conv_nd(dims, condition_channels["trajectory"], model_channels, 3, padding=1))
                else:
                    self.trajectory_embedding = ControlNetConditioningEmbedding(model_channels, condition_channels["trajectory"])
            elif control_injection_mode == 'concat':
                if use_vae_for_trajectory:
                    self.trajectory_embedding = zero_module(conv_nd(dims, condition_channels["trajectory"], condition_channels["trajectory"], 3, padding=1))
                else:
                    self.trajectory_embedding = ControlNetConditioningEmbedding(condition_channels["trajectory"], condition_channels["trajectory"])
            else:
                raise ValueError(f"control_injection_mode {control_injection_mode} is not supported, use 'add' or 'concat'")

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))
            ]
        )

        if self.addition_attention:
            self.init_attn = TimestepEmbedSequential(
                TemporalTransformer(
                    model_channels,
                    n_heads=8,
                    d_head=num_head_channels,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint, only_self_att=temporal_selfatt_only,
                    causal_attention=False, relative_position=use_relative_position,
                    temporal_length=temporal_length
                )
            )

        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, time_embed_dim, dropout,
                        out_channels=mult * model_channels, dims=dims, use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm, tempspatial_aware=tempspatial_aware,
                        use_temporal_conv=temporal_conv
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(ch, num_heads, dim_head,
                            depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                            use_checkpoint=use_checkpoint, disable_self_attn=False,
                            video_length=temporal_length, image_cross_attention=self.image_cross_attention,
                            image_cross_attention_scale_learnable=self.image_cross_attention_scale_learnable,
                        )
                    )
                    if self.temporal_attention:
                        layers.append(
                            TemporalTransformer(ch, num_heads, dim_head,
                                depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
                                use_checkpoint=use_checkpoint, only_self_att=temporal_self_att_only,
                                causal_attention=use_causal_attention, relative_position=use_relative_position,
                                temporal_length=temporal_length
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))

            if level < len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch, time_embed_dim, dropout,
                            out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                ds *= 2

    def forward(
        self,
        noisy_latents,
        timesteps,
        context_text,
        context_img=None,
        fps=None,
        layer_latents=None,     # [b, n_layer, t, c, h, w]
        layer_latent_mask=None, # [b, n_layer, t, 1, h, w]
        motion_scores=None,     # [b, n_layer]
        sketch=None,            # [b, n_layer, t, c, h, w]
        trajectory=None,        # [b, n_layer, t, c, h, w]
    ):
        if self.ignore_noisy_latents:
            noisy_latents_shape = list(noisy_latents.shape)
            noisy_latents_shape[1] = 0
            noisy_latents = torch.zeros(noisy_latents_shape, device=noisy_latents.device, dtype=noisy_latents.dtype)

        b, _, t, height, width = noisy_latents.shape
        n_layer = layer_latents.shape[1]
        t_emb = self.time_proj(timesteps).type(noisy_latents.dtype)
        emb = self.time_embed(t_emb)

        ## repeat t times for context [(b t) 77 768] & time embedding
        ## check if we use per-frame image conditioning
        if context_img is not None: ## decompose context into text and image
            context_text = repeat(context_text, 'b l c -> (b n t) l c', n=n_layer, t=t)
            context_img = repeat(context_img, 'b tl c -> b n tl c', n=n_layer)
            context_img = rearrange(context_img, 'b n (t l) c -> (b n t) l c', t=t)
            context = torch.cat([context_text, context_img], dim=1)
        else:
            context = repeat(context_text, 'b l c -> (b n t) l c', n=n_layer, t=t)
        emb = repeat(emb, 'b c -> (b n t) c', n=n_layer, t=t)

        ## always in shape (b n t) c h w, except for temporal layer
        noisy_latents = repeat(noisy_latents, 'b c t h w -> (b n t) c h w', n=n_layer)

        ## combine emb
        if self.fps_condition:
            if fps is None:
                fps = torch.tensor(
                    [self.default_fs] * b, dtype=torch.long, device=noisy_latents.device)
            fps_emb = self.time_proj(fps).type(noisy_latents.dtype)

            fps_embed = self.fps_embedding(fps_emb)
            fps_embed = repeat(fps_embed, 'b c -> (b n t) c', n=n_layer, t=t)
            emb = emb + fps_embed

        ## process conditions
        layer_condition = torch.cat([layer_latents, layer_latent_mask], dim=3)
        layer_condition = rearrange(layer_condition, 'b n t c h w -> (b n t) c h w')
        h = torch.cat([noisy_latents, layer_condition], dim=1)

        if "motion_score" in self.condition_channels:
            motion_condition = repeat(motion_scores, 'b n -> b n t 1 h w', t=t, h=height, w=width)
            motion_condition = torch.cat([motion_condition, layer_latent_mask], dim=3)
            motion_condition = rearrange(motion_condition, 'b n t c h w -> (b n t) c h w')
            motion_condition = self.motion_embedding(motion_condition)
            if self.control_injection_mode == 'concat':
                h = torch.cat([h, motion_condition], dim=1)

        if "sketch" in self.condition_channels:
            sketch_condition = rearrange(sketch, 'b n t c h w -> (b n t) c h w')
            sketch_condition = self.sketch_embedding(sketch_condition)
            if self.control_injection_mode == 'concat':
                h = torch.cat([h, sketch_condition], dim=1)

        if "trajectory" in self.condition_channels:
            traj_condition = rearrange(trajectory, 'b n t c h w -> (b n t) c h w')
            traj_condition = self.trajectory_embedding(traj_condition)
            if self.control_injection_mode == 'concat':
                h = torch.cat([h, traj_condition], dim=1)

        layer_features = []
        for id, module in enumerate(self.input_blocks):
            h = module(h, emb, context=context, batch_size=b*n_layer)
            if id == 0:
                if self.control_injection_mode == 'add':
                    if "motion_score" in self.condition_channels:
                        h = h + motion_condition
                    if "sketch" in self.condition_channels:
                        h = h + sketch_condition
                    if "trajectory" in self.condition_channels:
                        h = h + traj_condition
                if self.addition_attention:
                    h = self.init_attn(h, emb, context=context, batch_size=b*n_layer)
            if SpatialTransformer in [type(m) for m in module]:
                layer_features.append(rearrange(h, '(b n t) c h w -> b n t c h w', b=b, n=n_layer))

        return layer_features

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, layer_controlnet_additional_kwargs={}, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # load config
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            **kwargs,
        )

        for key, value in layer_controlnet_additional_kwargs.items():
            if isinstance(value, (ListConfig, DictConfig)):
                config[key] = OmegaConf.to_container(value, resolve=True)
            else:
                config[key] = value

        # load model
        model_file = None
        if use_safetensors:
            try:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    commit_hash=commit_hash,
                )

            except IOError as e:
                logger.error(f"An error occurred while trying to fetch {pretrained_model_name_or_path}: {e}")
                if not allow_pickle:
                    raise
                logger.warning(
                    "Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead."
                )

        if model_file is None:
            model_file = _get_model_file(
                pretrained_model_name_or_path,
                weights_name=_add_variant(WEIGHTS_NAME, variant),
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                commit_hash=commit_hash,
            )

        model = cls.from_config(config, **unused_kwargs)
        state_dict = load_state_dict(model_file, variant)

        if state_dict['input_blocks.0.0.weight'].shape[1] != model.input_blocks[0][0].weight.shape[1]:
            state_dict.pop('input_blocks.0.0.weight')

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"LayerControlNet loaded from {model_file} with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys.")
        return model
