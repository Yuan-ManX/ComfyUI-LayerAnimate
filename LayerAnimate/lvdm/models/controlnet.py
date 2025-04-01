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


class ResBlock_v2(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        use_checkpoint=False,
        use_conv=False,
        up=False,
        down=False,
        use_temporal_conv=False,
        tempspatial_aware=False
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_temporal_conv = use_temporal_conv

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            zero_module(conv_nd(dims, channels, self.out_channels, 3, padding=1)),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock(
                self.out_channels,
                self.out_channels,
                dropout=0.1,
                spatial_aware=tempspatial_aware
            )

    def forward(self, x, batch_size=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        input_tuple = (x, )
        if batch_size:
            forward_batchsize = partial(self._forward, batch_size=batch_size)
            return checkpoint(forward_batchsize, input_tuple, self.parameters(), self.use_checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(), self.use_checkpoint)

    def _forward(self, x, batch_size=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv and batch_size:
            h = rearrange(h, '(b t) c h w -> b c t h w', b=batch_size)
            h = self.temopral_conv(h)
            h = rearrange(h, 'b c t h w -> (b t) c h w')
        return h


class TrajectoryEncoder(nn.Module):
    def __init__(self, cin, time_embed_dim, channels=[320, 640, 1280, 1280], nums_rb=3,
                 dropout=0.0, use_checkpoint=False, tempspatial_aware=False, temporal_conv=False):
        super(TrajectoryEncoder, self).__init__()
        # self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        # self.conv_out = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResBlock_v2(channels[i - 1], time_embed_dim, dropout,
                            out_channels=channels[i], dims=2, use_checkpoint=use_checkpoint,
                            tempspatial_aware=tempspatial_aware,
                            use_temporal_conv=temporal_conv,
                            down=True
                        )
                    )
                else:
                    self.body.append(
                        ResBlock_v2(channels[i], time_embed_dim, dropout,
                            out_channels=channels[i], dims=2, use_checkpoint=use_checkpoint,
                            tempspatial_aware=tempspatial_aware,
                            use_temporal_conv=temporal_conv,
                            down=False
                        )
                    )
        self.body.append(
            ResBlock_v2(channels[-1], time_embed_dim, dropout,
                out_channels=channels[-1], dims=2, use_checkpoint=use_checkpoint,
                tempspatial_aware=tempspatial_aware,
                use_temporal_conv=temporal_conv,
                down=True
            )
        )
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)
        self.conv_out = zero_module(conv_nd(2, channels[-1], channels[-1], 3, 1, 1))

    def forward(self, x, batch_size=None):
        # unshuffle
        # x = self.unshuffle(x)
        # extract features
        # features = []
        x = self.conv_in(x)
        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x, batch_size)
        x = self.body[-1](x, batch_size)
        out = self.conv_out(x)
        return out


class ControlNet(ModelMixin, ConfigMixin):
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
        conditioning_channels=4,
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

        ## Time embedding blocks
        self.time_proj = Timesteps(model_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embed = TimestepEmbedding(model_channels, time_embed_dim)

        if fps_condition:
            self.fps_embedding = TimestepEmbedding(model_channels, time_embed_dim)
            nn.init.zeros_(self.fps_embedding.linear_2.weight)
            nn.init.zeros_(self.fps_embedding.linear_2.bias)

        # self.cond_embedding = TrajectoryEncoder(
        #         cin=conditioning_channels, time_embed_dim=time_embed_dim, channels=trajectory_channels, nums_rb=3,
        #         dropout=dropout, use_checkpoint=use_checkpoint, tempspatial_aware=tempspatial_aware, temporal_conv=False
        #     )
        self.cond_embedding = zero_module(conv_nd(dims, conditioning_channels, model_channels, 3, padding=1))
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))
            ]
        )

        ## Output Block
        self.downsample_output = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(32, model_channels),
                    nn.SiLU(),
                    zero_module(conv_nd(dims, model_channels, model_channels, 3, padding=1))
                )
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
                self.downsample_output.append(
                    nn.Sequential(
                        nn.GroupNorm(32, ch),
                        nn.SiLU(),
                        zero_module(conv_nd(dims, ch, ch, 3, padding=1))
                    )
                )
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
                self.downsample_output.append(
                    nn.Sequential(
                        nn.GroupNorm(32, out_ch),
                        nn.SiLU(),
                        zero_module(conv_nd(dims, out_ch, out_ch, 3, padding=1))
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
        condition=None,            # [b, t, c, h, w]
    ):
        if self.ignore_noisy_latents:
            noisy_latents = torch.zeros_like(noisy_latents)

        b, _, t, height, width = noisy_latents.shape
        t_emb = self.time_proj(timesteps).type(noisy_latents.dtype)
        emb = self.time_embed(t_emb)

        ## repeat t times for context [(b t) 77 768] & time embedding
        ## check if we use per-frame image conditioning
        if context_img is not None: ## decompose context into text and image
            context_text = context_text.repeat_interleave(repeats=t, dim=0)
            context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
            context = torch.cat([context_text, context_img], dim=1)
        else:
            context = context_text.repeat_interleave(repeats=t, dim=0)
        emb = emb.repeat_interleave(repeats=t, dim=0)

        ## always in shape (b n t) c h w, except for temporal layer
        noisy_latents = rearrange(noisy_latents, 'b c t h w -> (b t) c h w')
        condition = rearrange(condition, 'b t c h w -> (b t) c h w')

        ## combine emb
        if self.fps_condition:
            if fps is None:
                fps = torch.tensor(
                    [self.default_fs] * b, dtype=torch.long, device=noisy_latents.device)
            fps_emb = self.time_proj(fps).type(noisy_latents.dtype)

            fps_embed = self.fps_embedding(fps_emb)
            fps_embed = fps_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + fps_embed

        h = noisy_latents.type(self.dtype)
        hs = []
        for id, module in enumerate(self.input_blocks):
            h = module(h, emb, context=context, batch_size=b)
            if id == 0:
                h = h + self.cond_embedding(condition)
                if self.addition_attention:
                    h = self.init_attn(h, emb, context=context, batch_size=b)
            hs.append(h)

        guidance_feature_list = []
        for hidden, module in zip(hs, self.downsample_output):
            h = module(hidden)
            guidance_feature_list.append(h)

        return guidance_feature_list

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, layer_encoder_additional_kwargs={}, **kwargs):
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

        for key, value in layer_encoder_additional_kwargs.items():
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

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Controlnet loaded from {model_file} with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys.")
        return model
