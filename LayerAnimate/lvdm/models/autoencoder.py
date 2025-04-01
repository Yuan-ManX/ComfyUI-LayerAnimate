import os
from functools import partial
from dataclasses import dataclass

import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.utils import BaseOutput

from ..modules.ae_modules import Encoder, Decoder
from ..modules.ae_dualref_modules import VideoDecoder
from ..utils import instantiate_from_config


@dataclass
class DecoderOutput(BaseOutput):
    """
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Decoded output sample of the model. Output of the last layer of the model.
    """

    sample: torch.FloatTensor


@dataclass
class AutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: "DiagonalGaussianDistribution"


class AutoencoderKL(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 image_key="image",
                 input_dim=4,
                 use_checkpoint=False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.use_checkpoint = use_checkpoint

    def encode(self, x, return_hidden_states=False, **kwargs):
        if return_hidden_states:
            h, hidden = self.encoder(x, return_hidden_states)
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            return AutoencoderKLOutput(latent_dist=posterior), hidden
        else:
            h = self.encoder(x)
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z, **kwargs):
        if len(kwargs) == 0: ## use the original decoder in AutoencoderKL
            z = self.post_quant_conv(z)
        dec = self.decoder(z, **kwargs)  ##change for SVD decoder by adding **kwargs
        return dec

    def forward(self, input, sample_posterior=True, **additional_decode_kwargs):
        input_tuple = (input, )
        forward_temp = partial(self._forward, sample_posterior=sample_posterior, **additional_decode_kwargs)
        return checkpoint(forward_temp, input_tuple, self.parameters(), self.use_checkpoint)


    def _forward(self, input, sample_posterior=True, **additional_decode_kwargs):
        posterior = self.encode(input)[0]
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, **additional_decode_kwargs)
        ## print(input.shape, dec.shape) torch.Size([16, 3, 256, 256]) torch.Size([16, 3, 256, 256])
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if x.dim() == 5 and self.input_dim == 4:
            b,c,t,h,w = x.shape
            self.b = b
            self.t = t
            x = rearrange(x, 'b c t h w -> (b t) c h w')

        return x

    def get_last_layer(self):
        return self.decoder.conv_out.weight


class AutoencoderKL_Dualref(AutoencoderKL):
    @register_to_config
    def __init__(self,
                 ddconfig,
                 embed_dim,
                 image_key="image",
                 input_dim=4,
                 use_checkpoint=False,
                 ):
        super().__init__(ddconfig, embed_dim, image_key, input_dim, use_checkpoint)
        self.decoder = VideoDecoder(**ddconfig)

    def _forward(self, input, batch_size, sample_posterior=True, **additional_decode_kwargs):
        posterior, hidden_states = self.encode(input, return_hidden_states=True)

        hidden_states_first_last = []
        ### use only the first and last hidden states
        for hid in hidden_states:
            hid = rearrange(hid, '(b t) c h w -> b c t h w', b=batch_size)
            hid_new = torch.cat([hid[:, :, 0:1], hid[:, :, -1:]], dim=2)
            hidden_states_first_last.append(hid_new)

        if sample_posterior:
            z = posterior[0].sample()
        else:
            z = posterior[0].mode()
        dec = self.decode(z, ref_context=hidden_states_first_last, **additional_decode_kwargs)
        ## print(input.shape, dec.shape) torch.Size([16, 3, 256, 256]) torch.Size([16, 3, 256, 256])
        return dec, posterior
