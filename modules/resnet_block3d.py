# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py

from .downsample3d import Downsample3D
from .upsample3d import Upsample3D
from .convs import InflatedConv3d
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
ops = comfy.ops.disable_weight_init


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
        dtype=None,
        device=None,
        operations=ops,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.channels = channels
        out_channels = channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv
        # comfy setup
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        if groups_out is None:
            groups_out = groups

        if isinstance(kernel_size, list):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2
        self.in_layers = nn.Sequential(
            operations.GroupNorm(32, channels, dtype=dtype, device=device),
            nn.SiLU(),  # comfy
            InflatedConv3d(channels, out_channels, kernel_size=3,
                           stride=1, padding=padding).half()
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample3D(
                channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Upsample3D(
                channels, False, dims, dtype=dtype, device=device)
        elif down:
            downsample_padding = 1
            self.h_upd = Downsample3D(
                out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
            )
            self.x_upd = Downsample3D(
                channels, False, dims, dtype=dtype, device=device)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.skip_t_emb = skip_t_emb
        if self.skip_t_emb:
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                operations.Linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels, dtype=dtype, device=device
                ),
            )

        self.out_layers = nn.Sequential(
            operations.GroupNorm(32, self.out_channels,
                                 dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            InflatedConv3d(out_channels, out_channels, kernel_size=3,
                           stride=1, padding=1, dtype=dtype, device=device),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = InflatedConv3d(
                channels, out_channels, kernel_size=kernel_size, padding=padding).to(dtype)
        else:
            self.skip_connection = InflatedConv3d(channels, out_channels, kernel_size=1).half(
            ).to(dtype)

        # save features
        self.out_layers_features = None
        self.out_layers_inject_features = None

    def forward(self, x, temb, transformer_options={}, **kwargs):
        input_tensor = x
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb = temb
        emb_out = None
        if not self.skip_t_emb:
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_norm(h)
            if emb_out is not None:
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h *= (1 + scale)
                h += shift
            h = out_rest(h)
        else:
            if emb_out is not None:
                if self.exchange_temb_dims:
                    emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
                if emb_out.shape[0] != h.shape[0]:  # ControlNet Hack TODO
                    video_length = transformer_options['flatten']['original_shape'][0]
                    emb_out = rearrange(
                        emb_out, '(b f) t c h w -> b t (c f) h w', f=video_length)
                h = h + emb_out  # (2, 320, 10, 64, 64) + (2, 320, 1, 1, 1)
            h = self.out_layers(h)

        if self.skip_connection is not None:
            input_tensor = self.skip_connection(input_tensor)

        self.out_layers_features = h
        if self.out_layers_inject_features is not None:
            h = self.out_layers_inject_features

        output_tensor = input_tensor + h

        return output_tensor
