# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
ops = comfy.ops.disable_weight_init

from .convs import InflatedConv3d
from .upsample3d import Upsample3D
from .downsample3d import Downsample3D


# DONE -- with quite a few assumptions
class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels, # =512,
        dropout, # =0.0,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False, # not in flatten
        dims=2, # not in flatten
        use_checkpoint=False, # not in flatten
        up=False, # not in flatten
        down=False, # not in flatten
        kernel_size=3, # not in flatten
        exchange_temb_dims=False, # not in flatten
        skip_t_emb=False, # not in flatten
        dtype=None, # not in flatten
        device=None, # not in flatten
        operations=ops, # not in flatten
        groups=32,  # not in comfy
        groups_out=None, # not in comfy
        pre_norm=True, # not in comfy
        eps=1e-6, # not in comfy
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

        
        # self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=channels, eps=eps, affine=True)
        # self.conv1 = InflatedConv3d(channels, out_channels, kernel_size=3, stride=1, padding=1)
        if isinstance(kernel_size, list):
            padding = [k // 2 for k in kernel_size]
        else:
            padding = kernel_size // 2
        # from comfy
        self.in_layers = nn.Sequential(
            torch.nn.GroupNorm(num_groups=groups, num_channels=channels, eps=eps, affine=True), # flatten
            nn.SiLU(), # comfy
            InflatedConv3d(channels, out_channels, kernel_size=3, stride=1, padding=padding)
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample3D(channels, False, dims, dtype=dtype, device=device)
            self.x_upd = Upsample3D(channels, False, dims, dtype=dtype, device=device)
        elif down:
            downsample_padding = 1 # VALIDATE
            self.h_upd = Downsample3D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
            self.x_upd = Downsample3D(channels, False, dims, dtype=dtype, device=device)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # if emb_channels is not None:
        #     if self.time_embedding_norm == "default":
        #         time_emb_proj_out_channels = out_channels
        #     elif self.time_embedding_norm == "scale_shift":
        #         time_emb_proj_out_channels = out_channels * 2
        #     else:
        #         raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

        #     self.time_emb_proj = torch.nn.Linear(emb_channels, time_emb_proj_out_channels)
        # else:
        #     self.time_emb_proj = None
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

        # self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        # self.dropout = torch.nn.Dropout(dropout)
        # self.conv2 = InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.out_layers = nn.Sequential(
            operations.GroupNorm(32, self.out_channels, dtype=dtype, device=device),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            # operations.conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device)
            InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype, device=device)
            ,
        )

        # covered in in_layers default to silu
        # if non_linearity == "swish":
        #     self.nonlinearity = lambda x: F.silu(x)
        # elif non_linearity == "mish":
        #     self.nonlinearity = Mish()
        # elif non_linearity == "silu":
        #     self.nonlinearity = nn.SiLU()

        # self.use_in_shortcut = self.channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        # self.conv_shortcut = None
        # if self.use_in_shortcut:
        #     self.conv_shortcut = InflatedConv3d(channels, out_channels, kernel_size=1, stride=1, padding=0)

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            # self.skip_connection = operations.conv_nd(
            #     dims, channels, self.out_channels, kernel_size, padding=padding, dtype=dtype, device=device
            # )
            self.skip_connection = InflatedConv3d(channels, out_channels, kernel_size=kernel_size, padding=padding)
        else:
            # self.skip_connection = operations.conv_nd(dims, channels, self.out_channels, 1, dtype=dtype, device=device)
            self.skip_connection = InflatedConv3d(channels, out_channels, kernel_size=1) # validate 1 should be kernel_size
        
        # save features
        self.out_layers_features = None
        self.out_layers_inject_features = None

    def forward(self, x, temb):
        input_tensor = x
        # hidden_states = input_tensor

        # hidden_states = self.norm1(hidden_states)
        # hidden_states = self.nonlinearity(hidden_states)

        # hidden_states = self.conv1(hidden_states)
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)  # VALIDATE this changed x ... should it also change input_tensor?
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        # if temb is not None:
        #     temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None] # VALIDATE [:, :,...] extra dimension here

        # if temb is not None and self.time_embedding_norm == "default":
        #     hidden_states = hidden_states + temb

        # hidden_states = self.norm2(hidden_states)

        # if temb is not None and self.use_scale_shift_norm:
        #     scale, shift = torch.chunk(temb, 2, dim=1)
        #     hidden_states = hidden_states * (1 + scale) + shift

        # hidden_states = self.nonlinearity(hidden_states)

        # hidden_states = self.dropout(hidden_states)
        # hidden_states = self.conv2(hidden_states)
            
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
                h = h + emb_out
            h = self.out_layers(h)

        if self.skip_connection is not None:
            x = self.skip_connection(x)  # VALIDATE This was changed to x instead of input_tensor

        # save features  -- VALIDATE that nothing was skipped above
        self.out_layers_features = h
        if self.out_layers_inject_features is not None:
            h = self.out_layers_inject_features 

        output_tensor = input_tensor + h

        return output_tensor
