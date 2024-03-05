from comfy.ldm.util import exists
from .transformer_block import BasicTransformerBlock
from einops import rearrange, repeat
import torch
import torch.nn as nn

import comfy.ops
ops = comfy.ops.disable_weight_init


class Transformer3DModel(nn.Module):
    def __init__(self,
                 in_channels,
                 n_heads,
                 d_head: int = 88,
                 depth=1,
                 dropout=0.,
                 context_dim=None,
                 disable_self_attn=False,
                 use_linear=False,
                 use_checkpoint=True,
                 dtype=None,
                 device=None,
                 operations=ops,
                 ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = operations.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)
        if not use_linear:
            self.proj_in = operations.Conv2d(in_channels,
                                             inner_dim,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0, dtype=dtype, device=device)
        else:
            self.proj_in = operations.Linear(
                in_channels, inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, dtype=dtype, device=device, operations=operations)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = operations.Conv2d(inner_dim, in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0, dtype=dtype, device=device)
        else:
            self.proj_out = operations.Linear(
                in_channels, inner_dim, dtype=dtype, device=device)
        self.use_linear = use_linear

    def forward(self, hidden_states, context=None, transformer_options={}):
        # Input

        assert hidden_states.dim(
        ) == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        inter_frame = False
        if 'flatten' in transformer_options and 'inter_frame' in transformer_options["flatten"]:
            inter_frame = transformer_options["flatten"]['inter_frame']
        video_length = hidden_states.shape[2]
        cond_size = hidden_states.shape[0]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        context = repeat(
            context, 'b n c -> (b f) n c', f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        # check resolution

        resolu = hidden_states.shape[-2]  # height
        height = resolu
        width = hidden_states.shape[-1]
        traj_options = {"resolution": resolu,
                        "cond_size": cond_size, "height": height, "width": width}

        hidden_states = self.norm(hidden_states)
        if not self.use_linear:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(
                0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(
                0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                context=context,
                video_length=video_length,
                inter_frame=inter_frame,
                transformer_options=transformer_options,
                traj_options=traj_options
            )

        # Output
        if not self.use_linear:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(
                    0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(
                    0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)

        return output
