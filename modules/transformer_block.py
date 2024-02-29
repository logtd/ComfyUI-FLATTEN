from einops import rearrange, repeat
from .fully_attention import FullyFrameAttention
from .patch3d import apply_patch3d
from comfy.ldm.modules.attention import FeedForward, CrossAttention
import torch
import torch.nn as nn

import comfy.ops
ops = comfy.ops.disable_weight_init


# DONE
class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        ff_in=False,
        inner_dim=None,
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
        dtype=None,
        device=None,
        operations=ops,
        num_embeds_ada_norm=None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
    ):
        super().__init__()
        # comfy setup
        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim
        self.is_res = inner_dim == dim

        # flatten setup
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        # self.norm1 = nn.LayerNorm(dim)

        if self.ff_in:
            self.norm_in = operations.LayerNorm(
                dim, dtype=dtype, device=device)
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout,
                                     glu=gated_ff, dtype=dtype, device=device, operations=operations)

        self.disable_self_attn = disable_self_attn
        # Fully
        self.attn1 = FullyFrameAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            bias=attention_bias,
            context_dim=context_dim if self.disable_self_attn else None,
            dtype=dtype,
            device=device,
        )
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout,
                              glu=gated_ff, dtype=dtype, device=device, operations=operations)

        # Cross-Attn
        # if context_dim is not None:
        #     self.attn2 = CrossAttention(
        #         query_dim=dim,
        #         context_dim=context_dim,
        #         heads=n_heads,
        #         dim_head=d_head,
        #         dropout=dropout,
        #         bias=attention_bias,
        #         upcast_attention=upcast_attention,
        #     )
        # else:
        #     self.attn2 = None
        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.norm2 = None
        else:
            context_dim_attn2 = None
            if not switch_temporal_ca_to_sa:
                context_dim_attn2 = context_dim

            self.attn2 = CrossAttention(query_dim=inner_dim, context_dim=context_dim_attn2,
                                        heads=n_heads, dim_head=d_head, dropout=dropout, dtype=dtype, device=device, operations=operations)  # is self-attn if context is none
            self.norm2 = operations.LayerNorm(
                inner_dim, dtype=dtype, device=device)

        self.norm1 = operations.LayerNorm(
            inner_dim, dtype=dtype, device=device)
        self.norm3 = operations.LayerNorm(
            inner_dim, dtype=dtype, device=device)
        self.checkpoint = checkpoint
        self.n_heads = n_heads
        self.d_head = d_head
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

    def forward(
            self,
            x,
            context=None,
            transformer_options={},
            attention_mask=None,  # VALIDATE that this gets sent to the block via transformer_options
            video_length=None,  # VALIDATE that this gets sent to the block via transformer_options
            inter_frame=False,  # VALIDATE that this gets sent to the block via transformer_options
            trajs_dict=None
    ):
        # Comfy setup
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None

        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head

        hidden_states = x
        encoder_hidden_states = context
        norm_hidden_states = self.norm1(hidden_states)

        if self.only_cross_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask,
                           inter_frame=inter_frame, trajs_dict=trajs_dict) + hidden_states
            )
        else:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask,
                                       video_length=video_length, inter_frame=inter_frame, trajs_dict=trajs_dict) + hidden_states

        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                hidden_states = p(hidden_states, extra_options)

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = self.norm2(hidden_states)
            # switch cross attention to self attention
            if self.switch_temporal_ca_to_sa:
                context_attn2 = norm_hidden_states
            else:
                context_attn2 = encoder_hidden_states

            value_attn2 = None
            if "attn2_patch" in transformer_patches:
                patch = transformer_patches["attn2_patch"]
                value_attn2 = context_attn2
                for p in patch:
                    norm_hidden_states, context_attn2, value_attn2 = p(
                        norm_hidden_states, context_attn2, value_attn2, extra_options)

            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block

            if block_attn2 is not None and block_attn2 in attn2_replace_patch:
                if value_attn2 is None:
                    value_attn2 = context_attn2
                norm_hidden_states = self.attn2.to_q(norm_hidden_states)
                context_attn2 = self.attn2.to_k(context_attn2)
                value_attn2 = self.attn2.to_v(value_attn2)
                attn2_hidden_states = attn2_replace_patch[block_attn2](
                    norm_hidden_states, context_attn2, value_attn2, extra_options)
                hidden_states = self.attn2.to_out(
                    attn2_hidden_states) + hidden_states
            else:
                # Flatten adds the hidden states here and after the feed-forward
                hidden_states = self.attn2(
                    norm_hidden_states, context=context_attn2, mask=attention_mask) + hidden_states

        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                norm_hidden_states = p(norm_hidden_states, extra_options)

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states

    def _forward(
            self,
            x,
            context=None,
            transformer_options={},
            attention_mask=None,  # VALIDATE that this gets sent to the block via transformer_options
            video_length=None,  # VALIDATE that this gets sent to the block via transformer_options
            inter_frame=False,  # VALIDATE that this gets sent to the block via transformer_options
            trajs_dict=None
    ):
        # Comfy setup
        extra_options = {}
        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)
        transformer_patches = {}
        transformer_patches_replace = {}

        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head

        # Comfy ff_in
        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        # SparseCausal-Attention
        n = self.norm1(x)
        if self.disable_self_attn:
            context_attn1 = context
        else:
            context_attn1 = None
        value_attn1 = None

        if "attn1_patch" in transformer_patches:
            patch = transformer_patches["attn1_patch"]
            if context_attn1 is None:
                context_attn1 = n
            value_attn1 = context_attn1
            for p in patch:
                n, context_attn1, value_attn1 = p(
                    n, context_attn1, value_attn1, extra_options)

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None

        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        # attn1
        # if self.only_cross_attention:
        #     x = (
        #         self.attn1(n, context, transformer_options=transformer_options, inter_frame=inter_frame, **kwargs) + x
        #     )
        # else:
        #     x = self.attn1(n, attention_mask=attention_mask, video_length=video_length, inter_frame=inter_frame, **kwargs) + x
        if block_attn1 in attn1_replace_patch:
            if context_attn1 is None:
                context_attn1 = n
                value_attn1 = n
            n = self.attn1.to_q(n)
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            n = attn1_replace_patch[block_attn1](
                n, context_attn1, value_attn1, extra_options)
            n = self.attn1.to_out(n)
        else:
            n = self.attn1(n, context=context_attn1,
                           value=value_attn1,
                           video_length=video_length,
                           inter_frame=inter_frame,
                           attention_mask=attention_mask,
                           trajs_dict=trajs_dict)

        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                n = apply_patch3d(n, extra_options, patch)

        if self.attn2 is not None:
            # 3 -- norm2
            n = self.norm2(n)
            if self.switch_temporal_ca_to_sa:
                context_attn2 = n
            else:
                context_attn2 = context
            value_attn2 = None
            if "attn2_patch" in transformer_patches:
                patch = transformer_patches["attn2_patch"]
                value_attn2 = context_attn2
                for p in patch:
                    n, context_attn2, value_attn2 = p(
                        n, context_attn2, value_attn2, extra_options)

            attn2_replace_patch = transformer_patches_replace.get("attn2", {})
            block_attn2 = transformer_block
            if block_attn2 not in attn2_replace_patch:
                block_attn2 = block

            # 4 -- attn2
            if block_attn2 in attn2_replace_patch:
                if value_attn2 is None:
                    value_attn2 = context_attn2
                n = self.attn2.to_q(n)
                context_attn2 = self.attn2.to_k(context_attn2)
                value_attn2 = self.attn2.to_v(value_attn2)
                n = attn2_replace_patch[block_attn2](
                    n, context_attn2, value_attn2, extra_options)
                n = self.attn2.to_out(n)
            else:
                n = self.attn2(n, context=context_attn2, value=value_attn2)

        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        # x += n
        # if self.is_res:
        #     x_skip = x

        # # 5 -- norm3/ff
        # x = self.ff(self.norm3(x))
        # if self.is_res:
        #     x += x_skip
        n = self.ff(self.norm3(n)) + n

        return n
