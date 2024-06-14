# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py
from comfy.ldm.modules.attention import optimized_attention, optimized_attention_masked, attention_basic
import torch
import torch.nn
import torch.nn.functional as F

from typing import Optional
import math
from torch import nn
from einops import rearrange

import comfy.model_management
if comfy.model_management.xformers_enabled():
    import xformers
    import xformers.ops
from comfy.cli_args import args
import comfy.ops
ops = comfy.ops.disable_weight_init


if args.dont_upcast_attention:
    print("disabling upcasting of attention")
    _ATTN_PRECISION = "fp16"
else:
    _ATTN_PRECISION = "fp32"


class FullyFrameAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        context_dim (`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        dtype=None,
        device=None,
        operations=ops,  # is ops in original CrossAttention module
        # Flatten params
        bias=False,
        norm_num_groups: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.upcast_attention = _ATTN_PRECISION == 'fp32'  # upcast_attention

        self.scale = dim_head**-0.5

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = True

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(
                num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = operations.Linear(
            query_dim, inner_dim, bias=bias, dtype=dtype, device=device)
        self.to_k = operations.Linear(
            context_dim, inner_dim, bias=bias, dtype=dtype, device=device)
        self.to_v = operations.Linear(
            context_dim, inner_dim, bias=bias, dtype=dtype, device=device)

        self.to_out = nn.ModuleList([])
        self.to_out.append(operations.Linear(
            inner_dim, query_dim, dtype=dtype, device=device))
        self.to_out.append(nn.Dropout(dropout))

        self.q = None
        self.inject_q = None
        self.k = None
        self.inject_k = None

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len,
                                head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_heads_to_batch_dim3(self, tensor):
        batch_size1, batch_size2, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size1, batch_size2,
                                seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 3, 1, 2, 4)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size,
                                head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def _memory_efficient_attention_xformers(self, query, key, value, attention_mask):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _attention_mechanism(self, query, key, value, attention_mask):
        # Comfy default attention mechanism
        if attention_mask is not None:
            hidden_states = optimized_attention_masked(
                query, key, value, self.heads, attention_mask)
        else:
            hidden_states = optimized_attention(
                query, key, value, self.heads)
        return hidden_states

    def forward(self, hidden_states, context=None, value=None, attention_mask=None, video_length=None, inter_frame=False, transformer_options={}, traj_options={}):
        batch_size, sequence_length, _ = hidden_states.shape
        flatten_options = transformer_options['flatten']

        transformer_block = transformer_options.get('block', ('', -1))[0]
        transformer_index = transformer_options.get('transformer_index', -1)
        patches_replace = transformer_options.get('patches_replace', {})
        attn1_replace = patches_replace.get('attn1', {})
        block = (transformer_block, transformer_index)
        if block in attn1_replace:
            replace_fn = attn1_replace[block]
            hidden_states = replace_fn(
                self.to_q(hidden_states),
                self.to_k(hidden_states),
                self.to_v(hidden_states),
                extra_options=transformer_options
            )
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states

        h = traj_options['height']
        w = traj_options['width']
        target_resolution = flatten_options['input_shape'][-2]
        if self.group_norm is not None:
            hidden_states = self.group_norm(
                hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)  # (bf) x d(hw) x c
        self.q = query
        if self.inject_q is not None:
            query = self.inject_q

        query_old = None
        if flatten_options['old_qk'] == 1:
            query_old = query.clone()

        context = context if context is not None else hidden_states
        key = self.to_k(context)
        self.k = key
        if self.inject_k is not None:
            key = self.inject_k

        key_old = None
        if flatten_options['old_qk'] == 1:
            key_old = key.clone()
        value = self.to_v(context)

        query = rearrange(query, "(b f) d c -> b (f d) c", f=video_length)
        key = rearrange(key, "(b f) d c -> b (f d) c", f=video_length)
        value = rearrange(value, "(b f) d c -> b (f d) c", f=video_length)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(
                    attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(
                    self.heads, dim=0)

        hidden_states = self._attention_mechanism(
            query, key, value, attention_mask)
        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if h in [target_resolution]:
            hidden_states = rearrange(
                hidden_states, "b (f d) c -> (b f) d c", f=video_length)
            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)).transpose(1, 2)

            if flatten_options['old_qk'] == 1:
                query = query_old
                key = key_old
            else:
                query = hidden_states
                key = hidden_states
            value = hidden_states

            cond_size = traj_options['cond_size']
            resolu = traj_options['resolution']
            trajs = flatten_options['trajs'][f'traj{resolu}']
            traj_mask = flatten_options['trajs'][f'mask{resolu}']

            start = -video_length+1
            end = trajs.shape[2]

            traj_key_sequence_inds = torch.cat(
                [trajs[:, :, 0, :].unsqueeze(-2), trajs[:, :, start:end, :]], dim=-2)
            traj_mask = torch.cat([traj_mask[:, :, 0].unsqueeze(-1),
                                   traj_mask[:, :, start:end]], dim=-1)

            t_inds = traj_key_sequence_inds[:, :, :, 0]
            x_inds = traj_key_sequence_inds[:, :, :, 1]
            y_inds = traj_key_sequence_inds[:, :, :, 2]

            query_tempo = query.unsqueeze(-2)
            _key = rearrange(key, '(b f) (h w) d -> b f h w d',
                             b=int(batch_size/video_length), f=video_length, h=h, w=w)
            _value = rearrange(value, '(b f) (h w) d -> b f h w d',
                               b=int(batch_size/video_length), f=video_length, h=h, w=w)
            key_tempo = _key[:, t_inds, x_inds, y_inds]  # This fails
            value_tempo = _value[:, t_inds, x_inds, y_inds]
            key_tempo = rearrange(key_tempo, 'b f n l d -> (b f) n l d')
            value_tempo = rearrange(value_tempo, 'b f n l d -> (b f) n l d')

            traj_mask = rearrange(torch.stack(
                [traj_mask] * cond_size),  'b f n l -> (b f) n l')
            traj_mask = traj_mask[:, None].repeat(
                1, self.heads, 1, 1).unsqueeze(-2)
            attn_bias = torch.zeros_like(
                traj_mask, dtype=key_tempo.dtype, device=query.device)  # regular zeros_like
            attn_bias[~traj_mask] = -torch.inf

            # flow attention
            query_tempo = self.reshape_heads_to_batch_dim3(query_tempo)
            key_tempo = self.reshape_heads_to_batch_dim3(key_tempo)
            value_tempo = self.reshape_heads_to_batch_dim3(value_tempo)

            attn_matrix2 = query_tempo @ key_tempo.transpose(-2, -1) / math.sqrt(
                query_tempo.size(-1)) + attn_bias
            attn_matrix2 = F.softmax(attn_matrix2, dim=-1)
            out = (attn_matrix2@value_tempo).squeeze(-2)

            hidden_states = rearrange(out, '(b f) k (h w) d -> b (f h w) (k d)', b=int(
                batch_size/video_length), f=video_length, h=h, w=w)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)

        # All frames
        hidden_states = rearrange(
            hidden_states, "b (f d) c -> (b f) d c", f=video_length)

        return hidden_states
