import math
import torch
import torch.nn.functional as F
from einops import rearrange

from comfy.ldm.modules.attention import optimized_attention
from comfy.model_patcher import ModelPatcher


def reshape_heads_to_batch_dim3(tensor, head_size):
    batch_size1, batch_size2, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size1, batch_size2,
                            seq_len, head_size, dim // head_size)
    tensor = tensor.permute(0, 3, 1, 2, 4)
    return tensor


def apply_flow(query,
               key,
               value,
               trajectories,
               extra_options):
    # TODO: Hardcoded for SD1.5
    height = trajectories['height']//8
    width = trajectories['width']//8
    n_heads = extra_options['n_heads']
    cond_size = len(extra_options['cond_or_uncond'])
    video_length = len(query) // cond_size

    ad_params = extra_options.get('ad_params', {})
    sub_idxs = ad_params.get('sub_idxs', None)
    idx = 0
    if sub_idxs is not None:
        idx = sub_idxs[0]

    traj_window = trajectories['trajectory_windows'][idx]
    trajs = traj_window[f'traj{height}']
    traj_mask = traj_window[f'mask{height}']

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
                     b=cond_size,  h=height, w=width)
    _value = rearrange(value, '(b f) (h w) d -> b f h w d',
                       b=cond_size, h=height, w=width)
    key_tempo = _key[:, t_inds, x_inds, y_inds]
    value_tempo = _value[:, t_inds, x_inds, y_inds]
    key_tempo = rearrange(key_tempo, 'b f n l d -> (b f) n l d')
    value_tempo = rearrange(value_tempo, 'b f n l d -> (b f) n l d')

    traj_mask = rearrange(torch.stack(
        [traj_mask] * cond_size),  'b f n l -> (b f) n l')
    traj_mask = traj_mask[:, None].repeat(
        1, n_heads, 1, 1).unsqueeze(-2)
    attn_bias = torch.zeros_like(
        traj_mask, dtype=key_tempo.dtype, device=query.device)  # regular zeros_like
    attn_bias[~traj_mask] = -torch.inf

    # flow attention
    query_tempo = reshape_heads_to_batch_dim3(query_tempo, n_heads)
    key_tempo = reshape_heads_to_batch_dim3(key_tempo, n_heads)
    value_tempo = reshape_heads_to_batch_dim3(value_tempo, n_heads)

    attn_matrix2 = query_tempo @ key_tempo.transpose(-2, -1) / math.sqrt(
        query_tempo.size(-1)) + attn_bias
    attn_matrix2 = F.softmax(attn_matrix2, dim=-1)
    out = (attn_matrix2@value_tempo).squeeze(-2)

    hidden_states = rearrange(out, 'b k r d -> b r (k d)')

    return hidden_states


def get_flatten_attention(trajectories, use_old_qk=False):
    def flatten_attention(q, k, v, extra_options):
        n_heads = extra_options['n_heads']

        hidden_states = optimized_attention(q, k, v, n_heads, mask=None)

        _, hw, _ = q.shape
        # TODO: Hardcoded for SD1.5
        target_height = trajectories['height']//8
        target_width = trajectories['width']//8

        if target_height * target_width == hw:
            if use_old_qk is True:
                query = q
                key = k
            else:
                query = hidden_states
                key = hidden_states
            hidden_states = apply_flow(
                query,
                key,
                hidden_states,
                trajectories,
                extra_options
            )

        return hidden_states
    return flatten_attention


class ApplyFlattenAttentionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"model": ("MODEL",),
                 "trajectories": ("TRAJECTORY",),
                 "use_old_qk": ("BOOLEAN", {"default": False}),
                 "input_attn_1": ("BOOLEAN", {"default": True}),
                 "input_attn_2": ("BOOLEAN", {"default": True}),
                 "output_attn_9": ("BOOLEAN", {"default": True}),
                 "output_attn_10": ("BOOLEAN", {"default": True}),
                 "output_attn_11": ("BOOLEAN", {"default": True}),
                 }
                }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"

    CATEGORY = "flatten"

    def apply(self, model, trajectories, use_old_qk,
              input_attn_1, input_attn_2, output_attn_9, output_attn_10, output_attn_11):
        model: ModelPatcher = model.clone()

        # TODO: Hardcoded for SD1.5
        attn = get_flatten_attention(trajectories, use_old_qk)
        if input_attn_1:
            model.set_model_patch_replace(attn, 'attn1',
                                          'input', 1)
        if input_attn_2:
            model.set_model_patch_replace(attn, 'attn1',
                                          'input', 2)
        if output_attn_9:
            model.set_model_patch_replace(attn, 'attn1',
                                          'output', 9)
        if output_attn_10:
            model.set_model_patch_replace(attn, 'attn1',
                                          'output', 10)
        if output_attn_11:
            model.set_model_patch_replace(attn, 'attn1',
                                          'output', 11)

        return (model, )
