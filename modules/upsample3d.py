import torch
import torch.nn as nn
import torch.nn.functional as F

from .convs import InflatedConv3d


# DONE
class Upsample3D(nn.Module):
    def __init__(self,
                 channels,
                 use_conv=False,
                 dims=2,
                 out_channels=None,
                 padding=1,  # using this instead of hardcoded flatten value of 1
                 dtype=None,
                 device=None,
                 operations=None,
                 ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        conv = None
        if use_conv:
            self.conv = InflatedConv3d(
                self.channels, self.out_channels, 3, padding=padding).half()

            self.Conv2d_0 = conv

    def forward(self, hidden_states, output_shape=None):
        assert hidden_states.shape[1] == self.channels

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        shape = None
        if self.dims == 3:
            shape = [hidden_states.shape[2], hidden_states.shape[3]
                     * 2, hidden_states.shape[4] * 2]
            if output_shape is not None:
                shape[1] = output_shape[3]
                shape[2] = output_shape[4]
        else:
            shape = [hidden_states.shape[2] * 2, hidden_states.shape[3] * 2]
            if output_shape is not None:
                shape[0] = output_shape[2]
                shape[1] = output_shape[3]
        if shape is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=[
                                          1.0, 2.0, 2.0], mode="nearest")
        else:
            hidden_states = F.interpolate(
                hidden_states, size=shape, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states
