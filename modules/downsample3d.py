from .convs import InflatedConv3d
import torch.nn as nn

import comfy.ops
ops = comfy.ops.disable_weight_init


# DONE
class Downsample3D(nn.Module):
    def __init__(self,
                 channels,
                 use_conv=False,
                 dims=2,  # not in flatten
                 out_channels=None,
                 padding=1,
                 dtype=None,  # not in flatten
                 device=None,  # not in flatten
                 operations=ops,  # not in flatten
                 ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)  # was always 2

        if use_conv:
            self.op = InflatedConv3d(
                self.channels, self.out_channels, 3, stride=stride, padding=padding).half()
        else:
            raise NotImplementedError

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            raise NotImplementedError

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.op(hidden_states)

        return hidden_states
