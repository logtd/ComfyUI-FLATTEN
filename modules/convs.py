# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class InflatedConv3d(nn.Conv2d):

    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x
