# taken from: https://github.com/lllyasviel/ControlNet
# and modified with Flatten modules
# Mostly an experiment

from einops import rearrange
import torch
import torch as th
import torch.nn as nn

import comfy.ops
from comfy.ldm.modules.diffusionmodules.util import (
    zero_module,
    timestep_embedding,
)
from comfy.ldm.util import exists

from .unet import TimestepEmbedSequential
from .convs import InflatedConv3d
from .transformer3d import Transformer3DModel
from .downsample3d import Downsample3D
from .resnet_block3d import ResnetBlock3D


class FlattenControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        dtype=torch.float32,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        # custom support for prediction of discrete ids into codebook of first stage vq model
        n_embed=None,
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        transformer_depth_middle=None,
        transformer_depth_output=None,
        device=None,
        operations=comfy.ops.disable_weight_init,
        **kwargs,
    ):
        super().__init__()
        assert use_spatial_transformer == True, "use_spatial_transformer has to be true"
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            # from omegaconf.listconfig import ListConfig
            # if type(context_dim) == ListConfig:
            #     context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(
                len(num_attention_blocks))))

        transformer_depth = transformer_depth[:]

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            operations.Linear(model_channels, time_embed_dim,
                              dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.Linear(time_embed_dim, time_embed_dim,
                              dtype=self.dtype, device=device),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        operations.Linear(
                            adm_in_channels, time_embed_dim, dtype=self.dtype, device=device),
                        nn.SiLU(),
                        operations.Linear(
                            time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    InflatedConv3d(in_channels, model_channels,
                                   3, padding=1).half()
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(
            model_channels, operations=operations, dtype=self.dtype, device=device)])

        self.input_hint_block = TimestepEmbedSequential(
            InflatedConv3d(hint_channels, 16, 3, padding=1).half(),
            nn.SiLU(),
            InflatedConv3d(16, 16, 3, padding=1).half(),
            nn.SiLU(),
            InflatedConv3d(16, 32, 3, padding=1, stride=2).half(),
            nn.SiLU(),
            InflatedConv3d(32, 32, 3, padding=1).half(),
            nn.SiLU(),
            InflatedConv3d(32, 96, 3, padding=1, stride=2).half(),
            nn.SiLU(),
            InflatedConv3d(96, 96, 3, padding=1).half(),
            nn.SiLU(),
            InflatedConv3d(96, 256, 3, padding=1, stride=2).half(),
            nn.SiLU(),
            InflatedConv3d(256, model_channels, 3, padding=1).half(),
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResnetBlock3D(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    )
                ]
                ch = mult * model_channels
                num_transformers = transformer_depth.pop(0)
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            Transformer3DModel(
                                ch, num_heads, dim_head, depth=num_transformers, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint, dtype=self.dtype, device=device, operations=operations
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(
                    ch, operations=operations, dtype=self.dtype, device=device))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResnetBlock3D(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=self.dtype,
                            device=device,
                            operations=operations
                        )
                        if resblock_updown
                        else Downsample3D(
                            ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype, device=device, operations=operations
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(
                    ch, operations=operations, dtype=self.dtype, device=device))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        mid_block = [
            ResnetBlock3D(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                device=device,
                operations=operations
            )]
        if transformer_depth_middle >= 0:
            mid_block += [Transformer3DModel(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint, dtype=self.dtype, device=device, operations=operations
            ),
                ResnetBlock3D(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                device=device,
                operations=operations
            )]
        self.middle_block = TimestepEmbedSequential(*mid_block)
        self.middle_block_out = self.make_zero_conv(
            ch, operations=operations, dtype=self.dtype, device=device)
        self._feature_size += ch
        self.injection = {}
        self.timestep = -1
        self.step = 0

    def make_zero_conv(self, channels, operations=None, dtype=None, device=None):
        return TimestepEmbedSequential(InflatedConv3d(channels, channels, 1, padding=0).half())

    def forward(self, x, hint, timesteps, context, y=None, model_options={}, batched_number=1, **kwargs):
        transformer_options = model_options['transformer_options']
        original_shape = transformer_options['flatten']['original_shape']
        video_length = original_shape[0]

        stage = transformer_options['flatten'].get('stage', None)
        injection_steps = transformer_options['flatten'].get(
            'injection_steps', 0)
        step = None
        if stage == 'inversion' or stage is None:
            self.injection = {
                'input': {},
                'middle': {}
            }
            for i in range(len(self.input_blocks)):
                self.injection['input'][i] = {}
        elif stage == 'sampling':
            if int(timesteps[0]) > self.timestep:
                self.timestep = int(timesteps[0])
                self.step = 0
            elif int(timesteps[0]) < self.timestep:
                self.timestep = int(timesteps[0])
                self.step += 1
            step = self.step

        incoming_shape = x.shape
        if len(incoming_shape) == 4:
            # TODO get video_length in here
            x = rearrange(x, '(b f) c h w -> b c f h w', f=video_length)
        if len(hint.shape) == 4:
            hint = rearrange(hint, '(b f) c h w -> b c f h w', f=video_length)

        cond_length = x.shape[0]
        t_emb = timestep_embedding(
            timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        hs = []
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        i = 0
        for k, attn in enumerate(self.named_modules()):
            if hasattr(attn, 'inject_q') and hasattr(attn, 'inject_k'):
                attn.inject_q = None
                attn.inject_k = None

        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if stage == 'sampling':
                for k, attn in enumerate(self.named_modules()):
                    if hasattr(attn, 'inject_q') and hasattr(attn, 'inject_k'):
                        if step < injection_steps:
                            attn.inject_q = torch.cat(
                                [self.injection['input'][i][k]['q'][step]]*cond_length).to('cuda')
                            attn.inject_k = torch.cat(
                                [self.injection['input'][i][k]['k'][step]]*cond_length).to('cuda')
            if guided_hint is not None:
                h = module(h, emb, context,
                           transformer_options=transformer_options)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context,
                           transformer_options=transformer_options)
            result = rearrange(zero_conv(h, emb, context),
                               'b c f h w -> (b f) c h w')
            outs.append(result)
            if stage == 'inversion':
                for k, attn in enumerate(self.input_blocks.named_modules()):
                    if hasattr(attn, 'q') and hasattr(attn, 'k'):
                        if k not in self.injection['input'][i]:
                            self.injection['input'][i][k] = {'q': [], 'k': []}
                        self.injection['input'][i][k]['q'].append(attn.q.cpu())
                        self.injection['input'][i][k]['k'].append(attn.k.cpu())
            i += 1

        if stage == 'sampling' and step < injection_steps:
            for k, attn in enumerate(self.middle_block.named_modules()):
                if hasattr(attn, 'inject_q') and hasattr(attn, 'inject_k'):
                    attn.inject_q = torch.cat(
                        [self.injection['middle'][k]['q'][step]]*cond_length).to('cuda')
                    attn.inject_k = torch.cat(
                        [self.injection['middle'][k]['k'][step]]*cond_length).to('cuda')
        h = self.middle_block(
            h, emb, context, transformer_options=transformer_options)
        if stage == 'inversion':
            for k, attn in enumerate(self.named_modules()):
                if hasattr(attn, 'q') and hasattr(attn, 'k'):
                    if k not in self.injection['middle']:
                        self.injection['middle'][k] = {'q': [], 'k': []}
                    self.injection['middle'][k]['q'].append(attn.q.cpu())
                    self.injection['middle'][k]['k'].append(attn.k.cpu())
        result = rearrange(self.middle_block_out(
            h, emb, context), 'b c f h w -> (b f) c h w')
        outs.append(result)

        return outs
