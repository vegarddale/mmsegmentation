# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
############# Adapted from mmseg/models/backbones/mscan.py #############
import math
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmseg.registry import MODELS

import sys
sys.path.append("./mmseg/models")
from utils import dcnv3_ka



class ToChannelsLast(nn.Module):
    def forward(self, x):
        # Convert from N, C, H, W -> N, H, W, C
        return x.permute(0, 2, 3, 1)

class ToChannelsFirst(nn.Module):
    def forward(self, x):
        # Convert from N, H, W, C -> N, C, H, W
        return x.permute(0, 3, 1, 2)

class Mlp(BaseModule):
    """Multi Layer Perceptron (MLP) Module.

    Args:
        in_features (int): The dimension of input features.
        hidden_features (int): The dimension of hidden features.
            Defaults: None.
        out_features (int): The dimension of output features.
            Defaults: None.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=True,
            groups=hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward function."""
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(BaseModule):
    """Stem Block at the beginning of Semantic Branch.

    Args:
        in_channels (int): The dimension of input channels.
        out_channels (int): The dimension of output channels.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).

    Input dimensions: (B, C, H, W)
    Output dimensions: (B, H*W, C)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            ToChannelsLast(),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            ToChannelsFirst(),
            build_activation_layer(act_cfg),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            ToChannelsLast(),
            build_norm_layer(norm_cfg, out_channels)[1],
            ToChannelsFirst(),
        )

    def forward(self, x):
        """Forward function."""
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W

#Attention block
class DSCASpatialAttention(BaseModule):
  
    """ Deformable Strip Convolutional Attention module(DSCA)."""

    def __init__(self,
                 core_op,
                 attn_module,
                 channels,
                 kernel_size,
                 pad,
                 group,
                 act_layer='GELU',
                 norm_layer='LN',
                 offset_scale=1.0,
                 dw_kernel_size=None, # for InternImage-H/G
                 center_feature_scale=False, 
                 act_cfg=dict(type='GELU')):

        super().__init__()
        self.proj_1 = nn.Conv2d(channels, channels, 1)
        self.activation = build_activation_layer(act_cfg)
        self.attn = attn_module(
            core_op=core_op,
            channels=channels,
            kernel_size=kernel_size,
            stride=1,
            pad=pad,
            dilation=1,
            group=group,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size, # for InternImage-H/G
            center_feature_scale=center_feature_scale)
        
        self.proj_2 = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        """Forward function."""
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = x.permute(0,2,3,1).contiguous()
        x = self.attn(x)
        x = self.proj_2(x)
        x = x + shorcut
        x = x.permute(0,2,3,1)
        return x

# entire block
class DSCABlock(BaseModule):
    """Deformable Strip Convolution Attention block. 
       Utilizes the deformable large kernel approximation module.
    """

    def __init__(self,
                 core_op,
                 attn_module,
                 channels,
                 kernel_size,
                 pad,
                 group,
                 act_layer='GELU',
                 norm_layer='LN',
                 offset_scale=1.0,
                 dw_kernel_size=None, # for InternImage-H/G
                 center_feature_scale=False,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]
        self.attn = DSCASpatialAttention(
                 core_op=core_op,
                 attn_module=attn_module,
                 channels=channels,
                 kernel_size=kernel_size,
                 pad=pad,
                 group=group,
                 act_layer=act_layer,
                 norm_layer=norm_layer,
                 offset_scale=offset_scale,
                 dw_kernel_size=dw_kernel_size, # for InternImage-H/G
                 center_feature_scale=center_feature_scale,
                 act_cfg=act_cfg)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]
        mlp_hidden_channels = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_channels,
            act_cfg=act_cfg,
            drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)

    def forward(self, x, H, W):
        """Forward function."""
        
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * ToChannelsFirst()(
            self.norm1(self.attn(x))))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * ToChannelsFirst()(
            self.norm2(ToChannelsLast()(self.mlp(x)))))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(BaseModule):
    """Image to Patch Embedding.

    Args:
        patch_size (int): The patch size.
            Defaults: 7.
        stride (int): Stride of the convolutional layer.
            Default: 4.
        in_channels (int): The number of input channels.
            Defaults: 3.
        embed_dims (int): The dimensions of embedding.
            Defaults: 768.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_channels=3,
                 embed_dim=768,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2)
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        """Forward function."""

        x = self.proj(x)
        _, _, H, W = x.shape
        x = ToChannelsLast()(x)
        x = self.norm(x)
        x = ToChannelsFirst()(x)
        x = x.flatten(2).transpose(1, 2)
    
        return x, H, W

@MODELS.register_module()
class DSCAN(BaseModule):
    def __init__(self,
                 core_op="DCNv3",
                 attn_module="DCNv3KA",
                 groups=[2, 4, 8, 16],
                 act_layer='GELU',
                 norm_layer='LN',
                 offset_scale=1.0,
                 dw_kernel_size=None, # for InternImage-H/G
                 center_feature_scale=False,
                 in_channels=3,
                 kernel_size=[5,7],
                 pad=[2,3],
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None,
                 channel_attention=False):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.core_op = core_op
        self.attn_module=attn_module
        self.depths = depths
        self.num_stages = num_stages

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        
        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    norm_cfg=norm_cfg)
            block = nn.ModuleList([
                DSCABlock(
                    core_op=core_op,
                    attn_module=getattr(dcnv3_ka, attn_module),
                    group=groups[i],
                    channels=embed_dims[i],
                    kernel_size=kernel_size,
                    pad=pad,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)
        self.apply(self._init_deform_weights)

    def init_weights(self):
        """Initialize modules of MSCAN."""

        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def _init_deform_weights(self, m):
        if isinstance(m, getattr(dcnv3_ka, self.attn_module)):
            m._reset_parameters()

    def forward(self, x):
        """Forward function."""

        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs
