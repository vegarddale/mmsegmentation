# -----------------------------------------------------------------------------------
#  Modified from
# https://github.com/OpenGVLab/InternImage 
# ------------------------------------------------------------------------------------

import math
import warnings
import torch
import torch.nn as nn
import sys
sys.path.append("./ops_dcnv3")
import modules as dcnv3
from mmengine.model import BaseModule

# husk att de gjør kernel decomposition inne i dcvn3, dette burde vi endre på i strip wise siden det løser samme problemet
class DCNv3KA(BaseModule):
    def __init__(self,
                 core_op,
                 channels,
                 group,
                 kernel_size,
                 stride,
                 pad,
                 dilation,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 offset_scale=1.0,
                 dw_kernel_size=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False): 
        super().__init__()
        self.channels = channels
        self.core_op = getattr(dcnv3, core_op)
        self.output_proj = nn.Linear(channels, channels)
        # self.dw_dcn = nn.Conv2d(
        #     channels,
        #     channels,
        #     kernel_size=tuple((kernel_size[0], kernel_size[0])),
        #     padding=tuple((pad[0], pad[0])),
        #     groups=channels)
        self.dw_dcn = self.core_op( TODO her vil vi teste å bytte dw_dcn med en vanlig dw convolution, det vi vil teste med det er large deformable kernel attention(disse modellene burde være
                                                                                                                                                                    små nok i base att vi kan øke størrelsen, hvis ikke er de for store som base)
            channels=channels,
            kernel_size=tuple((kernel_size[0], kernel_size[0])),
            stride=1,
            pad=tuple((pad[0], pad[0])),
            dilation=1,
            group=channels,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size, # for InternImage-H/G
            center_feature_scale=center_feature_scale)
        
        self.dw_d_dcn = self.core_op(
            channels=channels,
            kernel_size=tuple((kernel_size[1], kernel_size[1])),
            stride=1,
            pad=tuple((pad[1], pad[1])),
            dilation=1,
            group=channels,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size, # for InternImage-H/G
            center_feature_scale=center_feature_scale
        )

    def forward(self, x):
        u = x.clone()
        u = u.permute(0,3,1,2)
        # TODO add a dw 1x1 that is paralell with the attn(similar to v3+)(will be used to mimic channel attention)
        # x = x.permute(0,3,1,2)
        x = self.dw_dcn(x)
        # x = x.permute(0,2,3,1)
        x = self.dw_d_dcn(x)
        x = self.output_proj(x)
        x = x.permute(0,3,1,2)
        return x*u
        
    def _reset_parameters(self):
        self.dw_dcn._reset_parameters()
        self.dw_d_dcn._reset_parameters()
        

class DCNv3_SW_KA(BaseModule):
    """Attention Module in Multi-Scale Convolutional Attention Module (MSCA).

    Args:
        channels (int): The dimension of channels.
        kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
    """

    def __init__(self,
                 core_op,
                 channels,
                 group,
                 kernel_size=[5, [1, 7], [1, 11], [1, 21]],
                 stride=1,
                 pad=[2, [0, 3], [0, 5], [0, 10]],
                 dilation=1,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 offset_scale=1.0,
                 dw_kernel_size=None, # for InternImage-H/G
                 res_post_norm=False, # for InternImage-H/G
                 center_feature_scale=False,
                 strip_conv=1
                ):
        super().__init__()
        self.core_op = getattr(dcnv3, core_op)
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size[0],
            padding=pad[0],
            groups=channels)
        for i, (_kernel_size,
                padding) in enumerate(zip(kernel_size[1:], pad[1:])):
            kernel_size_ = [_kernel_size, _kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    self.core_op(
                        channels=channels,
                        kernel_size=tuple(i_kernel),
                        stride=stride,
                        pad=i_pad,
                        dilation=dilation,
                        group=channels,
                        offset_scale=offset_scale,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        dw_kernel_size=dw_kernel_size, # for InternImage-H/G
                        center_feature_scale=center_feature_scale,
                        strip_conv=strip_conv
                    ))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """Forward function."""
        x = x.permute(0,3,1,2).contiguous()
        u = x.clone()

        attn = self.conv0(x)
        attn = attn.permute(0,2,3,1).contiguous()
        
        # Multi-Scale Feature extraction
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        
        # Channel Mixing
        attn = attn.permute(0,3,1,2).contiguous()
        attn = self.conv3(attn)
        # Convolutional Attention
        x = attn * u

        return x

    
    def _reset_parameters(self):
        
        self.conv0_1._reset_parameters()
        self.conv0_2._reset_parameters()
        self.conv1_1._reset_parameters()
        self.conv1_2._reset_parameters()
        self.conv2_1._reset_parameters()
        self.conv2_2._reset_parameters()







