# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import warnings

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.plugin.models.utils.yolov5_common import Focus, autopad
from mmdet.models.utils.make_divisible import make_divisible
from mmdet.models.builder import BACKBONES

from mmdet.plugin.models.utils.yolov5_common import Bottleneck


class ResBlock(BaseModule):
    """The basic residual block used in Darknet. Each ResBlock consists of two
    ConvModules and the input is added to the final output. Each ConvModule is
    composed of Conv, BN, and LeakyReLU. In YoloV3 paper, the first convLayer
    has half of the number of the filters as much as the second convLayer. The
    first convLayer has filter size of 1x1 and the second one has the filter
    size of 3x3.

    Args:
        in_channels (int): The input channels. Must be even.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg=None):
        super(ResBlock, self).__init__(init_cfg)
        assert in_channels % 2 == 0  # ensure the in_channels is even
        half_in_channels = in_channels // 2

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1 = ConvModule(in_channels, half_in_channels, 1, **cfg)
        self.conv2 = ConvModule(
            half_in_channels, in_channels, 3, padding=1, **cfg)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual

        return out


class BottleneckCSP(BaseModule):
    
    def __init__(self,
                 input_channel,
                 output_channel,
                 bottle_nums=1,
                 shortcut=True,
                 padding=None,
                 groups=1,
                 expansion=0.5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Hardswish'),
                 init_cfg=None):
        super(BottleneckCSP, self).__init__(init_cfg=init_cfg)
        hidden_channel = int(output_channel * expansion)  # hidden channels
        self.conv1 = ConvModule(input_channel, hidden_channel, 1, 1, autopad(1, padding), groups=groups, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv2 = ConvModule(input_channel, hidden_channel, 1, 1, 0, conv_cfg=None, norm_cfg=None, act_cfg=None, bias=False)
        self.conv3 = ConvModule(hidden_channel, hidden_channel, 1, 1, 0, conv_cfg=None, norm_cfg=None, act_cfg=None, bias=False)

        self.conv4 = ConvModule(2 * hidden_channel, output_channel, 1, 1, autopad(1, padding), groups=groups, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.cat_bn = nn.BatchNorm2d(2 * hidden_channel)
        self.cat_act = nn.LeakyReLU(0.1, inplace=True)

        self.block = nn.Sequential()
        
        for i in range(bottle_nums):
            layer_name = f'Bottleneck_block{i + 1}'
            self.block.add_module(
                layer_name,
                Bottleneck(
                    hidden_channel, 
                    hidden_channel, 
                    shortcut, 
                    groups=groups, 
                    expansion=1.0, 
                    conv_cfg=conv_cfg, 
                    norm_cfg=norm_cfg, 
                    act_cfg=act_cfg)
            )

    def forward(self, x):
        y0 = self.conv1(x)
        y1 = self.conv3(self.block(y0))
        y2 = self.conv2(x)
        y = torch.cat((y1, y2), dim=1)
        out = self.cat_act(self.cat_bn(y))
        out = self.conv4(out)

        return out

#TODO
class C3(BaseModule):
    def __init__(self,
                 input_channel,
                 output_channel,
                 bottle_nums=1,
                 shortcut=True,
                 padding=None,
                 groups=1,
                 expansion=0.5,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Hardswish'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        hidden_channel = int(output_channel * expansion)  # hidden channels
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(input_channel, hidden_channel, 1, 1, autopad(1, padding), groups=groups, **cfg)
        self.conv2 = ConvModule(input_channel, hidden_channel, 1, 1, autopad(1, padding), groups=groups, **cfg)
        self.conv3 = ConvModule(2 * hidden_channel, output_channel, 1, 1, autopad(1, padding), groups=groups, **cfg)

        self.block = nn.Sequential()
        for i in range(bottle_nums):
            layer_name = f'C3_block{i + 1}'
            self.block.add_module(
                layer_name,
                Bottleneck(
                    hidden_channel, 
                    hidden_channel, 
                    shortcut, 
                    groups=groups, 
                    expansion=1.0, 
                    **cfg)
            )
    
    def forward(self, x):
        x_ = self.conv1(x)
        out = torch.cat((self.block(x_), self.conv2(x)), dim=1)
        out = self.conv3(out)
        return out
        
class SPP(BaseModule):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Hardswish'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        hidden_channel = input_channel // 2  # hidden channels
        self.conv1 = ConvModule(input_channel, hidden_channel, 1, 1, padding=None, **cfg)
        self.conv2 = ConvModule(hidden_channel * (len(kernel_sizes) + 1), output_channel, 1, 1, padding=None, **cfg)
        self.pooling_layers = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in kernel_sizes])
    
    def forward(self, x):
        x = self.conv1(x)
        out = torch.cat([x] + [m(x) for m in self.pooling_layers], dim=1)
        out = self.conv2(out)

        return out

@BACKBONES.register_module()
class Darknet_v5(BaseModule):
    stages_repeats = [1, 3, 9, 9, 1]
    stages_channels = [64, 128, 256, 512, 1024]
    _block = {
    "BottleneckCSP": BottleneckCSP,
    "C3": C3,
    }

    def __init__(self,
                depth_multiple,
                width_multiple,
                out_indices=(2, 3),
                frozen_stages=-1,
                round_nearest=8,
                block_name="BottleneckCSP",
                conv_cfg=None,
                norm_cfg=dict(type='BN', requires_grad=True),
                act_cfg=dict(type='SiLU'),
                norm_eval=True,
                pretrained=None,
                init_cfg=None):
        super().__init__(init_cfg)

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.block = self._block[block_name]

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # building first layer
        assert self.stages_repeats[0] == 1
        assert len(self.stages_repeats) == len(self.stages_channels)

        output_channel =self.stages_channels[0]

        output_channel = make_divisible(output_channel * width_multiple, round_nearest)
        self.conv1 = Focus(3, output_channel, 3, padding=1, **cfg)
        self.cr_blocks = ['conv1']
        input_channel = output_channel

        # building CSP blocks
        for i, (bottle_nums, out_channel) in enumerate(zip(self.stages_repeats[1:-1], self.stages_channels[1:-1])):
            layer_name = f'csp1_block{i + 1}'

            bottle_nums = max(round(bottle_nums * depth_multiple), 1)
            out_channel = make_divisible(out_channel * width_multiple, round_nearest)

            self.add_module(
                layer_name,
                self.make_conv_CSP_block(input_channel, out_channel, self.block, bottle_nums, **cfg))
            self.cr_blocks.append(layer_name)
            input_channel = out_channel

        # building last CSP blocks
        assert self.stages_repeats[-1] == 1
        
        last_channel = make_divisible(self.stages_channels[-1] * width_multiple, round_nearest)
        self.last_stage = self.make_conv_last_stage(input_channel, last_channel, kernel_sizes=(5, 9, 13), **cfg)
        self.cr_blocks.append('last_stage')

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')
    
    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.cr_blocks):
            cr_block = getattr(self, layer_name)
            x = cr_block(x)
            if i in self.out_indices:
                outs.append(x)
        
        outs.append(x)
        return tuple(outs)

    @staticmethod
    def make_conv_CSP_block(in_channels,
                            out_channels,
                            block,
                            num_stages,
                            conv_cfg=None,
                            norm_cfg=dict(type='BN', requires_grad=True),
                            act_cfg=dict(type='SiLU')):
        
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        model = nn.Sequential()
        model.add_module(
            'conv',
            ConvModule(
                in_channels, out_channels, 3, stride=2, **cfg))
        model.add_module('block', block(out_channels, out_channels, num_stages, **cfg))

        return model
    
    @staticmethod
    def make_conv_last_stage(in_channels,
                            out_channels,
                            kernel_sizes=(5, 9, 13),
                            conv_cfg=None,
                            norm_cfg=dict(type='BN', requires_grad=True),
                            act_cfg=dict(type='SiLU')):
        
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        model = nn.Sequential()
        model.add_module(
            'conv',
            ConvModule(
                in_channels, out_channels, 3, stride=2, **cfg))
        model.add_module('SPP', SPP(out_channels, out_channels, kernel_sizes, **cfg))

        return model
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = getattr(self, self.cr_blocks[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    
    def train(self, mode=True):
        super(Darknet_v5, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
