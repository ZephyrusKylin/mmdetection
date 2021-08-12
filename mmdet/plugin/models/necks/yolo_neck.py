# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmdet.models.builder import NECKS
from mmdet.plugin.models.backbones.yolov5_backbone import autopad, BottleneckCSP
from mmdet.models.utils.make_divisible import make_divisible


@NECKS.register_module()
class YOLOV5Neck(BaseModule):
    """The neck of YOLOV3.

    It can be treated as a simplified version of FPN. It
    will take the result from Darknet backbone and do some upsampling and
    concatenation. It will finally output the detection result.

    Note:
        The input feats should be from top to bottom.
            i.e., from high-lvl to low-lvl
        But YOLOV3Neck will process them in reversed order.
            i.e., from bottom (high-lvl) to top (low-lvl)

    Args:
        num_scales (int): The number of scales / stages.
        in_channels (List[int]): The number of input channels per scale.
        out_channels (List[int]): The number of output channels  per scale.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Dictionary to construct and config norm
            layer. Default: dict(type='BN', requires_grad=True)
        act_cfg (dict, optional): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    stages_repeat = 3


    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 CSP_out_channels,
                 conv_behind_CSP_channels,
                 depth_multiple,
                 width_multiple,
                 round_nearest=8,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg=None):
        super(YOLOV5Neck, self).__init__(init_cfg)
        assert (num_scales == len(in_channels) == len(out_channels))
        self.stages_repeats = [self.stages_repeat] * (len(num_scales) * 2 - 1)
        

        self.num_scales = num_scales
        self.in_channels = [make_divisible(in_channel * width_multiple, round_nearest) for in_channel in in_channels]
        self.out_channels = [make_divisible(out_channel * width_multiple, round_nearest) for out_channel in out_channels]
        self.CSP_out_channels = [make_divisible(CSP_out_channel * width_multiple, round_nearest) for CSP_out_channel in CSP_out_channels]
        self.conv_behind_CSP_channels = [make_divisible(conv_behind_CSP_channel * width_multiple, round_nearest) for conv_behind_CSP_channel in conv_behind_CSP_channels]
        

        assert len(CSP_out_channels) == len(self.stages_repeats)
        assert len(conv_behind_CSP_channels) == len(CSP_out_channels) - 1

        self.bottle_nums = [max(round(stages_repeat * depth_multiple), 1) for stages_repeat in self.stages_repeats]
        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # To support arbitrary scales, the code looks awful, but it works.
        # Better solution is welcomed.
        # self.BottleneckCSP1 = BottleneckCSP(self.in_channels[0], self.stages_channels[0], self.bottle_nums[0], shortcut=False, **cfg)
        # self.conv1 = ConvModule(self.out_channels[0], self.out_channels[1], 1, 1, autopad(1, None), **cfg)
        in_CSP_channel = self.in_channels[0]
        conv_behind_CSP_channel_list =[]
        for i in range(num_scales-1):
            self.add_module(f'BottleneckCSP{i+1}',
                            BottleneckCSP(in_CSP_channel, self.CSP_out_channels[i], self.bottle_nums[i], shortcut=False, **cfg))
            self.add_module(f'conv{i+1}', ConvModule(self.CSP_out_channels[i], self.conv_behind_CSP_channels[i], 1, 1, autopad(1, None), **cfg))

            conv_behind_CSP_channel_list.append(self.conv_behind_CSP_channels[i])
            in_CSP_channel = self.conv_behind_CSP_channels[i] + self.in_channels[i+1]
        for i in range(num_scales-1, len(self.stages_repeats) - 1):
            self.add_module(f'BottleneckCSP{i+1}',
                            BottleneckCSP(in_CSP_channel, self.CSP_out_channels[i], self.bottle_nums[i], shortcut=False, **cfg))
            self.add_module(f'conv{i+1}', ConvModule(self.CSP_out_channels[i], self.conv_behind_CSP_channels[i], 1, 1, autopad(1, None), **cfg))

            in_CSP_channel = self.conv_behind_CSP_channels[i] + conv_behind_CSP_channel_list.pop()

        max_len = len(self.stages_repeats)
        self.add_module(f'BottleneckCSP{max_len}',
                            BottleneckCSP(in_CSP_channel, self.CSP_out_channels[-1], self.bottle_nums[-1], shortcut=False, **cfg))
        
    def forward(self, feats):
        assert len(feats) == self.num_scales
        reversed_feats = reversed(feats)
        tmp_feats = []
        input_feats = []
        input_feats.append(reversed_feats[0])
        for i in range(self.num_scales - 1):
            x = input_feats[i]
                
            CSP_block = getattr(self, f'BottleneckCSP{i+1}')
            conv = getattr(self, f'conv{i+1}')
            x = CSP_block(x)
            x = conv(x)
            tmp_feats.append(x)
            x = F.interpolate(x, scale_factor=2)
            cat_feat = torch.cat((reversed_feats[i+1], x), dim=1)
            input_feats.append(cat_feat)
        
        outs = []
        last_stage_feat = input_feats[-1]
        for i in range(self.num_scales):
            CSP_block = getattr(self, f'BottleneckCSP{self.num_scales + i}')
            out = CSP_block(last_stage_feat)
            outs.append(out)
            if i < self.num_scales - 1:
                conv = getattr(self, f'conv{self.num_scales + i}')
                down_feat = conv(out)
                tmp_feat = tmp_feats.pop()
                last_stage_feat = torch.cat((down_feat, tmp_feat), dim=1)
        outs = reversed(outs)
        return tuple(outs)
