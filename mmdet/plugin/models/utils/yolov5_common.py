import torch
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.registry import ACTIVATION_LAYERS, CONV_LAYERS, NORM_LAYERS
from mmcv.cnn.bricks.norm import build_norm_layer
from mmdet.models.utils.make_divisible import make_divisible
# @ACTIVATION_LAYERS.register_module()
# class SiLU(nn.module):
#     def __init__(self, inplace=False):
#         super(SiLU, self).__init__()
#         self.inplace = inplace
    
#     def forward(self, x):
#         return nn.SiLU(x, self.inplace)
ACTIVATION_LAYERS.register_module('SiLU', module=nn.SiLU)
ACTIVATION_LAYERS.register_module('Hardswish', module=nn.Hardswish)

def autopad(kernel_size, padding=None):  # kernel, padding
    # Pad to 'same'
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]  # auto-pad
    return padding

@CONV_LAYERS.register_module('USConv')
class USConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[True, True], ratio=[1, 1], width_mult=1,):
        super().__init__(in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        # self.width_mult = None
        self.width_mult = width_mult
        self.us = us
        self.ratio = ratio

        #TODO
        assert self.groups == 1
        assert self.depthwise == False
    def forward(self, input):
        if self.us[0]:
            self.in_channels = make_divisible(
                self.in_channels_max
                * self.width_mult
                / self.ratio[0], 2) * self.ratio[0]
        if self.us[1]:
            self.out_channels = make_divisible(
                self.out_channels_max
                * self.width_mult
                / self.ratio[1], 2) * self.ratio[1]
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias

        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y

@NORM_LAYERS.register_module('SBN')
class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, in_channel, WIDTH_LIST, rate=1, **kargs):
        super(SwitchableBatchNorm2d, self).__init__()
        self.WIDTH_LIST = WIDTH_LIST
        num_features_list = [make_divisible(in_channel*i/rate, 2)*rate for i in self.WIDTH_LIST]
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)

        if 'type' not in kargs:
            kargs.setdefault('type', 'BN')

        bns = []
        for i in num_features_list:
            bns.append(build_norm_layer(kargs, i)[1])
        self.bn = nn.ModuleList(bns)
        self.width_mult = max(self.WIDTH_LIST)
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = self.WIDTH_LIST.index(self.width_mult)
        # print('--'*20)
        # print(*self.num_features_list)
        # print('idx {} self.mult {} input.size{}'.format(idx,self.width_mult,input.size()))
        y = self.bn[idx](input)
        # self.bn = nn.BatchNorm2d(32)(input)
        # y = self.bn
        # print('bn done')
        return y

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channel, out_channel, shortcut=True, padding=None, groups=1, expansion=0.5, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='SiLU')):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        hidden_channel = int(out_channel * expansion)  # hidden channels
        self.conv1 = ConvModule(in_channel, hidden_channel, 1, 1, autopad(1, padding), conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(hidden_channel, out_channel, 3, 1, autopad(3, padding), groups=groups, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.add = shortcut and in_channel == out_channel

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class BottleneckCSP(nn.Module):
    def __init__(self, input_channel, output_channel, WIDTH_LIST, bottle_nums=1, shortcut=True, padding=None, groups=1, expand=0.5, conv_cfg=dict(type='USConv', ratio=[2, 1]), norm_cfg=dict(type='SBN', requires_grad=True), act_cfg=dict(type='SiLU')):
        super(BottleneckCSP, self).__init__()

        hidden_channel = int(output_channel * expand)

        self.conv1 = ConvModule(input_channel, hidden_channel, 1, 1,padding=autopad(1, padding), conv_cfg=dict(type='USConv', ratio=[1, 1]), norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(input_channel, hidden_channel, 1, 1,padding=None, conv_cfg=dict(type='Conv'), norm_cfg=None, act_cfg=None)
        self.conv3 = ConvModule(hidden_channel, hidden_channel, 1, 1,padding=None, conv_cfg=dict(type='Conv'), norm_cfg=None, act_cfg=None)
        self.conv4 = ConvModule(hidden_channel * 2, output_channel, 1, 1,padding=autopad(1, padding), conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.bn = SwitchableBatchNorm2d(hidden_channel * 2, WIDTH_LIST, 2)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.layers = nn.Sequential(*[Bottleneck(hidden_channel, hidden_channel, shortcut, groups, expand=1.0) for _ in range(bottle_nums)])
    
    def forward(self, x):
        y0=self.conv1(x)
        y1 = self.conv3(self.layers(y0))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, conv_cfg=dict(type='USConv', us=[False, True]), norm_cfg=dict(type='SBN', requires_grad=True), act_cfg=dict(type='SiLU')):
        super().__init__()
        norm_cfg.setdefault('WIDTH_LIST', WIDTH_LIST)
        self.conv_focus = ConvModule(in_channels * 4, out_channels, kernel_size, stride, autopad(kernel_size, padding), groups=groups, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
    
    def forward(self, x):
        y = self._focus_transform(x)
        y = self.conv_focus(y)
        return y

    @staticmethod
    def _focus_transform(x):
        y = torch.cat([x[..., ::2, ::2],
                       x[..., 1::2, ::2],
                       x[..., ::2, 1::2],
                       x[..., 1::2, 1::2]], 1)
        return y