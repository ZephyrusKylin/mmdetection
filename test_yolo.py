import pytest
import torch
import math
from mmdet.models.utils.make_divisible import make_divisible
from mmcv import assert_params_all_zeros
from mmcv.ops import DeformConv2dPack
from torch.nn.modules import AvgPool2d, GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.backbones import ResNet, ResNetV1d
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet.models.utils import ResLayer, SimplifiedBasicBlock
# from .utils import check_norm_state, is_block, is_norm
from mmdet.models.utils.yolov5_common import Focus, USConv, SwitchableBatchNorm2d, Bottleneck, BottleneckCSP


if __name__ == "__main__":
    # input_tensor = torch.tensor()
    # input_tensor1 = torch.rand((4,3,10,10))
    # focus = Focus(3, 32)
    # out_1 = focus(input_tensor1)
    # print(out_1.shape)

    width_mult = 0.8
    input_tensor2 = torch.rand((4,math.ceil(64 * width_mult),100,100))
    usconv = USConv(64, 128, 1, us=[True, True], ratio=[1, 1], width_mult=width_mult)
    out_2 = usconv(input_tensor2)
    print(out_2.shape)

    # #TODO logit bug
    # WIDTH_LIST = [0.5, 0.2]
    # in_channel = out_2.shape[1]
    # input_tensor3 = out_2[:,:make_divisible(out_2.shape[1] * max(WIDTH_LIST), 2), :, :]

    # print(out_2.shape)
    # print(input_tensor3.shape)

    

    # sbn = SwitchableBatchNorm2d(in_channel, WIDTH_LIST, rate=1)
    # out_3 = sbn(input_tensor3)
    # print(out_3.shape)