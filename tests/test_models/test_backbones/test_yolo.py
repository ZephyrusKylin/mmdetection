import pytest
import torch
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
    input_tensor = torch.rand((3,10,10))

    focus = Focus(3, 3)
    out_1 = focus(input_tensor)
    print(out_1.shape)
    print(out_1)