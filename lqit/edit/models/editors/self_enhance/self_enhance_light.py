import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor

from lqit.registry import MODELS
from lqit.utils import ConfigType, OptConfigType


@MODELS.register_module()
class SelfEnhanceLight(BaseModule):
    """Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement."""

    def __init__(self,
                 in_channels=3,
                 feat_channels=64,
                 out_channels=3,
                 num_blocks=4,
                 expand_ratio=1.0,
                 kernel_size=[1, 3, 5, 7],
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 use_depthwise: bool = True,
                 init_cfg=[
                     dict(type='Normal', layer='Conv2d', mean=0, std=0.02),
                     dict(
                         type='Normal',
                         layer='BatchNorm2d',
                         mean=1.0,
                         std=0.02,
                         bias=0),
                 ]):
        super().__init__(init_cfg=init_cfg)
        assert len(kernel_size) == num_blocks
        self.in_channels = in_channels
        self.stem = ConvModule(
            in_channels,
            feat_channels,
            3,
            padding=1,
            stride=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        layers = []
        for i in range(num_blocks):
            if i == num_blocks - 1:
                _out_channels = out_channels
            else:
                _out_channels = feat_channels
            layer = SelfEnhanceLayer(
                in_channels=feat_channels,
                out_channels=_out_channels,
                expand_ratio=expand_ratio,
                use_depthwise=use_depthwise,
                kernel_size=kernel_size[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                conv_cfg=conv_cfg)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x_stem = self.stem(x)

        out = self.layers(x_stem)  # enhanced img structure
        out_img = 0.8 * out + 0.2 * x  # enhance img
        cat_tensor = torch.cat([out_img, out], dim=1)
        return cat_tensor


class SelfEnhanceLayer(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 use_depthwise: bool = False,
                 kernel_size: int = 3,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 conv_cfg: OptConfigType = None,
                 init_cfg=[
                     dict(type='Normal', layer='Conv2d', mean=0, std=0.02),
                     dict(
                         type='Normal',
                         layer='BatchNorm2d',
                         mean=1.0,
                         std=0.02,
                         bias=0),
                 ]):
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        super().__init__(init_cfg=init_cfg)
        mid_channels = int(out_channels * expand_ratio)

        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = conv(
            in_channels,
            mid_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            2 * mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.block = SelfEnhanceBasicBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            expansion=1.0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.block(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        out = self.final_conv(x_final)
        return out


class SelfEnhanceBasicBlock(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 1.0,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 init_cfg=[
                     dict(type='Normal', layer='Conv2d', mean=0, std=0.02),
                     dict(
                         type='Normal',
                         layer='BatchNorm2d',
                         mean=1.0,
                         std=0.02,
                         bias=0),
                 ]):
        super().__init__(init_cfg=init_cfg)
        mid_channels = int(out_channels * expansion)
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity

        return out
