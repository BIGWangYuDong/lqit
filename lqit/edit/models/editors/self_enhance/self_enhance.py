import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule
from torch import Tensor

from lqit.registry import MODELS
from lqit.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class SelfEnhance(BaseModule):
    """Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement."""

    def __init__(self,
                 in_channels=3,
                 feat_channels=64,
                 out_channels=3,
                 num_convs=3,
                 expand_ratio=1.0,
                 num_blocks=1,
                 add_identity=True,
                 channel_attention=True,
                 kernel_size=5,
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
        for i in range(num_convs):
            if i == num_convs - 1:
                _out_channels = out_channels
            else:
                _out_channels = feat_channels
            layer = SelfEnhanceLayer(
                in_channels=feat_channels,
                out_channels=_out_channels,
                expand_ratio=expand_ratio,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                channel_attention=channel_attention,
                kernel_size=kernel_size,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                conv_cfg=conv_cfg)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x_stem = self.stem(x)

        out = self.layers(x_stem)  # enhanced img structure
        out_img = out + x  # enhance img
        cat_tensor = torch.cat([out_img, out], dim=1)
        return cat_tensor


class SelfEnhanceLayer(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: float = 0.5,
                 num_blocks: int = 2,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 channel_attention: bool = False,
                 kernel_size: int = 5,
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
        super().__init__(init_cfg=init_cfg)
        block = SelfEnhanceBasicBlock
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.short_conv = ConvModule(
            in_channels,
            mid_channels,
            1,
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
        self.blocks = nn.Sequential(*[
            block(
                mid_channels,
                mid_channels,
                1.0,
                add_identity,
                use_depthwise,
                kernel_size=kernel_size,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        out = self.final_conv(x_final)
        return out


class SelfEnhanceBasicBlock(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 1.0,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 kernel_size: int = 5,
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
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            mid_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            out = out + identity

        return out


class ChannelAttention(BaseModule):
    """Channel attention Module.

    Args:
        channels (int): The input (and output) channels of the attention layer.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self, channels: int, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out
