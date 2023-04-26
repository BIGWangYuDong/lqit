# Modified from https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/model.py  # noqa
# This work is licensed under Attribution-NonCommercial 4.0 International License.  # noqa
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from lqit.registry import MODELS


@MODELS.register_module()
class ZeroDCE(BaseModule):
    """Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement."""

    def __init__(self,
                 in_channels=3,
                 feat_channels=32,
                 out_channels=3,
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
        assert out_channels == in_channels
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self._init_layer()

    def _init_layer(self):
        self.e_conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.e_conv2 = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.e_conv3 = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.e_conv4 = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.e_conv5 = nn.Conv2d(
            in_channels=self.feat_channels * 2,
            out_channels=self.feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.e_conv6 = nn.Conv2d(
            in_channels=self.feat_channels * 2,
            out_channels=self.feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.e_conv7 = nn.Conv2d(
            in_channels=self.feat_channels * 2,
            out_channels=self.out_channels * 8,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x1 = F.relu(self.e_conv1(x))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        (r1, r2, r3, r4, r5, r6, r7, r8) = torch.split(
            x_r, self.out_channels, dim=1)

        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        x = x + r4 * (torch.pow(x, 2) - x)
        x = x + r5 * (torch.pow(x, 2) - x)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return enhance_image, r


@MODELS.register_module()
class ZeroDCEFlexibleModel(BaseModule):
    """More flexible Zero-Reference Deep Curve Estimation for Low-Light Image
    Enhancement."""

    def __init__(self,
                 in_channels=3,
                 feat_channels=32,
                 out_channels=3,
                 num_convs=7,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 out_act_cfg=dict(type='Tanh'),
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
        assert out_channels == in_channels
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.num_convs = num_convs
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.out_act_cfg = out_act_cfg
        self._init_layer()

    def _init_layer(self):
        dce_convs = []
        for i in range(self.num_convs):
            if i == 0:
                in_channels = self.in_channels
                out_channels = self.feat_channels
            elif i == (self.num_convs - 1):
                in_channels = self.feat_channels * 2
                out_channels = self.out_channels * (self.num_convs + 1)
                dce_convs.append(
                    ConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.out_act_cfg))
                continue
            elif i <= (self.num_convs // 2):
                in_channels = self.feat_channels
                out_channels = self.feat_channels
            else:
                in_channels = self.feat_channels * 2
                out_channels = self.feat_channels
            dce_convs.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.model = nn.ModuleList(dce_convs)

    def forward(self, inputs):
        x = inputs
        xs = []
        for i, dce_conv in enumerate(self.model):
            if i <= self.num_convs // 2:
                x = dce_conv(x)
                xs.append(x)
            else:
                x = xs.pop(-1)
                x = dce_conv(torch.cat([xs.pop(-1), x], dim=1))
                xs.append(x)
        rs = torch.split(x, 3, dim=1)

        outs = inputs
        for r in rs:
            outs = outs + r * (outs**2 - outs)
        # compress the outs, this will be uncompress when calculating loss.
        outs = torch.cat([outs, x], dim=1)
        return outs
