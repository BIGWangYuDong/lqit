# Modified from https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/model.py  # noqa
import torch
import torch.nn as nn
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
