import copy
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.utils import OptConfigType
from mmengine.model import BaseModule

from lqit.registry import MODELS


@MODELS.register_module()
class DFFPN(BaseModule):
    """Dynamic feature fusion pyramid network."""

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int = 256,
                 num_outs: int = 5,
                 start_level: int = 0,
                 end_level: int = -1,
                 add_extra_convs: Union[bool, str] = False,
                 shape_level: int = 2,
                 relu_before_extra_convs: bool = False,
                 no_norm_on_lateral: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 upsample_cfg: OptConfigType = dict(mode='nearest'),
                 init_cfg: OptConfigType = dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')) \
            -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.shape_level = shape_level
        self.no_norm_on_lateral = no_norm_on_lateral
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.upsample_cfg = upsample_cfg.copy()
        self.relu_before_extra_convs = relu_before_extra_convs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.la1_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpl_convs = nn.ModuleList()
        self.dff = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channels * self.num_outs, self.num_outs, 1))
        for i in range(self.start_level, self.backbone_end_level):
            l1_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            f2_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.la1_convs.append(l1_conv)
            self.fpn_convs.append(f2_conv)

        for j in range(self.num_outs):
            fl_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.fpl_convs.append(fl_conv)
        self.pooling = F.adaptive_avg_pool2d

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs: tuple) -> tuple:
        """Forward function."""
        assert len(inputs) == self.num_ins

        # 1. Unify channel through 1*1 Conv layer
        lat = [
            la1_conv(inputs[i + self.start_level])
            for i, la1_conv in enumerate(self.la1_convs)
        ]

        laterals = copy.copy(lat)
        # 2. fpn up to down:
        used_backbone_levels = len(lat)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = lat[i - 1].shape[2:]
            laterals[i - 1] = lat[i - 1] + F.interpolate(
                lat[i], size=prev_shape, **self.upsample_cfg)

        laterals = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # add extra layers
        if self.num_outs > len(laterals):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - self.num_ins):
                    laterals.append(F.max_pool2d(laterals[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = laterals[-1]
                else:
                    raise NotImplementedError
                laterals.append(
                    self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        laterals.append(self.fpn_convs[i](F.relu(
                            laterals[-1])))
                    else:
                        laterals.append(self.fpn_convs[i](laterals[-1]))

        # 3. pooling and concat
        t_outs = []
        pool_shape = laterals[self.shape_level].size()[2:]

        for i in range(0, self.num_outs):
            t_outs.append(self.pooling(laterals[i], pool_shape))

        t_out = torch.cat(t_outs, dim=1)
        # 4. get each feature map weights
        ws = self.dff(t_out)
        ws = torch.sigmoid(ws)
        w = torch.split(ws, 1, dim=1)

        inner_outs = []

        for i in range(0, self.num_outs):
            inner_outs.append(laterals[i] * w[i])

        for i in range(self.num_outs - 1):
            prev_shape = inner_outs[i + 1].shape[2:]
            inner_outs[i + 1] = inner_outs[i + 1] + F.interpolate(
                inner_outs[i], size=prev_shape, **self.upsample_cfg)

        outs = [
            self.fpl_convs[i](inner_outs[i] + laterals[i])
            for i in range(self.num_outs)
        ]

        return tuple(outs)
