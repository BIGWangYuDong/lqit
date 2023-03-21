# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.models.necks import FPN
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType
from torch import Tensor


@MODELS.register_module()
class UFPN(FPN):
    """
    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int = 6,
                 start_level: int = 0,
                 end_level: int = -1,
                 add_extra_convs: Union[bool, str] = 'on_output',
                 relu_before_extra_convs: bool = False,
                 no_norm_on_lateral: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 upsample_cfg: ConfigType = dict(mode='nearest'),
                 init_cfg: MultiConfig = dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            start_level=start_level,
            end_level=end_level,
            add_extra_convs=add_extra_convs,
            relu_before_extra_convs=relu_before_extra_convs,
            no_norm_on_lateral=no_norm_on_lateral,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            upsample_cfg=upsample_cfg,
            init_cfg=init_cfg)
        # add encoder pathway
        self.encode_convs = nn.ModuleList()
        self.connect_convs = nn.ModuleList()
        for i in range(self.start_level, self.num_outs + self.start_level):
            connect_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            if i < self.num_outs + self.start_level - 1:
                encode_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)

                self.encode_convs.append(encode_conv)
            self.connect_convs.append(connect_conv)

        # add decoder pathway
        self.decode_convs = nn.ModuleList()
        decode_conv = ConvModule(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,  # may use ReLU or LeakyReLU
            inplace=False)

        self.decode_convs.append(decode_conv)

        for _ in range(self.start_level + 1, self.num_outs + self.start_level):
            conv1 = ConvModule(
                out_channels * 2,  # concat with other feature map
                out_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,  # may use ReLU or LeakyReLU
                inplace=False)
            conv2 = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,  # may use ReLU or LeakyReLU
                inplace=False)
            decode_conv = nn.Sequential(conv1, conv2)
            self.decode_convs.append(decode_conv)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # build extra outputs

        if self.num_outs > len(inter_outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    inter_outs.append(
                        F.max_pool2d(inter_outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = inter_outs[-1]
                else:
                    raise NotImplementedError
                inter_outs.append(
                    self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        inter_outs.append(self.fpn_convs[i](F.relu(
                            inter_outs[-1])))
                    else:
                        inter_outs.append(self.fpn_convs[i](inter_outs[-1]))

        # part 2: add encoder path
        connect_feats = [
            self.connect_convs[i](inter_outs[i]) for i in range(self.num_outs)
        ]

        encode_outs = []
        encode_outs.append(connect_feats[0])
        for i in range(0, self.num_outs - 1):
            encode_outs.append(self.encode_convs[i](connect_feats[i]) +
                               connect_feats[i + 1])

        # part 3: add decoder levels
        decode_outs = [
            torch.zeros_like(encode_outs[i]) for i in range(self.num_outs)
        ]
        decode_outs[-1] = self.decode_convs[0](encode_outs[-1])
        for i in range(1, self.num_outs):
            reverse_i = self.num_outs - i
            prev_shape = encode_outs[reverse_i - 1].shape[2:]
            up_feat = F.interpolate(
                decode_outs[reverse_i], size=prev_shape, **self.upsample_cfg)
            decode_outs[reverse_i - 1] = self.decode_convs[i](
                torch.cat((encode_outs[reverse_i - 1], up_feat), dim=1))

        return tuple(decode_outs)
