import torch
import torch.nn as nn
import warnings
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.activation import build_activation_layer
from mmengine.model import BaseModule

from lqit.registry import MODELS


@MODELS.register_module()
class AODNet(BaseModule):
    """AOD-Net: All-in-One Dehazing Network."""

    def __init__(self,
                 in_channels=(1, 1, 2, 2, 4),
                 base_channels=3,
                 out_channels=(3, 3, 3, 3, 3),
                 num_stages=5,
                 kernel_size=(1, 3, 5, 7, 3),
                 padding=(0, 1, 2, 3, 1),
                 act_cfg=dict(type='ReLU'),
                 plugins=None,
                 pretrained=None,
                 norm_eval=False,
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

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')
        assert plugins is None, 'Not implemented yet.'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_stages = num_stages
        self.base_channels = base_channels
        self.padding = padding
        self.with_activation = act_cfg is not None
        self.norm_eval = norm_eval
        self.act_cfg = act_cfg
        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
            ]:
                act_cfg_.setdefault('inplace', True)
            self.activate = build_activation_layer(act_cfg_)

        self._init_layer()

    def _init_layer(self):

        self.CONVM = nn.ModuleList()
        for i in range(self.num_stages):
            conv_act = ConvModule(
                in_channels=self.in_channels[i] * self.base_channels, out_channels=self.out_channels[i],
                kernel_size=self.kernel_size[i], stride=1, padding=self.padding[i], bias=True, act_cfg=self.act_cfg)
            self.CONVM.append(conv_act)


    def forward(self, inputs):
        outs = []
        x1 = inputs
        for i in range(self.num_stages):
            if i > 1 and i != (self.num_stages - 1):  # from i=2 concat
                x1 = torch.cat((outs[i - 2], outs[i - 1]), 1)

            if i == self.num_stages - 1:  # last concat all
                x1 = torch.cat([outs[j] for j in range(len(outs))], 1)

            x1 = self.CONVM[i](x1)
            outs.append(x1)
        result = self.activate((outs[-1] * inputs) - outs[-1] + 1)

        return result
