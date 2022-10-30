import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from lqit.registry import MODELS
from .base_head import BaseEnhanceHead


@MODELS.register_module()
class EdgeHead(BaseEnhanceHead):
    """[conv+GN+relu]*4+1*1conv."""

    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 num_convs=5,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 gt_preprocessor=None,
                 loss_enhance=dict(type='lqit.L1Loss', loss_weight=1.0),
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01)):
        super().__init__(
            loss_enhance=loss_enhance,
            gt_preprocessor=gt_preprocessor,
            init_cfg=init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self._init_layers()

    def _init_layers(self):
        assert self.num_convs > 0
        enhance_conv = []
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 \
                else self.feat_channels
            if i < (self.num_convs - 1):
                enhance_conv.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            else:
                enhance_conv.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=1,
                        kernel_size=1,
                        stride=1,
                        padding=1))
        self.enhance_conv = nn.Sequential(*enhance_conv)

    def forward(self, x):
        if len(x) > 1 and (isinstance(x, tuple) or isinstance(x, list)):
            x = x[0]
        outs = self.enhance_conv(x)
        return outs

    def loss_by_feat(self, enhance_img, gt_imgs, img_metas):
        reshape_gt_imgs = F.interpolate(
            gt_imgs, size=enhance_img.shape[-2:], mode='bilinear')
        enhance_loss = self.loss_enhance(enhance_img, reshape_gt_imgs)
        return dict(loss_enhance=enhance_loss)
