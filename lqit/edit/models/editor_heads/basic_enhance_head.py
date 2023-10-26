import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from lqit.registry import MODELS
from lqit.utils import OptConfigType, OptMultiConfig
from .base_head import BaseEnhanceHead


@MODELS.register_module()
class SingleEnhanceHead(BaseEnhanceHead):
    """[Conv-BN-ReLU] * num_convs"""

    def __init__(
        self,
        in_channels: int = 256,
        feat_channels: int = 256,
        num_convs: int = 5,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = dict(type='ReLU'),
        gt_preprocessor: OptConfigType = None,
        single_img_loss: bool = False,
        depad_gt: bool = True,
        loss_enhance=dict(type='lqit.L1Loss', loss_weight=1.0),
        init_cfg: OptMultiConfig = dict(
            type='Normal', layer='Conv2d', std=0.01)
    ) -> None:
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
        self.single_img_loss = single_img_loss
        self.depad_gt = depad_gt
        self._init_layers()

    def _init_layers(self):
        assert self.num_convs > 0
        enhance_conv = []
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 \
                else self.feat_channels
            enhance_conv.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            if i == self.num_convs - 1:
                enhance_conv.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=3,
                        kernel_size=3,
                        stride=1,
                        padding=1))
        self.enhance_conv = nn.Sequential(*enhance_conv)

    def forward(self, x):
        if len(x) > 1 and (isinstance(x, tuple) or isinstance(x, list)):
            feat = x[0]
        elif isinstance(x, torch.Tensor):
            feat = x
        else:
            raise TypeError('The type of the input of enhance head should be '
                            'a list/tuple of Tensor or Tensor, but got '
                            f'{type(x)}')
        outs = self.enhance_conv(feat)
        return outs

    def loss_by_feat(self, batch_enhance_img, batch_gt_img, batch_img_metas):
        if self.single_img_loss:
            losses = []
            for i in range(len(batch_img_metas)):
                losses.append(
                    self.loss_by_feat_single(
                        enhance_img=batch_enhance_img[i, ...],
                        gt_img=batch_gt_img[i, ...],
                        img_meta=batch_img_metas[i]))
            enhance_loss = sum(losses) / len(losses)
        else:
            if self.depad_gt:
                weights = self.get_loss_weights(
                    batch_gt_img=batch_gt_img, batch_img_metas=batch_img_metas)
            else:
                weights = torch.ones_like(batch_gt_img)

            reshape_batch_gt_img = F.interpolate(
                batch_gt_img,
                size=batch_enhance_img.shape[-2:],
                mode='bilinear')
            reshape_weights = F.interpolate(
                weights, size=batch_enhance_img.shape[-2:], mode='bilinear')
            enhance_loss = self.loss_enhance(
                batch_enhance_img,
                reshape_batch_gt_img,
                weight=reshape_weights,
                avg_factor=reshape_weights.sum())

        return dict(loss_enhance=enhance_loss)

    def loss_by_feat_single(self, enhance_img, gt_img, img_meta):
        if self.depad_gt:
            h, w = img_meta['img_shape']
            no_padding_gt_img = gt_img[:, :h, :w][None, ...]
            no_padding_enhance_img = enhance_img[:, :h, :w][None, ...]
            no_padding_gt_img = F.interpolate(
                no_padding_gt_img,
                size=no_padding_enhance_img.shape[-2:],
                mode='bilinear')
            enhance_loss = self.loss_enhance(no_padding_enhance_img,
                                             no_padding_gt_img)
        else:
            enhance_loss = self.loss_enhance(enhance_img[None, ...],
                                             gt_img[None, ...])
        return enhance_loss


@MODELS.register_module()
class UpSingleEnhanceHead(SingleEnhanceHead):

    def __init__(self,
                 *args,
                 up_range=2,
                 upsample_cfg=dict(mode='bilinear'),
                 **kwargs):
        self.up_range = up_range
        assert 'scale_factor' not in upsample_cfg and \
               'size' not in upsample_cfg
        self.upsample_cfg = upsample_cfg.copy()
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        assert self.num_convs > 0 and self.up_range >= 1

        self.enhance_convs = nn.ModuleList()
        for j in range(self.up_range + 1):
            enhance_conv = []
            for i in range(self.num_convs):
                in_channels = self.in_channels if \
                    (j == 0 and i == 0) else self.feat_channels
                enhance_conv.append(
                    ConvModule(
                        in_channels=in_channels,
                        out_channels=self.feat_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if j == self.up_range and i == self.num_convs - 1:
                    enhance_conv.append(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=3,
                            kernel_size=3,
                            stride=1,
                            padding=1))
            self.enhance_convs.append(nn.Sequential(*enhance_conv))

    def forward(self, x):
        if len(x) > 1 and (isinstance(x, tuple) or isinstance(x, list)):
            feat = x[0]
        elif isinstance(x, torch.Tensor):
            feat = x
        else:
            raise TypeError('The type of the input of enhance head should be '
                            'a list/tuple of Tensor or Tensor, but got '
                            f'{type(x)}')
        assert self.batch_input_shape is not None
        input_shape = feat.shape[-2:]
        gt_shape = self.batch_input_shape
        scare_factor = (gt_shape[0] / input_shape[0]) / self.up_range
        for i in range(self.up_range):
            feat = self.enhance_convs[i](feat)
            if i == (self.up_range - 1):
                feat = F.interpolate(feat, size=gt_shape, **self.upsample_cfg)
            else:
                feat = F.interpolate(
                    feat, scale_factor=scare_factor, **self.upsample_cfg)

        outs = self.enhance_convs[self.up_range](feat)
        return outs

    def loss_by_feat(self, batch_enhance_img, batch_gt_img, batch_img_metas):
        if self.single_img_loss:
            losses = []
            for i in range(len(batch_img_metas)):
                losses.append(
                    self.loss_by_feat_single(
                        enhance_img=batch_enhance_img[i, ...],
                        gt_img=batch_gt_img[i, ...],
                        img_meta=batch_img_metas[i]))
            enhance_loss = sum(losses) / len(losses)
        else:
            if self.depad_gt:
                weights = self.get_loss_weights(
                    batch_gt_img=batch_gt_img, batch_img_metas=batch_img_metas)
            else:
                weights = torch.ones_like(batch_gt_img)
            enhance_loss = self.loss_enhance(
                batch_enhance_img,
                batch_gt_img,
                weight=weights,
                avg_factor=weights.sum())
        return dict(loss_enhance=enhance_loss)

    def loss_by_feat_single(self, enhance_img, gt_img, img_meta):
        if self.depad_gt:
            h, w = img_meta['img_shape']
            no_padding_gt_img = gt_img[:, :h, :w][None, ...]
            no_padding_enhance_img = enhance_img[:, :h, :w][None, ...]
            enhance_loss = self.loss_enhance(no_padding_enhance_img,
                                             no_padding_gt_img)
        else:
            enhance_loss = self.loss_enhance(enhance_img[None, ...],
                                             gt_img[None, ...])
        return enhance_loss


@MODELS.register_module()
class BasicEnhanceHead(BaseEnhanceHead):
    """[(convs)+ShufflePixes] * 2"""

    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 num_convs=2,
                 conv_cfg=None,
                 norm_cfg=None,
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
                        out_channels=3,
                        kernel_size=3,
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


@MODELS.register_module()
class CombineEnhanceHead(BaseEnhanceHead):
    """[Conv-BN-ReLU] * num_convs"""

    def __init__(
        self,
        in_channels: int = 256,
        feat_channels: int = 128,
        start_level: int = 0,
        end_level: int = 4,
        num_convs: int = 2,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = dict(type='ReLU'),
        gt_preprocessor: OptConfigType = None,
        loss_enhance=dict(type='lqit.FFTLoss', loss_weight=1.0),
        init_cfg: OptMultiConfig = dict(
            type='Normal', layer='Conv2d', std=0.01)
    ) -> None:
        super().__init__(
            loss_enhance=loss_enhance,
            gt_preprocessor=gt_preprocessor,
            init_cfg=init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.start_level = start_level
        self.end_level = end_level
        self.num_convs = num_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the enhance head."""
        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                convs_per_level.add_module(
                    f'conv{i}',
                    ConvModule(
                        self.in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=False))
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    convs_per_level.add_module(
                        f'conv{j}',
                        ConvModule(
                            self.in_channels,
                            self.feat_channels,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            inplace=False))
                    convs_per_level.add_module(
                        f'upsample{j}',
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False))
                    continue

                convs_per_level.add_module(
                    f'conv{j}',
                    ConvModule(
                        self.feat_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        inplace=False))
                convs_per_level.add_module(
                    f'upsample{j}',
                    nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False))

            self.convs_all_levels.append(convs_per_level)

        enhance_conv = []
        for i in range(self.num_convs + 1):
            if i < self.num_convs:
                enhance_conv.append(
                    ConvModule(
                        self.feat_channels,
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
                        in_channels=self.feat_channels,
                        out_channels=3,
                        kernel_size=3,
                        stride=1,
                        padding=1))
        self.enhance_conv = nn.Sequential(*enhance_conv)

    def forward(self, x):
        inputs = x[self.start_level:self.end_level + 1]
        assert len(inputs) == (self.end_level - self.start_level + 1)
        feature_add_all_level = self.convs_all_levels[0](inputs[0])
        for i in range(1, len(inputs)):
            input_p = inputs[i]
            feature_add_all_level = feature_add_all_level + \
                self.convs_all_levels[i](input_p)
        outs = self.enhance_conv(feature_add_all_level)
        return outs

    def loss(self, x, batch_data_samples):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        self.batch_input_shape = batch_img_metas[0]['batch_input_shape']
        outs = self(x)
        batch_gt_pixel = self.gt_preprocessor(
            batch_data_samples, training=True)

        # single gt
        batch_enhance_img = F.interpolate(
            outs, size=batch_gt_pixel.shape[-2:], mode='bilinear')
        loss_inputs = (batch_enhance_img, ) + (batch_gt_pixel, batch_img_metas)

        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_and_predict(self, x, batch_data_samples):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        self.batch_input_shape = batch_img_metas[0]['batch_input_shape']
        outs = self(x)
        batch_gt_pixel = self.gt_preprocessor(
            batch_data_samples, training=True)

        # single gt
        batch_enhance_img = F.interpolate(
            outs, size=batch_gt_pixel.shape[-2:], mode='bilinear')
        loss_inputs = (batch_enhance_img, ) + (batch_gt_pixel, batch_img_metas)

        losses = self.loss_by_feat(*loss_inputs)
        return losses, batch_enhance_img

    def loss_by_feat(self, batch_enhance_img, batch_gt_img, batch_img_metas):
        losses = []
        for i in range(len(batch_img_metas)):
            losses.append(
                self.loss_by_feat_single(
                    enhance_img=batch_enhance_img[i, ...],
                    gt_img=batch_gt_img[i, ...],
                    img_meta=batch_img_metas[i]))
        return dict(loss_enhance=losses)

    def loss_by_feat_single(self, enhance_img, gt_img, img_meta):
        h, w = img_meta['img_shape']
        assert len(gt_img.shape) == len(enhance_img.shape) == 3
        no_padding_gt_img = gt_img[:, :h, :w]
        no_padding_enhance_img = enhance_img[..., :h, :w]
        if hasattr(self.loss_enhance, 'fft_meta'):
            no_padding_enhance_img = self.gt_preprocessor.destructor(
                no_padding_enhance_img, img_meta, rescale=False)
            no_padding_gt_img = self.gt_preprocessor.destructor(
                no_padding_gt_img, img_meta, rescale=False)
            enhance_loss = self.loss_enhance(
                no_padding_enhance_img,
                no_padding_gt_img,
                fft_meta=img_meta['fft_meta'])
        else:
            enhance_loss = self.loss_enhance(no_padding_enhance_img,
                                             no_padding_gt_img)
        return enhance_loss
