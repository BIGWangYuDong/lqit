import copy

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import ModuleList

from lqit.registry import MODELS
from lqit.utils.typing import OptConfigType, OptMultiConfig
from ..post_processor import add_pixel_pred_to_datasample
from .base_head import BaseEnhanceHead


@MODELS.register_module()
class AENetEnhanceHead(BaseEnhanceHead):

    def __init__(
        self,
        in_channels=256,
        upscale_factor=4,
        num_convs=2,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = dict(type='ReLU'),
        gt_preprocessor: OptConfigType = None,
        enhance_loss=None,
        spacial_loss=None,
        tv_loss=None,
        structure_loss=None,
        init_cfg: OptMultiConfig = dict(
            type='Normal', layer='Conv2d', std=0.01)
    ) -> None:
        super().__init__(gt_preprocessor=gt_preprocessor, init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_convs = num_convs

        assert isinstance(upscale_factor, int)
        self.upscale_factor = upscale_factor

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        assert hasattr(self.gt_preprocessor, 'outputs_std')
        assert hasattr(self.gt_preprocessor, 'outputs_mean')
        self.outputs_std = self.gt_preprocessor.outputs_std
        self.outputs_mean = self.gt_preprocessor.outputs_mean

        if spacial_loss is not None:
            self.spacial_loss = MODELS.build(spacial_loss)
        else:
            self.spacial_loss = None

        if tv_loss is not None:
            self.tv_loss = MODELS.build(tv_loss)
        else:
            self.tv_loss = None

        if structure_loss is not None:
            self.structure_loss = MODELS.build(structure_loss)
        else:
            self.structure_loss = None

        if enhance_loss is not None:
            self.enhance_loss = MODELS.build(enhance_loss)
        else:
            self.enhance_loss = None

        self._init_layers()

    def _init_layers(self):
        assert 2 <= self.upscale_factor <= 4 or self.upscale_factor == 8
        if self.upscale_factor == 2 or self.upscale_factor == 4:
            self.upsample_blocks = ModuleList()
            for i in range(self.upscale_factor // 2):
                upsample_convs = []
                for _ in range(self.num_convs):
                    upsample_convs.append(
                        ConvModule(
                            self.in_channels,
                            self.in_channels,
                            3,
                            stride=1,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                upsample_convs.append(nn.PixelShuffle(2))
                self.upsample_blocks.append(nn.Sequential(*upsample_convs))
                self.in_channels = self.in_channels // 4
        elif self.upscale_factor == 8:
            self.upsample_blocks = ModuleList()
            for i in range(3):
                upsample_convs = []
                for _ in range(self.num_convs):
                    upsample_convs.append(
                        ConvModule(
                            self.in_channels,
                            self.in_channels,
                            3,
                            stride=1,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                upsample_convs.append(nn.PixelShuffle(2))
                self.upsample_blocks.append(nn.Sequential(*upsample_convs))
                self.in_channels = self.in_channels // 4
        else:
            upsample_convs = []
            for _ in range(self.num_convs):
                upsample_convs.append(
                    ConvModule(
                        self.in_channels,
                        self.in_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
            upsample_convs.append(nn.PixelShuffle(3))
            self.upsample_blocks.append(nn.Sequential(*upsample_convs))
            self.in_channels = self.in_channels // 9

        self.output = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, x):
        if len(x) > 1 and (isinstance(x, tuple) or isinstance(x, list)):
            feat = x[0]
        elif isinstance(x, torch.Tensor):
            feat = x
        else:
            raise TypeError('The type of the input of enhance head should be '
                            'a list/tuple of Tensor or Tensor, but got '
                            f'{type(x)}')
        for upsample_block in self.upsample_blocks:
            feat = upsample_block(feat)
        outs = self.output(feat)
        return outs

    def loss_and_predict(self, x, batch_data_samples):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        self.batch_input_shape = batch_img_metas[0]['batch_input_shape']
        outs = self(x)
        if isinstance(outs, Tensor):
            outs = (outs, )
        batch_gt_pixel = self.gt_preprocessor(
            batch_data_samples, training=True)
        if isinstance(batch_gt_pixel, Tensor):
            # single gt
            loss_inputs = outs + (batch_gt_pixel, batch_img_metas)
        elif is_list_of(batch_gt_pixel, Tensor):
            # multi gt
            loss_inputs = outs + (*batch_gt_pixel, batch_img_metas)
        else:
            raise TypeError('batch_gt_pixel should be a Tensor or a list of '
                            f'Tensor, but got {type(batch_gt_pixel)}')
        losses = self.loss_by_feat(*loss_inputs)
        results_list = self.predict_by_feat(*loss_inputs)

        cp_batch_data_samples = copy.deepcopy(batch_data_samples)
        predictions = add_pixel_pred_to_datasample(
            data_samples=cp_batch_data_samples, pixel_list=results_list)

        return losses, predictions

    def loss_by_feat(self, batch_enhance_img, batch_gt_img, batch_img_metas):
        # batch_gt = batch_input
        losses = dict()
        if self.tv_loss is not None:
            tv_loss = self.tv_loss(batch_enhance_img)
            losses['tv_loss'] = tv_loss

        if self.spacial_loss is not None:
            spacial_loss = self.spacial_loss(batch_enhance_img, batch_gt_img)
            losses['spacial_loss'] = spacial_loss

        if self.enhance_loss is not None:
            enhance_loss = self.enhance_loss(batch_enhance_img, batch_gt_img)
            losses['enhance_loss'] = enhance_loss

        if self.structure_loss is not None:
            # De-normalization
            de_batch_gt = self.destructor_batch(batch_gt_img, batch_img_metas)
            de_batch_enhance = self.destructor_batch(batch_enhance_img,
                                                     batch_img_metas)

            structure_loss = self.structure_loss(de_batch_enhance, de_batch_gt,
                                                 batch_img_metas)
            losses['structure_loss'] = structure_loss
        return losses

    def predict_by_feat(self,
                        batch_enhance_img,
                        batch_img_metas,
                        rescale=False):
        self.gt_preprocessor.norm_input_flag = False
        enhance_img_list = self.destructor_results(batch_enhance_img,
                                                   batch_img_metas)
        return enhance_img_list

    def destructor_results(self, batch_outputs, batch_img_metas):
        results_list = []
        for i in range(len(batch_img_metas)):
            outputs = batch_outputs[i, ...]
            img_meta = batch_img_metas[i]

            outputs = self.gt_preprocessor.destructor(outputs, img_meta)
            results_list.append(outputs)
        return results_list

    def destructor_batch(self, batch_outputs, batch_img_metas):
        result_list = self.destructor_results(batch_outputs, batch_img_metas)
        destructor_batch = self.gt_preprocessor.stack_batch(result_list)
        return destructor_batch
