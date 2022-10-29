# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmdet.utils import OptMultiConfig
from mmengine.model import BaseModule
from mmengine.utils import is_list_of
from torch import Tensor

from lqit.common.structures import SampleList
from lqit.registry import MODELS


class BaseEnhanceHead(BaseModule, metaclass=ABCMeta):
    """Base class for EnhanceHead."""

    def __init__(self,
                 loss_enhance=dict(type='mmdet.L1Loss', loss_weight=1.0),
                 gt_preprocessor=None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.loss_enhance = MODELS.build(loss_enhance)
        self.gt_preprocessor = self.build_gt_processor(gt_preprocessor)
        self.batch_input_shape = None

    @abstractmethod
    def loss_by_feat(self, *args, **kwargs):
        """Calculate the loss based on the features extracted by the enhance
        head."""
        pass

    def predict_by_feat(self,
                        batch_enhance_img,
                        batch_img_metas,
                        rescale=False):
        enhance_img_list = []
        for i in range(len(batch_img_metas)):
            enhance_img = batch_enhance_img[i, ...]
            enhance_img = self.gt_preprocessor.destructor(
                enhance_img, batch_img_metas[i], rescale=rescale)
            enhance_img_list.append(enhance_img)
        return enhance_img_list

    def loss(self, x: Union[List[Tensor], Tuple[Tensor]],
             batch_data_samples: SampleList):
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
        return losses

    def loss_and_predict(self, x: Union[List[Tensor], Tuple[Tensor]],
                         batch_data_samples: SampleList):
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
        predict_input = outs + (batch_img_metas, )
        predictions = self.predict_by_feat(*predict_input, rescale=False)
        return losses, predictions

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(x)
        if isinstance(outs, Tensor):
            outs = (outs, )
        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)

        return predictions

    @staticmethod
    def get_loss_weights(batch_gt_img, batch_img_metas):
        weights = torch.zeros_like(batch_gt_img)
        for i in range(len(batch_img_metas)):
            h, w = batch_img_metas[i]['img_shape']
            weights[i, :, :h, :w] = 1
        return weights

    @staticmethod
    def build_gt_processor(gt_preprocessor):
        if gt_preprocessor is None:
            gt_preprocessor = dict(type='lqit.BasePixelPreprocessor')
        if isinstance(gt_preprocessor, nn.Module):
            gt_preprocessor = gt_preprocessor
        elif isinstance(gt_preprocessor, dict):
            gt_preprocessor = MODELS.build(gt_preprocessor)
        else:
            raise TypeError('gt_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(gt_preprocessor)}')
        return gt_preprocessor
