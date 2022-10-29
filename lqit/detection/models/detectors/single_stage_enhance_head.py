# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmdet.models import SingleStageDetector
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from torch import Tensor

from lqit.common.structures import SampleList
from lqit.edit.models import add_pixel_pred_to_datasample


@MODELS.register_module()
class SingleStageWithEnhanceHead(SingleStageDetector):
    """Base class for two-stage detectors with enhance head.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 enhance_head: OptConfigType = None,
                 vis_enhance: Optional[bool] = False,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        if enhance_head is not None:
            self.enhance_head = MODELS.build(enhance_head)
        self.vis_enhance = vis_enhance

    @property
    def with_enhance_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'enhance_head') and self.enhance_head is not None

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        if self.with_enhance_head:
            enhance_outs = self.enhance_head.forward(x)
            results = results + (enhance_outs, )
        return results

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)

        losses = dict()
        if self.with_enhance_head:

            enhance_loss = self.enhance_head.loss(x, batch_data_samples)
            # avoid loss override
            assert not set(enhance_loss.keys()) & set(losses.keys())
            losses.update(enhance_loss)

        det_losses = self.bbox_head.loss(x, batch_data_samples)
        losses.update(det_losses)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)

        if self.vis_enhance and self.with_enhance_head:
            enhance_list = self.enhance_head.predict(
                x, batch_data_samples, rescale=rescale)
            batch_data_samples = add_pixel_pred_to_datasample(
                data_samples=batch_data_samples, pixel_list=enhance_list)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
