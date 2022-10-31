import copy
from typing import Optional

import torch
from mmdet.models import TwoStageDetector
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.utils import is_list_of
from torch import Tensor

from lqit.common.structures import SampleList
from lqit.edit.models.post_processor import add_pixel_pred_to_datasample


@MODELS.register_module()
class TwoStageWithEnhanceModel(TwoStageDetector):
    """Base class for two-stage detectors with enhance head.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 enhance_model: OptConfigType = None,
                 vis_enhance: Optional[bool] = False,
                 enhance_pred: bool = False,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 loss_weight: list = [0.5, 0.5],
                 detach_enhance_img: bool = False,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        if enhance_model is not None:
            self.enhance_model = MODELS.build(enhance_model)
        self.vis_enhance = vis_enhance
        self.enhance_pred = enhance_pred

        assert isinstance(loss_weight, list) and len(loss_weight) == 2
        self.detach_enhance_img = detach_enhance_img
        self.loss_weight = loss_weight
        self.loss_name = ['raw', 'enhance']

    @property
    def with_enhance_model(self) -> bool:
        """bool: whether the detector has a Enhance Model"""
        return (hasattr(self, 'enhance_model')
                and self.enhance_model is not None)

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
        results = ()
        x = self.extract_feat(batch_inputs)

        if self.with_enhance_model and self.enhance_pred:
            enhance_outs = self.enhance_model._forward(batch_inputs)
            results = results + (enhance_outs, )

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_outs = self.roi_head.forward(x, rpn_results_list)
        results = results + (roi_outs, )
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
        x_raw = self.extract_feat(batch_inputs)

        losses = dict()

        if self.with_enhance_model:
            enhance_loss, enhance_results = \
                self.enhance_model.loss_and_predict(
                    batch_inputs, batch_data_samples)
            losses.update(enhance_loss)
            enhance_results = self.data_preprocessor(
                dict(inputs=enhance_results))['inputs']
            if self.detach_enhance_img:
                enhance_results = enhance_results.detach()
            x_enhance = self.extract_feat(enhance_results)
        else:
            # Ablation
            x_enhance = self.extract_feat(batch_inputs)

        # RPN forward and loss
        for i, x in enumerate([x_raw, x_enhance]):
            temp_losses = dict()
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                rpn_data_samples = copy.deepcopy(batch_data_samples)
                # set cat_id of gt_labels to 0 in RPN
                for data_sample in rpn_data_samples:
                    data_sample.gt_instances.labels = \
                        torch.zeros_like(data_sample.gt_instances.labels)

                rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                    x, rpn_data_samples, proposal_cfg=proposal_cfg)
                # avoid get same name with roi_head loss
                keys = rpn_losses.keys()
                for key in keys:
                    if 'loss' in key and 'rpn' not in key:
                        rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
                temp_losses.update(rpn_losses)
            else:
                assert batch_data_samples[0].get('proposals', None) is not None
                # use pre-defined proposals in InstanceData for the second
                # stage to extract ROI features.
                rpn_results_list = [
                    data_sample.proposals for data_sample in batch_data_samples
                ]
            roi_losses = self.roi_head.loss(x, rpn_results_list,
                                            batch_data_samples)
            temp_losses.update(roi_losses)

            for loss_name, loss_value in temp_losses.items():
                if 'loss' in loss_name:
                    if isinstance(loss_value, Tensor):
                        loss_value = loss_value * self.loss_weight[i]
                    elif is_list_of(loss_value, Tensor):
                        loss_value = sum(_loss.mean()
                                         for _loss in loss_value) * \
                                     self.loss_weight[i]
                losses[f'{self.loss_name[i]}_{loss_name}'] = loss_value
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
        if self.enhance_pred:
            enhance_list = self.enhance_model.predict(batch_inputs,
                                                      batch_data_samples)
            enhance_results = self.data_preprocessor(
                dict(inputs=enhance_list))['inputs']
            x = self.extract_feat(enhance_results)
        else:
            enhance_list = None
            x = self.extract_feat(batch_inputs)

        if self.vis_enhance and enhance_list is not None:
            # TODO: handle rescale case
            batch_data_samples = add_pixel_pred_to_datasample(
                data_samples=batch_data_samples, pixel_list=enhance_list)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
