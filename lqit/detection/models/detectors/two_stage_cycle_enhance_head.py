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
class CycleTwoStageWithEnhanceHead(TwoStageDetector):
    """Base class for two-stage detectors with enhance head.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 enhance_head: OptConfigType = None,
                 cycle_det: bool = True,
                 loss_weight: list = [0.5, 0.5],
                 vis_enhance: Optional[bool] = False,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
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

        if enhance_head is not None:
            self.enhance_head = MODELS.build(enhance_head)
        self.vis_enhance = vis_enhance
        self.cycle_det = cycle_det
        if cycle_det:
            assert len(loss_weight) == 2
            self.prefix_name = ['raw', 'enhance']
        self.loss_weight = loss_weight

    @property
    def with_enhance_head(self) -> bool:
        """bool: whether the detector has a Enhance head"""
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
        results = ()
        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        if self.with_enhance_head:
            enhance_outs = self.enhance_head.forward(x)
            results = results + (enhance_outs, )

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
        enhance_results = None
        if self.with_enhance_head:
            if not self.cycle_det:
                enhance_loss = self.enhance_head.loss(x_raw,
                                                      batch_data_samples)
            else:
                enhance_loss, enhance_results = \
                    self.enhance_head.loss_and_predict(
                        x_raw, batch_data_samples)

                enhance_img_list = [
                    result.pred_pixel.pred_img for result in enhance_results
                ]
                # get enhance_batch_inputs of detector
                enhance_data = {'inputs': enhance_img_list}
                enhance_det_data = self.data_preprocessor(enhance_data, True)

                if isinstance(enhance_det_data, dict):
                    enhance_batch_inputs = enhance_det_data['inputs']
                elif isinstance(enhance_det_data, (list, tuple)):
                    enhance_batch_inputs = enhance_det_data[0]
                else:
                    raise TypeError('Output of `data_preprocessor` should be '
                                    'list, tuple or dict, but got '
                                    f'{type(enhance_det_data)}')
            # avoid loss override
            assert not set(enhance_loss.keys()) & set(losses.keys())
            losses.update(enhance_loss)

        if enhance_results is None:
            # RPN forward and loss
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                  self.test_cfg.rpn)
                batch_data_samples[0].pop('gt_pixel')
                rpn_data_samples = copy.deepcopy(batch_data_samples)
                # set cat_id of gt_labels to 0 in RPN
                for data_sample in rpn_data_samples:
                    data_sample.gt_instances.labels = \
                        torch.zeros_like(data_sample.gt_instances.labels)

                rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                    x_raw, rpn_data_samples, proposal_cfg=proposal_cfg)
                # avoid get same name with roi_head loss
                keys = rpn_losses.keys()
                for key in keys:
                    if 'loss' in key and 'rpn' not in key:
                        rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
                losses.update(rpn_losses)
            else:
                assert batch_data_samples[0].get('proposals', None) is not None
                # use pre-defined proposals in InstanceData for the
                # second stage to extract ROI features.
                rpn_results_list = [
                    data_sample.proposals for data_sample in batch_data_samples
                ]
            roi_losses = self.roi_head.loss(x_raw, rpn_results_list,
                                            batch_data_samples)
            losses.update(roi_losses)
        else:
            x_enhance = self.extract_feat(enhance_batch_inputs)
            det_losses = dict()
            temp_losses = dict()
            for i, x in enumerate([x_raw, x_enhance]):
                if self.with_rpn:
                    proposal_cfg = self.train_cfg.get('rpn_proposal',
                                                      self.test_cfg.rpn)
                    rpn_data_samples = copy.deepcopy(batch_data_samples)
                    # set cat_id of gt_labels to 0 in RPN
                    for data_sample in rpn_data_samples:
                        data_sample.gt_instances.labels = \
                            torch.zeros_like(data_sample.gt_instances.labels)

                    rpn_losses, rpn_results_list = \
                        self.rpn_head.loss_and_predict(
                            x, rpn_data_samples, proposal_cfg=proposal_cfg)
                    # avoid get same name with roi_head loss
                    keys = rpn_losses.keys()
                    for key in keys:
                        if 'loss' in key and 'rpn' not in key:
                            rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
                    temp_losses.update(rpn_losses)
                else:
                    assert batch_data_samples[0].get('proposals',
                                                     None) is not None
                    # use pre-defined proposals in InstanceData for
                    # the second stage to extract ROI features.
                    rpn_results_list = [
                        data_sample.proposals
                        for data_sample in batch_data_samples
                    ]
                roi_losses = self.roi_head.loss(x, rpn_results_list,
                                                batch_data_samples)
                temp_losses.update(roi_losses)

                for name, value in temp_losses.items():
                    if 'loss' in name:
                        if isinstance(value, Tensor):
                            value = value * self.loss_weight[i]
                        elif is_list_of(value, Tensor):
                            value = [_v * self.loss_weight[i] for _v in value]
                    det_losses[f'{self.prefix_name[i]}_{name}'] = value

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

        if self.vis_enhance and self.with_enhance_head:
            enhance_list = self.enhance_head.predict(
                x, batch_data_samples, rescale=rescale)
            batch_data_samples = add_pixel_pred_to_datasample(
                data_samples=batch_data_samples, pixel_list=enhance_list)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
