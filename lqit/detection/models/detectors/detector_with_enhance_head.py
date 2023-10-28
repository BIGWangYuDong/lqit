import copy
import warnings
from typing import Dict, Tuple, Union

import torch
from mmdet.models import SingleStageDetector, TwoStageDetector
from mmengine.model import BaseModel
from torch import Tensor

from lqit.common.structures import OptSampleList, SampleList
from lqit.edit.models import add_pixel_pred_to_datasample
from lqit.registry import MODELS
from lqit.utils import ConfigType, OptConfigType, OptMultiConfig

ForwardResults = Union[Dict[str, Tensor], SampleList, Tuple[Tensor], Tensor]


@MODELS.register_module()
class DetectorWithEnhanceHead(BaseModel):
    """Detector with enhance head.

    Args:
        detector (dict or ConfigDict): Config for detector.
        enhance_head (dict or ConfigDict, optional): Config for enhance head.
        process_gt_preprocessor (bool): Whether process `gt_preprocessor` same
            as the `data_preprocessor` in detector. Defaults to True.
        vis_enhance (bool): Whether to visualize the enhanced image during
            inference. Defaults to False.
        init_cfg (dict or ConfigDict, optional): The config to control the
           initialization. Defaults to None.
    """

    def __init__(self,
                 detector: ConfigType,
                 enhance_head: OptConfigType = None,
                 process_gt_preprocessor: bool = True,
                 vis_enhance: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        # process gt_preprocessor
        if enhance_head is not None and process_gt_preprocessor:
            enhance_head = self.process_gt_preprocessor(detector, enhance_head)

        # build data_preprocessor
        self.data_preprocessor = MODELS.build(detector['data_preprocessor'])

        # build detector
        self.detector = MODELS.build(detector)
        if isinstance(self.detector, SingleStageDetector):
            self.detector_type = 'SingleStage'
        elif isinstance(self.detector, TwoStageDetector):
            self.detector_type = 'TwoStage'
        else:
            raise TypeError(
                f'Only support SingleStageDetector and TwoStageDetector, '
                f'but got {type(self.detector)}.')
        # build enhance head
        if enhance_head is not None:
            self.enhance_head = MODELS.build(enhance_head)
        else:
            self.enhance_head = None
        if vis_enhance:
            assert self.with_enhance_head
        self.vis_enhance = vis_enhance

    @staticmethod
    def process_gt_preprocessor(detector, enhance_head):
        """Process the gt_preprocessor of enhance head."""
        data_preprocessor = detector.get('data_preprocessor', None)
        data_preprocessor_mean = data_preprocessor['mean']
        data_preprocessor_std = data_preprocessor['std']
        data_preprocessor_bgr_to_rgb = data_preprocessor['bgr_to_rgb']
        data_preprocessor_pad_size_divisor = \
            data_preprocessor['pad_size_divisor']

        gt_preprocessor = enhance_head.get('gt_preprocessor', None)
        gt_preprocessor_mean = gt_preprocessor['mean']
        gt_preprocessor_std = gt_preprocessor['std']
        gt_preprocessor_bgr_to_rgb = gt_preprocessor['bgr_to_rgb']
        gt_preprocessor_pad_size_divisor = gt_preprocessor['pad_size_divisor']

        if data_preprocessor_mean != gt_preprocessor_mean:
            warnings.warn(
                'the `mean` of data_preprocessor and gt_preprocessor'
                'are different, force to use the `mean` of data_preprocessor.')
            enhance_head['data_preprocessor']['mean'] = data_preprocessor_mean
        if data_preprocessor_std != gt_preprocessor_std:
            warnings.warn(
                'the `std` of data_preprocessor and gt_preprocessor'
                'are different, force to use the `std` of data_preprocessor.')
            enhance_head['data_preprocessor']['std'] = data_preprocessor_std
        if data_preprocessor_bgr_to_rgb != gt_preprocessor_bgr_to_rgb:
            warnings.warn(
                'the `bgr_to_rgb` of data_preprocessor and gt_preprocessor'
                'are different, force to use the `bgr_to_rgb` of '
                'data_preprocessor.')
            enhance_head['data_preprocessor']['bgr_to_rgb'] = \
                data_preprocessor_bgr_to_rgb
        if data_preprocessor_pad_size_divisor != \
                gt_preprocessor_pad_size_divisor:
            warnings.warn('the `pad_size_divisor` of data_preprocessor and '
                          'gt_preprocessor are different, force to use the '
                          '`pad_size_divisor` of data_preprocessor.')
            enhance_head['data_preprocessor']['pad_size_divisor'] = \
                data_preprocessor_pad_size_divisor
        return enhance_head

    @property
    def with_enhance_head(self) -> bool:
        """Whether has enhance head."""
        return hasattr(self, 'enhance_head') and self.enhance_head is not None

    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def calculate_det_loss(self, x: Tuple[Tensor],
                           batch_data_samples: SampleList) -> dict:
        """Calculate detection loss.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        if len(x) > 5:
            x = x[1:]
        if self.detector_type == 'SingleStage':
            losses = self.detector.bbox_head.loss(x, batch_data_samples)
        else:
            losses = dict()
            # RPN forward and loss
            if self.detector.with_rpn:
                proposal_cfg = self.detector.train_cfg.get(
                    'rpn_proposal', self.detector.test_cfg.rpn)
                rpn_data_samples = copy.deepcopy(batch_data_samples)
                # set cat_id of gt_labels to 0 in RPN
                for data_sample in rpn_data_samples:
                    data_sample.gt_instances.labels = \
                        torch.zeros_like(data_sample.gt_instances.labels)

                rpn_losses, rpn_results_list = \
                    self.detector.rpn_head.loss_and_predict(
                        x, rpn_data_samples, proposal_cfg=proposal_cfg)
                # avoid get same name with roi_head loss
                keys = rpn_losses.keys()
                for key in list(keys):
                    if 'loss' in key and 'rpn' not in key:
                        rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
                losses.update(rpn_losses)
            else:
                assert batch_data_samples[0].get('proposals', None) is not None
                # use pre-defined proposals in InstanceData for the second
                # stage to extract ROI features.
                rpn_results_list = [
                    data_sample.proposals for data_sample in batch_data_samples
                ]

            roi_losses = self.detector.roi_head.loss(x, rpn_results_list,
                                                     batch_data_samples)
            losses.update(roi_losses)

        return losses

    def predict_det_results(self,
                            x: Tuple[Tensor],
                            batch_data_samples: SampleList,
                            rescale: bool = True) -> SampleList:
        """Predict detection results.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
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
        if len(x) > 5:
            x = x[1:]
        if self.detector_type == 'SingleStage':
            results_list = self.detector.bbox_head.predict(
                x, batch_data_samples, rescale=rescale)
        else:
            assert self.detector.with_bbox, 'Bbox head must be implemented.'
            # If there are no pre-defined proposals, use RPN to get proposals
            if batch_data_samples[0].get('proposals', None) is None:
                rpn_results_list = self.detector.rpn_head.predict(
                    x, batch_data_samples, rescale=False)
            else:
                rpn_results_list = [
                    data_sample.proposals for data_sample in batch_data_samples
                ]

            results_list = self.detector.roi_head.predict(
                x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.detector.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def det_head_forward(self, x: Tuple[Tensor],
                         batch_data_samples: SampleList) -> tuple:
        """Forward process of detection head.

        Args:
            x (tuple[Tensor]): Tuple of multi-level img features.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple: A tuple of features from detector head (`bbox_head` in
                single-stage detector or `rpn_head` and `roi_head` in
                two-stage detector).
        """
        if len(x) > 5:
            x = x[1:]
        if self.detector_type == 'SingleStage':
            results = self.detector.bbox_head.forward(x)
        else:
            results = ()
            if self.detector.with_rpn:
                rpn_results_list = self.detector.rpn_head.predict(
                    x, batch_data_samples, rescale=False)
            else:
                assert batch_data_samples[0].get('proposals', None) is not None
                rpn_results_list = [
                    data_sample.proposals for data_sample in batch_data_samples
                ]
            roi_outs = self.detector.roi_head.forward(x, rpn_results_list,
                                                      batch_data_samples)
            results = results + (roi_outs, )
        return results

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
        x = self.detector.extract_feat(batch_inputs)

        results = self.det_head_forward(x, batch_data_samples)
        if self.vis_enhance:
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
        x = self.detector.extract_feat(batch_inputs)

        losses = dict()
        if self.with_enhance_head:
            enhance_loss = self.enhance_head.loss(x, batch_data_samples)
            # avoid loss override
            assert not set(enhance_loss.keys()) & set(losses.keys())
            losses.update(enhance_loss)

        det_losses = self.calculate_det_loss(x, batch_data_samples)
        # avoid loss override
        assert not set(det_losses.keys()) & set(losses.keys())
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
        x = self.detector.extract_feat(batch_inputs)
        batch_data_samples = self.predict_det_results(
            x, batch_data_samples, rescale=rescale)

        if self.vis_enhance:
            enhance_list = self.enhance_head.predict(
                x, batch_data_samples, rescale=rescale)
            batch_data_samples = add_pixel_pred_to_datasample(
                data_samples=batch_data_samples, pixel_list=enhance_list)

        return batch_data_samples
