import warnings
from typing import Dict, List, Tuple, Union

import torch
from mmengine.model import BaseModel
from torch import Tensor

from lqit.common.structures import DataSample, OptSampleList, SampleList
from lqit.edit.structures import BatchPixelData
from lqit.registry import MODELS
from lqit.utils.typing import OptMultiConfig
from ..post_processor import add_pixel_pred_to_datasample

ForwardResults = Union[Dict[str, torch.Tensor], List[DataSample],
                       Tuple[torch.Tensor], torch.Tensor]


@MODELS.register_module()
class BaseEditModel(BaseModel):
    """Base class for edit model."""

    def __init__(self,
                 generator,
                 data_preprocessor=None,
                 gt_preprocessor=None,
                 destruct_gt=False,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        # build generator
        self.generator = MODELS.build(generator)

        # gt preprocessor is going to deal with edit model not going through
        # the forward function, which means it directly called `loss`,
        # `predict` ,or `loss_and_predict` outside. If gt_preprocessor is
        # not None, gt will stack by `gt_preprocessor`, else
        # `data_preprocessor.stack_gt`.
        self.gt_preprocessor = MODELS.build(
            gt_preprocessor) if gt_preprocessor else None

        self.destruct_gt = destruct_gt

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
            inputs (torch.Tensor): The input tensor with shape
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

    def _forward(self, batch_inputs: Tensor, *args, **kwargs):
        """Forward tensor. Returns result of simple forward.

        Args:
            batch_inputs (Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.

        Returns:
            Tensor: result of simple forward.
        """

        feats = self.generator(batch_inputs)
        return feats

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_outputs = self.generator(batch_inputs)
        if self.gt_preprocessor is None:
            batch_gt_pixel = self.data_preprocessor.stack_gt(
                batch_data_samples)
        else:
            batch_gt_pixel = self.gt_preprocessor(
                batch_data_samples, training=True)

        loss_input = BatchPixelData()
        loss_input.output = batch_outputs
        loss_input.gt = batch_gt_pixel
        loss_input.input = batch_inputs

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        if self.destruct_gt:
            de_batch_outputs = self.destructor_batch(batch_outputs,
                                                     batch_img_metas)
            de_batch_gt_pixel = self.destructor_batch(batch_gt_pixel,
                                                      batch_img_metas)
            de_batch_inputs = self.destructor_batch(batch_inputs,
                                                    batch_img_metas)

            loss_input.de_output = de_batch_outputs
            loss_input.de_gt = de_batch_gt_pixel
            loss_input.de_input = de_batch_inputs

        losses = self.generator.loss(loss_input, batch_img_metas)
        return losses

    def loss_and_predict(self, batch_inputs: Tensor,
                         batch_data_samples: SampleList) -> tuple:
        """Calculate losses and predict results from a batch of inputs and data
        samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DataSample`]): The batch
                data samples.

        Returns:
            Tuple: A dictionary of loss components and results of the
            input images.
        """
        # batch_outputs should be a tensor, if it should returns a tuple,
        # it is suggested to concat together, If cannot concat together,
        # a subclass can be set.
        batch_outputs = self.generator(batch_inputs)
        if self.gt_preprocessor is None:
            batch_gt_pixel = self.data_preprocessor.stack_gt(
                batch_data_samples)
        else:
            batch_gt_pixel = self.gt_preprocessor(
                batch_data_samples, training=True)

        loss_input = BatchPixelData()
        loss_input.output = batch_outputs
        loss_input.gt = batch_gt_pixel
        loss_input.input = batch_inputs

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        if self.destruct_gt:
            de_batch_outputs = self.destructor_batch(batch_outputs,
                                                     batch_img_metas)
            de_batch_gt_pixel = self.destructor_batch(batch_outputs,
                                                      batch_img_metas)
            de_batch_inputs = self.destructor_batch(batch_inputs,
                                                    batch_img_metas)

            loss_input.de_output = de_batch_outputs
            loss_input.de_gt = de_batch_gt_pixel
            loss_input.de_input = de_batch_inputs

        losses = self.generator.loss(loss_input, batch_img_metas)

        predictions = loss_input.get('de_output')
        if predictions is None:
            predictions = self.destructor_results(batch_outputs,
                                                  batch_img_metas)

        return losses, predictions

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DataSample`]): The Data
                Samples.

        Returns:
            list[:obj:`SampleList`]: Results of the
            input images. Each DataSample usually contain
            'pred_instances'.
        """
        batch_outputs = self.generator(batch_inputs)

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        results_list = self.destructor_results(batch_outputs, batch_img_metas)

        predictions = add_pixel_pred_to_datasample(
            data_samples=batch_data_samples, pixel_list=results_list)
        return predictions

    def destructor_results(self, batch_outputs: Tensor,
                           batch_img_metas: List[dict]) -> list:
        results_list = []
        for i in range(len(batch_img_metas)):
            outputs = batch_outputs[i, ...]
            if outputs.size(0) > 3:
                warnings.warn('The channel of the outputs is larger than 3, '
                              'it should call self.generator.post_precess '
                              'to get the output image.')
                outputs = self.generator.post_precess(outputs)
            img_meta = batch_img_metas[i]
            if self.gt_preprocessor is None:
                outputs = self.data_preprocessor.destructor(outputs, img_meta)
            else:
                norm_input_flag = self.gt_preprocessor.norm_input_flag
                if norm_input_flag is None:
                    # TODO: check
                    norm_input_flag = (batch_outputs.max() <= 1)
                outputs = self.gt_preprocessor.destructor(
                    outputs, img_meta, norm_input_flag=norm_input_flag)
            results_list.append(outputs)

        return results_list

    def destructor_batch(self, batch_outputs: Tensor,
                         batch_img_metas: List[dict]) -> Tensor:
        result_list = self.destructor_results(batch_outputs, batch_img_metas)
        if self.gt_preprocessor is None:
            destructor_batch = self.data_preprocessor.stack_batch(result_list)
        else:
            destructor_batch = self.gt_preprocessor.stack_batch(result_list)

        return destructor_batch
