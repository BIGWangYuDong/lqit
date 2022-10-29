import copy
from typing import List, Tuple, Union

from mmdet.models.detectors import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils.typing import ConfigType, OptConfigType, OptMultiConfig
from mmengine.utils import is_list_of
from torch import Tensor


@MODELS.register_module()
class MultiInputDetectorWrapper(BaseDetector):

    def __init__(self,
                 detector: ConfigType,
                 multi_input_key: str = 'data',
                 loss_weight: list = [0.5, 0.5],
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.detector = MODELS.build(detector)

        if isinstance(multi_input_key, str):
            multi_input_key = [multi_input_key]
        else:
            assert is_list_of(multi_input_key, str)
        self.multi_input_key = multi_input_key

        self.input_name = ['raw']
        self.input_name.extend(self.multi_input_key)
        assert len(self.input_name) == len(loss_weight) and \
               is_list_of(loss_weight, float)
        self.loss_weight = loss_weight

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        # process multi input
        batch_multi_inputs = [batch_inputs]
        losses = dict()
        for input_key in self.multi_input_key:
            multi_inputs = self.data_preprocessor.process_multi_input(
                batch_data_samples=batch_data_samples, input_key=input_key)
            batch_multi_inputs.append(multi_inputs)

        for i, inputs in enumerate(batch_multi_inputs):
            cp_batch_data_samples = copy.deepcopy(batch_data_samples)
            temp_loss = self.detector.loss(inputs, cp_batch_data_samples)
            for loss_name, loss_value in temp_loss.items():
                if 'loss' in loss_name:
                    if isinstance(loss_value, Tensor):
                        loss_value = loss_value * self.loss_weight[i]
                    elif is_list_of(loss_value, Tensor):
                        loss_value = sum(_loss.mean()
                                         for _loss in loss_value) * \
                                     self.loss_weight[i]
                losses[f'{self.input_name[i]}_{loss_name}'] = loss_value
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        # TODO: Currently not support Aug Test
        # Raw input predict
        batch_data_samples = self.detector.predict(batch_inputs,
                                                   batch_data_samples, rescale)
        return batch_data_samples

    def _forward(self, *args, **kwargs) -> Tuple[List[Tensor]]:
        return self.detector._forward(*args, **kwargs)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        return self.detector.extract_feat(batch_inputs=batch_inputs)
