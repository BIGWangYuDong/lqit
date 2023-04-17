import copy
import warnings
from typing import List

from mmengine.model import BaseDataPreprocessor
from mmengine.structures import BaseDataElement
from torch import Tensor

from lqit.registry import MODELS
from lqit.utils import ConfigType, OptConfigType

SampleList = List[BaseDataElement]


@MODELS.register_module()
class MultiInputDataPreprocessor(BaseDataPreprocessor):

    def __init__(self,
                 data_preprocessor: ConfigType,
                 multi_input_data_preprocessor: OptConfigType = None):
        super().__init__()
        self.data_preprocessor = MODELS.build(data_preprocessor)

        if multi_input_data_preprocessor is None:
            multi_input_data_preprocessor = data_preprocessor
        self.multi_input_data_preprocessor = \
            MODELS.build(multi_input_data_preprocessor)
        self._training = None

    def forward(self, data: dict, training: bool = False) -> dict:
        self._training = training
        return self.data_preprocessor(data, training)

    def process_multi_input(self,
                            batch_data_samples: SampleList,
                            input_key: str = 'data'):
        cp_batch_data_samples = copy.deepcopy(batch_data_samples)
        input_list = [
            data_samples.multi_input.get(input_key)
            for data_samples in cp_batch_data_samples
        ]
        fake_data = dict(inputs=input_list, data_samples=cp_batch_data_samples)

        if self._training is None:
            warnings.warn('training will set to `self.training`, '
                          'which may get some potential error.'
                          'Please kindly run `forward` '
                          'before running `process_multi_input`.')
            self._training = self.training

        multi_data = self.multi_input_data_preprocessor(
            fake_data, self._training)
        del cp_batch_data_samples

        batched_multi_input = multi_data['inputs']
        return batched_multi_input

    def destructor_multi_input(self, batch_tensor: List[Tensor]):
        # TODO: Support for visualization
        pass
