from typing import List, Sequence

from mmengine.model import BaseDataPreprocessor
from mmengine.structures import BaseDataElement

from lqit.registry import MODELS
from lqit.utils import ConfigType

SampleList = List[BaseDataElement]


@MODELS.register_module()
class MIMBDataPreprocessor(BaseDataPreprocessor):

    def __init__(self, data_preprocessor: ConfigType):
        super().__init__()
        self.data_preprocessor = MODELS.build(data_preprocessor)

    def forward(self, data: dict, training: bool = False) -> dict:
        # multi input and multi batch
        if training:
            inputs, data_samples = data['inputs'], data['data_samples']
            assert isinstance(inputs, Sequence) and \
                   isinstance(data_samples, Sequence)
            assert len(inputs) == len(data_samples) and \
                   len(inputs[0]) == len(data_samples[0])

            new_inputs, new_data_samples = [], []

            for i in range(len(inputs)):
                new_inputs.extend(_input for _input in inputs[i])
                new_data_samples.extend(_input for _input in data_samples[i])

            new_data = {
                'inputs': new_inputs,
                'data_samples': new_data_samples,
            }
        else:
            new_data = data

        return self.data_preprocessor(new_data, training)
