import copy

import torch
from mmengine.model import BaseDataPreprocessor

from lqit.registry import MODELS
from lqit.utils import ConfigType


@MODELS.register_module()
class BatchDataPreprocessor(BaseDataPreprocessor):

    def __init__(self,
                 data_preprocessor: ConfigType,
                 multi_input_key: str = 'data'):
        super().__init__()
        self.data_preprocessor = MODELS.build(data_preprocessor)

        assert isinstance(multi_input_key, str)
        self.multi_input_key = multi_input_key

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """

        multi_inputs = None
        if data.get('data_samples') is not None and \
                data['data_samples'][0].get('multi_input') is not None and \
                training:
            multi_inputs_list = []
            for data_sample in data['data_samples']:
                multi_inputs_list.append(
                    data_sample.multi_input.get(self.multi_input_key))
            # process multi inputs
            fake_data = copy.deepcopy(data)
            fake_data['inputs'] = multi_inputs_list
            fake_data = self.data_preprocessor(
                data=fake_data, training=training)
            multi_inputs = fake_data['inputs']

        data = self.data_preprocessor(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']

        if multi_inputs is not None and training:
            inputs = torch.cat([inputs, multi_inputs], dim=0)
            for i in range(len(data_samples)):
                data_samples.append(data_samples[i])

        return {'inputs': inputs, 'data_samples': data_samples}
