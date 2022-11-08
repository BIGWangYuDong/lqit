from typing import Union

import numpy as np
import torch
from mmengine.structures import PixelData


class BatchPixelData(PixelData):
    """Data structure for batched pixel-level annnotations or predictions.

    Different from parent class:
        Support value.ndim == 4 for batched tensor.

    All data items in ``data_fields`` of ``PixelData`` meet the following
    requirements:

    - They all have 4 dimensions in orders of batch_size, channel, height,
      and width.
    - They should have the same height and width.
    """

    def __setattr__(self, name: str, value: Union[torch.Tensor, np.ndarray]):
        """Set attributes of ``PixelData``.

        If the dimension of value is 2 and its shape meet the demand, it
        will automatically expend its channel-dimension.

        Args:
            name (str): The key to access the value, stored in `PixelData`.
            value (Union[torch.Tensor, np.ndarray]): The value to store in.
                The type of value must be  `torch.Tensor` or `np.ndarray`,
                and its shape must meet the requirements of `PixelData`.
        """

        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f'{name} has been used as a '
                    f'private attribute, which is immutable. ')

        else:
            assert isinstance(value, (torch.Tensor, np.ndarray)), \
                f'Can set {type(value)}, only support' \
                f' {(torch.Tensor, np.ndarray)}'

            if self.shape:
                assert tuple(value.shape[-2:]) == self.shape, (
                    f'the height and width of '
                    f'values {tuple(value.shape[-2:])} is '
                    f'not consistent with'
                    f' the length of this '
                    f':obj:`PixelData` '
                    f'{self.shape} ')
            assert value.ndim == 4, \
                f'The dim of value must be 2, 3 or 4, but got {value.ndim}'

            # call BaseDataElement.__setattr__
            super(PixelData, self).__setattr__(name, value)
