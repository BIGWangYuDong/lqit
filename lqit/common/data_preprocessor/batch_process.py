import math

import torch
import torch.nn.functional as F
from mmengine.model.utils import stack_batch as mmengine_stack_batch
from mmengine.utils import is_list_of


def stack_batch(tensor_list,
                pad_size_divisor=1,
                pad_value=0,
                channel_conversion=False,
                mean=None,
                std=None):
    assert (mean is None) == (std is None), (
        'mean and std should be both None or tuple')
    if mean is not None:
        assert len(mean) == 3 or len(mean) == 1, (
            '`mean` should have 1 or 3 values, to be compatible with '
            f'RGB or gray image, but got {len(mean)} values')
        assert len(std) == 3 or len(std) == 1, (  # type: ignore
            '`std` should have 1 or 3 values, to be compatible with RGB '  # type: ignore # noqa: E501
            f'or gray image, but got {len(std)} values')
        enable_normalize = True
        mean = torch.tensor(mean).view(-1, 1, 1)
        std = torch.tensor(std).view(-1, 1, 1)
    else:
        enable_normalize = False

    if is_list_of(tensor_list, torch.Tensor):
        batch_inputs = []
        for tensor in tensor_list:
            # channel transform
            if channel_conversion:
                tensor = tensor[[2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            tensor = tensor.float()
            # Normalization.
            if enable_normalize:
                if mean.shape[0] == 3:
                    assert tensor.dim() == 3 and tensor.shape[0] == 3, (
                        'If the mean has 3 values, the input tensor '
                        'should in shape of (3, H, W), but got the tensor '
                        f'with shape {tensor.shape}')
                tensor = (tensor - mean) / std
            batch_inputs.append(tensor)
        # Pad and stack Tensor.
        batch_inputs = mmengine_stack_batch(batch_inputs, pad_size_divisor,
                                            pad_value)
    # Process data with `default_collate`.
    elif isinstance(tensor_list, torch.Tensor):
        assert tensor_list.dim() == 4, (
            'The input of `ImgDataPreprocessor` should be a NCHW tensor '
            'or a list of tensor, but got a tensor with shape: '
            f'{tensor_list.shape}')
        if channel_conversion:
            tensor_list = tensor_list[:, [2, 1, 0], ...]
        # Convert to float after channel conversion to ensure
        # efficiency
        tensor_list = tensor_list.float()
        if enable_normalize:
            tensor_list = (tensor_list - mean) / std
        h, w = tensor_list.shape[2:]
        target_h = math.ceil(h / pad_size_divisor) * pad_size_divisor
        target_w = math.ceil(w / pad_size_divisor) * pad_size_divisor
        pad_h = target_h - h
        pad_w = target_w - w
        batch_inputs = F.pad(tensor_list, (0, pad_w, 0, pad_h), 'constant',
                             pad_value)
    else:
        raise TypeError('input data should be a list of Tensor or Tensor,'
                        f'but got {type(tensor_list)}： {tensor_list}')
    return batch_inputs
