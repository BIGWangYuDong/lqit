# Modified from https://github.com/open-mmlab/mmediting/tree/1.x/
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.model import BaseDataPreprocessor

from lqit.common.data_preprocessor import stack_batch
from lqit.registry import MODELS


@MODELS.register_module()
class EditDataPreprocessor(BaseDataPreprocessor):
    """Basic data pre-processor used for collating and copying data to the
    target device in mmediting.

    ``EditDataPreprocessor`` performs data pre-processing according to the
    following steps:

    - Collates the data sampled from dataloader.
    - Copies data to the target device.
    - Stacks the input tensor at the first dimension.

    and post-processing of the output tensor of model.

    Args:
        mean (Sequence[float or int]): The pixel mean of R, G, B channels.
            Defaults to (0, 0, 0). If ``mean`` and ``std`` are not
            specified, ImgDataPreprocessor will normalize images to [0, 1].
        std (Sequence[float or int]): The pixel standard deviation of R, G, B
            channels. (255, 255, 255). If ``mean`` and ``std`` are not
            specified, ImgDataPreprocessor will normalize images to [0, 1].
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        input_view (Tuple | List): Tensor view of mean and std for input
            (without batch). Defaults to (-1, 1, 1) for (C, H, W).
        gt_name (str): Element name.
    """

    def __init__(self,
                 mean: Sequence[Union[float, int]] = (0, 0, 0),
                 std: Sequence[Union[float, int]] = (255, 255, 255),
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 input_view=(-1, 1, 1),
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 gt_name: str = 'img') -> None:

        super().__init__()

        assert len(mean) == 3 or len(mean) == 1, (
            'The length of mean should be 1 or 3 to be compatible with RGB '
            f'or gray image, but got {len(mean)}')
        assert len(std) == 3 or len(std) == 1, (
            'The length of mean should be 1 or 3 to be compatible with RGB '
            f'or gray image, but got {len(std)}')

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')

        # reshape mean and std for input (without batch).
        self.register_buffer('mean',
                             torch.tensor(mean).view(input_view), False)
        self.register_buffer('std', torch.tensor(std).view(input_view), False)

        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb

        self.norm_input_flag = None  # If input is normalized to [0, 1]

        assert isinstance(gt_name, str)
        self.gt_name = gt_name

    def forward(self,
                data: Sequence[dict],
                training: bool = False) -> Tuple[torch.Tensor, Optional[list]]:
        """Pre-process the data into the model input format.

        After the data pre-processing of :meth:`collate_data`, ``forward``
        will stack the input tensor list to a batch tensor at the first
        dimension.

        Args:
            data (Sequence[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Default: False.

        Returns:
            Tuple[torch.Tensor, Optional[list]]: Data in the same format as the
            model input.
        """

        # inputs, batch_data_samples = self.collate_data(data)
        data = super().forward(data=data, training=training)
        inputs, batch_data_samples = data['inputs'], data['data_samples']

        # Check if input is normalized to [0, 1]
        self.norm_input_flag = (inputs[0].max() <= 1)

        # Normalization, pad, and stack Tensor.
        inputs = stack_batch(
            inputs,
            pad_size_divisor=self.pad_size_divisor,
            pad_value=self.pad_value,
            channel_conversion=self._channel_conversion,
            mean=self.mean,
            std=self.std)

        data['inputs'] = inputs
        data['data_samples'] = batch_data_samples
        return data

    def stack_gt(self, batch_data_samples):
        gt_pixel_list = []
        for data_sample in batch_data_samples:
            gt_pixel = data_sample.gt_pixel.get(self.gt_name)
            gt_pixel_list.append(gt_pixel)

        batch_gt_pixel = stack_batch(
            gt_pixel_list,
            pad_size_divisor=self.pad_size_divisor,
            pad_value=self.pad_value,
            channel_conversion=self._channel_conversion,
            mean=self.mean,
            std=self.std)

        return batch_gt_pixel

    def destructor(self,
                   img_tensor,
                   img_meta,
                   rescale=False,
                   norm_input_flag=None):
        assert img_tensor.dim() == 3
        # De-normalization
        img_tensor = img_tensor * self.std + self.mean

        h, w = img_meta['img_shape']
        no_padding_img = img_tensor[:, :h, :w]

        if norm_input_flag is not None:
            norm_input_flag_ = norm_input_flag
        else:
            norm_input_flag_ = self.norm_input_flag

        assert norm_input_flag_ is not None, (
            'Please kindly run `forward` before running `destructor` or '
            'set `norm_input_flag`.')
        if norm_input_flag_:
            no_padding_img *= 255
        no_padding_img = no_padding_img.clamp_(0, 255)

        if self._channel_conversion:
            no_padding_img = no_padding_img[[2, 1, 0], ...]

        # TODO: check whether need to move to api or vis hook
        if rescale:
            ori_h, ori_w = img_meta['ori_shape']
            # TODO: check whether use torch.functional or mmcv.resize
            no_padding_img = F.interpolate(
                no_padding_img[None, ...],
                size=(ori_h, ori_w),
                mode='bilinear')[0]
        return no_padding_img

    def stack_batch(self, batch_outputs):
        batch_outputs = stack_batch(
            batch_outputs,
            pad_size_divisor=self.pad_size_divisor,
            pad_value=self.pad_value,
            channel_conversion=False,
            mean=None,
            std=None)
        return batch_outputs
