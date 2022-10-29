from numbers import Number
from typing import Sequence, Union

import torch
import torch.nn.functional as F
from mmengine.model import ImgDataPreprocessor

from lqit.registry import MODELS


@MODELS.register_module()
class GTPixelPreprocessor(ImgDataPreprocessor):

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 element_name: Union[str, list] = 'img'):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr)
        assert isinstance(element_name, str)
        self.element_name = element_name

        batched_output_view = [-1, 1, 1]
        self.register_buffer('outputs_mean',
                             torch.tensor(mean).view(batched_output_view),
                             False)
        self.register_buffer('outputs_std',
                             torch.tensor(std).view(batched_output_view),
                             False)
        self.norm_input_flag = None  # If input is normalized to [0, 1]

    def forward(self, batch_data_samples, training=True):
        data = {}
        gt_pixel_list = []
        for data_samples in batch_data_samples:
            gt_pixel = data_samples.gt_pixel.get(self.element_name)
            assert gt_pixel is not None
            gt_pixel_list.append(gt_pixel)
        data['inputs'] = gt_pixel_list
        data = super().forward(data=data, training=training)
        inputs = data['inputs']
        self.norm_input_flag = (inputs.max() >= 128)
        if self.norm_input_flag:
            inputs = inputs / 255
            self.norm_input_flag = True
        return inputs

    def destructor(self, img_tensor, img_meta, rescale):
        # De-normalization
        img_tensor = img_tensor * self.outputs_std + self.outputs_mean

        h, w = img_meta['img_shape']
        no_padding_img = img_tensor[:, :h, :w]

        assert self.norm_input_flag is not None, (
            'Please kindly run `forward` before running `destructor`')
        if self.norm_input_flag:
            no_padding_img *= 255
        no_padding_img = no_padding_img.clamp_(0, 255)
        if rescale:
            ori_h, ori_w = img_meta['ori_shape']
            # TODO: check whether use torch.functional or mmcv.resize
            no_padding_img = F.interpolate(
                no_padding_img[None, ...],
                size=(ori_h, ori_w),
                mode='bilinear')[0]
        return no_padding_img


@MODELS.register_module()
class MultiGTPixelPreprocessor(ImgDataPreprocessor):

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 element_name: list = ['img']):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr)
        assert isinstance(element_name, list)
        self.element_name = element_name

    def forward(self, batch_data_samples, training=True):
        batch_list = []
        for name in self.element_name:
            data = {}
            gt_pixel_list = []
            for data_samples in batch_data_samples:
                gt_pixel = data_samples.gt_pixel.get(name)
                assert gt_pixel is not None
                gt_pixel_list.append(gt_pixel)
            data['inputs'] = gt_pixel_list
            data = super().forward(data=data, training=training)
            batch_list.append(data['inputs'])
        return batch_list
