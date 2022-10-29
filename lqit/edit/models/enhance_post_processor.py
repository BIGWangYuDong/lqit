# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from mmengine.structures import PixelData

from lqit.common.structures import SampleList


def add_pixel_pred_to_datasample(data_samples: SampleList,
                                 pixel_list: List[PixelData]) -> SampleList:
    """Add predictions to `DataSample`.

    Args:
        data_samples (list[:obj:`DetDataSample`], optional): A batch of
            data samples that contain annotations and predictions.
        pixel_list (list[:obj:`PixelData`]): Pixel results of
            each image.
    Returns:
        list[:obj:`DetDataSample`]: Results of the input images.
    """
    for data_sample, pred_pixel in zip(data_samples, pixel_list):
        data_sample.pred_pixel = pred_pixel
    return data_samples
