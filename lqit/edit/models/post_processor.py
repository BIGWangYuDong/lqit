from typing import List, Union

from mmengine.structures import PixelData
from torch import Tensor

from lqit.common.structures import SampleList


def add_pixel_pred_to_datasample(data_samples: SampleList,
                                 pixel_list: Union[List[PixelData],
                                                   List[Tensor]],
                                 key: str = 'pred_img') -> SampleList:
    """Add predictions to `DataSample`.

    Args:
        data_samples (list[:obj:`DataSample`]): A batch of
            data samples that contain annotations and predictions.
        pixel_list (list[Tensor]): Pixel results of
            each image.
        key (str): The name of the pred_instance. Defaults to pred_img.
    Returns:
        list[:obj:`DetDataSample`]: Results of the input images.
    """
    for data_sample, pred_pixel in zip(data_samples, pixel_list):
        assert isinstance(pred_pixel, Tensor)
        if data_sample.get('pred_pixel', None) is None:
            pred_instance = PixelData()
        else:
            pred_instance = data_sample.pred_pixel
        keys = pred_instance.keys()
        assert key not in keys, f'{key} is already in pred_pixel'
        pred_instance.set_data({key: pred_pixel})
        data_sample.pred_pixel = pred_instance
    return data_samples
