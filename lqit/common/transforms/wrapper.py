import copy
from typing import Callable, Dict, List, Optional, Union

from mmcv.transforms import BaseTransform, Compose
from mmcv.transforms.utils import cache_random_params

from lqit.registry import TRANSFORMS

Transform = Union[Dict, Callable[[Dict], Dict]]


@TRANSFORMS.register_module()
class TransBroadcaster(BaseTransform):
    """A transform wrapper to apply the wrapped transforms to process both
    `src_key` and `dst_key` without adding any codes. It will do the following
    steps:

        1. Scatter the broadcasting targets to a list of inputs of the wrapped
           transforms. The type of the list should be list[dict, dict], which
           the first is the original inputs, the second is the processing
           results that `dst_key` being rewritten by the `src_key`.
        2. Apply ``self.transforms``, with same random parameters, which is
           sharing with a context manager. The type of the outputs is a
           list[dict, dict].
        3. Gather the outputs, update the `dst_key` in the first item of
           the outputs with the `src_key` in the second.

    NOTE: The transforms should be no pixel-level processing operations,
    such as random brightening, etc.

    Args:
        transforms (list[dict | callable]): Sequence of transform object or
            config dict to be wrapped.
        src_key (str): Source name of the key in the result dict from
            loading pipeline.
        dst_key (str): Destination name of the key in the result dict from
            loading pipeline.
    Examples:
        >>> pipeline = [
        ...     dict(type='LoadImageFromFile'),
        ...     dict(type='lqit.LoadGTImageFromFile'),
        ...     dict(type='LoadAnnotations', with_bbox=True),
        ...     dict(
        ...         type='lqit.TransBroadcaster',
        ...         src_key='img',
        ...         dst_key='gt_img',
        ...         transforms=[
        ...             dict(type='Resize', scale=(1333, 800),
        ...                  keep_ratio=True),
        ...             dict(type='RandomFlip', prob=0.5),
        ...         ]),
        ...     dict(type='lqit.PackInputs')]
    """

    def __init__(self, transforms: Optional[List[Transform]], src_key: str,
                 dst_key: str) -> None:
        if transforms is None:
            transforms = []
        self.transforms = Compose(transforms)
        self.src_key = src_key
        self.dst_key = dst_key

    def transform(self, results: dict) -> dict:
        """Apply wrapped transform functions to process both `src_key` and
        `dst_key`.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        assert results.get(self.src_key, None) is not None, \
            f'{self.src_key} should be in the results.'
        assert results.get(self.dst_key, None) is not None, \
            f'{self.dst_key} should be in the results.'

        inputs = self._process_input(results)
        outputs = self._apply_transforms(inputs)
        outputs = self._process_output(outputs)
        return outputs

    def _process_input(self, data: dict) -> list:
        """Scatter the broadcasting targets to a list of inputs of the wrapped
        transforms.

        Args:
            data (dict): The original input data.

        Returns:
            list[dict, dict]: A list of input data.
        """
        cp_data = copy.deepcopy(data)
        cp_data[self.src_key] = cp_data[self.dst_key]
        scatters = [data, cp_data]
        return scatters

    def _apply_transforms(self, inputs: list) -> list:
        """Apply ``self.transforms``.

        Args:
            inputs (list[dict, dict]): list of input data.

        Returns:
            list[dict, dict]: The output of the wrapped pipeline.
        """
        ctx = cache_random_params
        with ctx(self.transforms):
            output_scatters = [self.transforms(_input) for _input in inputs]
        return output_scatters

    def _process_output(self, output_scatters: list) -> dict:
        """Gathering and renaming data items.

        Args:
            output_scatters (list[dict, dict]): The output of the wrapped
                pipeline.

        Returns:
            dict: Updated result dict.
        """
        assert isinstance(output_scatters, list) and \
               isinstance(output_scatters[0], dict) and \
               len(output_scatters) == 2
        outputs = output_scatters[0]
        outputs[self.dst_key] = output_scatters[1][self.src_key]
        return outputs
