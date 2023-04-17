import copy
from typing import Callable, Dict, List, Union

from mmcv.transforms import BaseTransform, Compose
from mmcv.transforms.utils import cache_random_params
from mmengine.utils import is_list_of

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

    Note:
        1. The transforms should be no pixel-level processing operations,
           such as random brightening, etc.
        2. The `TransformBroadcaster` in MMCV can achieve the same operation as
          `TransBroadcaster`, but need to set more complex parameters.

    Args:
        transforms (list[dict | callable]): Sequence of transform object or
            config dict to be wrapped.
        src_key (str): Source name of the key in the result dict from
            loading pipeline.
        dst_key (str or list[str]): Destination name of the key in the result
            dict from loading pipeline.
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

    def __init__(self, transforms: List[Transform], src_key: str,
                 dst_key: Union[str, List[str]]) -> None:
        self.transforms = Compose(transforms)
        assert isinstance(src_key, str)
        self.src_key = src_key

        if isinstance(dst_key, str):
            self.dst_key = [dst_key]
        elif is_list_of(dst_key, str):
            self.dst_key = dst_key
        else:
            raise TypeError('dst_key should be a str or a list of str, but '
                            f'got {type(dst_key)}')

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
        scatters = [data]
        for dst_key in self.dst_key:
            assert data.get(dst_key, None) is not None, \
                f'{dst_key} should be in the results.'

            cp_data = copy.deepcopy(data)
            cp_data[self.src_key] = cp_data[dst_key]
            scatters.append(cp_data)
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
            output_scatters (list[dict]): The output of the wrapped
                pipeline.

        Returns:
            dict: Updated result dict.
        """
        assert isinstance(output_scatters, list) and \
               isinstance(output_scatters[0], dict) and \
               len(output_scatters) == (len(self.dst_key) + 1)
        outputs = output_scatters[0]
        for i, dst_key in enumerate(self.dst_key):
            outputs[dst_key] = output_scatters[i + 1][self.src_key]
        return outputs
