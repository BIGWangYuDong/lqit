from typing import Optional

import mmcv
import mmengine
import numpy as np
from mmcv.transforms import BaseTransform

from lqit.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadGTImageFromFile(BaseTransform):
    """Load an image from file.

    Required Keys:

        - gt_img_path

    Modified Keys:

        - gt_img

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:``mmcv.imfrombytes``.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :func:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        path_key (str): The name that get gt image path from results dict.
            Defaults to 'gt_img_path'.
        results_key (str): The name that going to save gt image in the results
            dict. Defaults to 'gt_img'.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: dict = dict(backend='disk'),
                 ignore_empty: bool = False,
                 path_key: str = 'gt_img_path',
                 results_key: str = 'gt_img') -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.path_key = path_key
        self.results_key = results_key
        self.file_client_args = file_client_args.copy()
        self.file_client = mmengine.FileClient(**self.file_client_args)

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert self.path_key in results, \
            f'{self.path_key} not in results dict, please have a check'
        filename = results[self.path_key]

        if filename == results['img_path']:
            # Avoid loading the same image repeatedly
            img = results['img']
            results[self.results_key] = img
            return results

        try:
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        if self.to_float32:
            img = img.astype(np.float32)

        assert img.shape[:2] == results['img_shape'], \
            'gt image shape is not equal to img shape'
        results[self.results_key] = img
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', "
                    f'file_client_args={self.file_client_args}, '
                    f"path_key='{self.path_key}', "
                    f"results_key='{self.results_key}')")
        return repr_str


@TRANSFORMS.register_module()
class SetInputImageAsGT(BaseTransform):
    """Set the input image as gt image, and set it into results dict.

    Required Keys:

        - img

    Modified Keys:

        - gt_img

    Args:
        results_key (str): The name that going to save gt image in the results
            dict. Defaults to 'gt_img'.

    Note:
        This transforms should add before `PackInputs`. Otherwise, some
        transforms will change the `img` and do not change `gt_img`.

    Note:
        If the transforms change the `img` property, it is suggested to use
        `TransBroadcaster` instead.

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
        >>> # equal to
        >>> train_pipeline = [
        ...     dict(type='LoadImageFromFile'),
        ...     dict(type='LoadAnnotations', with_bbox=True),
        ...     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        ...     dict(type='RandomFlip', prob=0.5),
        ...     dict(type='lqit.SetInputImageAsGT'),
        ...     dict(type='lqit.PackInputs')]
    """

    def __init__(self, results_key: str = 'gt_img') -> None:
        self.results_key = results_key

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to set input image as gt image.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert 'img' in results
        img = results['img']
        results[self.results_key] = img
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"results_key='{self.results_key}')")
        return repr_str
