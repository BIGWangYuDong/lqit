import copy
import logging
import os.path as osp
import warnings
from typing import Any, List, Union

import numpy as np
from mmengine.dataset import BaseDataset, force_full_init
from mmengine.logging import print_log

from lqit.registry import DATASETS


@DATASETS.register_module()
class DatasetWithGTImageWrapper:
    """Dataset wrapper that add `gt_image_path` simultaneously. The `gt_image`
    name should have same as image name.

    Args:
        dataset (BaseDataset or dict): The dataset
        suffix (str): gt_image suffix. Defaults to 'jpg'.
        lazy_init (bool, optional): whether to load annotation during
            instantiation. Defaults to False
    """

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 suffix: str = 'jpg',
                 lazy_init: bool = False) -> None:
        self.suffix = suffix
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')
        self._metainfo = self.dataset.metainfo

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the repeated dataset.

        Returns:
            dict: The meta information of repeated dataset.
        """
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return
        self.dataset.full_init()
        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the dataset.
        """

        return self.dataset.get_data_info(idx)

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        data_info = self.parse_gt_img_info(data_info)
        return self.dataset.pipeline(data_info)

    def parse_gt_img_info(self, data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """

        gt_img_root = self.dataset.data_prefix.get('gt_img_path', None)

        if gt_img_root is None:
            warnings.warn(
                'Cannot get gt_img_root, please set `gt_img_path` in '
                '`dataset.data_prefix`')
            data_info['gt_img_path'] = data_info['img_path']
        else:
            img_name = \
                f"{osp.split(data_info['img_path'])[-1].split('.')[0]}" \
                f'.{self.suffix}'
            data_info['gt_img_path'] = osp.join(gt_img_root, img_name)
        return data_info

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        if not self._fully_initialized:
            print_log(
                'Please call `full_init` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.dataset.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.dataset.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception('Cannot find valid image after '
                        f'{self.dataset.max_refetch}! '
                        'Please check your image path and pipeline')

    @force_full_init
    def __len__(self):
        return len(self.dataset)

    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))
