import os.path as osp
import warnings
from typing import Any, List, Union

from mmengine.dataset import BaseDataset

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
        return self._metainfo

    def full_init(self):
        self.dataset.full_init()

    def get_data_info(self, idx: int) -> dict:
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

    def __getitem__(self, idx):
        if not self.dataset._fully_initialized:
            warnings.warn(
                'Please call `full_init()` method manually to accelerate '
                'the speed.')
            self.dataset.full_init()

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
                idx = self.dataset._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')

    def __len__(self):
        return len(self.dataset)

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
