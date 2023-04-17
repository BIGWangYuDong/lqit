# Modified from https://github.com/open-mmlab/mmediting/tree/1.x/
import os.path as osp
from typing import Callable, List, Optional, Union

from lqit.registry import DATASETS
from .basic_image_dataset import BasicImageDataset


@DATASETS.register_module()
class CityscapeFoggyImageDataset(BasicImageDataset):
    """CityscapeFoggyImageDataset for pixel-level vision tasks that have
    aligned gts.

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for data. Defaults to
            dict(img='').
        mapping_table (dict): Mapping table for data.
            Defaults to dict().
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        search_key (str): The key used for searching the folder to get
            data_list. Defaults to 'gt'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to dict(backend='disk').
        img_suffix (str or dict[str]): Image suffix that we are interested in.
            Defaults to jpg.
        recursive (bool): If set to True, recursively scan the
            directory. Defaults to False.
        split_str (str): split image name to gt image name.
            Defaults to '_foggy'.
    """

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 mapping_table: dict = dict(),
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 search_key: Optional[str] = None,
                 file_client_args: dict = dict(backend='disk'),
                 img_suffix: Union[str, dict] = 'jpg',
                 recursive: bool = False,
                 split_str: str = '_foggy',
                 **kwards) -> None:

        self.split_str = split_str

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            mapping_table=mapping_table,
            pipeline=pipeline,
            test_mode=test_mode,
            search_key=search_key,
            file_client_args=file_client_args,
            img_suffix=img_suffix,
            recursive=recursive,
            **kwards)

    def load_data_list(self) -> List[dict]:
        """Load data list from folder or annotation file.

        Returns:
            list[dict]: A list of annotation.
        """
        img_ids = self._get_img_list()

        data_list = []
        # deal with img and gt img path
        for img_id in img_ids:
            data = dict(key=img_id)
            data['img_id'] = img_id
            for key in self.data_prefix:
                img_id = self.mapping_table[key].format(img_id)
                # The gt img name and img name do not match.
                # one gt img corresponds to three imgs
                if key == 'gt_img':
                    img_id = img_id.split(self.split_str)[0]

                path = osp.join(self.data_prefix[key],
                                f'{img_id}.{self.img_suffix[key]}')
                data[f'{key}_path'] = path
            data_list.append(data)
        return data_list
