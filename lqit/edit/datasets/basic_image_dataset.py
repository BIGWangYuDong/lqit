# Modified from https://github.com/open-mmlab/mmediting/tree/1.x/
import os.path as osp
from typing import Callable, List, Optional, Union

from mmengine.dataset import BaseDataset
from mmengine.fileio import FileClient, list_from_file

from lqit.registry import DATASETS

IMG_EXTENSIONS = ('jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'ppm', 'PPM',
                  'bmp', 'BMP', 'tif', 'TIF', 'tiff', 'TIFF')


@DATASETS.register_module()
class BasicImageDataset(BaseDataset):
    """BasicImageDataset for pixel-level vision tasks that have aligned gts,
    such as image enhancement.

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img='').
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
    """

    METAINFO = dict(dataset_type='basic_image_dataset', task_name='editing')

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 search_key: Optional[str] = None,
                 file_client_args: dict = dict(backend='disk'),
                 img_suffix: Union[str, dict] = 'jpg',
                 recursive: bool = False,
                 **kwards):

        # `search_key` is used in load image from folder. If there is no
        # special setting, it defaults to the first key in data_prefix.
        if search_key is None:
            keys = list(data_prefix.keys())
            search_key = keys[0]
        else:
            assert search_key in list(data_prefix.keys())
        self.search_key = search_key

        # if set ann_file, the data list will get from ann_file,
        # else get from folder.
        self.use_ann_file = (ann_file != '')
        self.file_client_args = file_client_args
        self.file_client = FileClient.infer_client(
            file_client_args=file_client_args, uri=data_root)

        self.img_suffix = dict()
        if isinstance(img_suffix, str):
            assert img_suffix in IMG_EXTENSIONS
            for key in data_prefix.keys():
                self.img_suffix[key] = img_suffix
        elif isinstance(img_suffix, dict):
            for key in data_prefix.keys():
                assert key in img_suffix, f'{key} not in img_suffix'
                assert img_suffix[key] in IMG_EXTENSIONS
            self.img_suffix = img_suffix
        else:
            raise TypeError('img_suffix should be a str or a dict, '
                            f'but got {type(img_suffix)}')
        self.recursive = recursive

        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
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
                path = osp.join(self.data_prefix[key],
                                f'{img_id}.{self.img_suffix[key]}')
                data[f'{key}_path'] = path
            data_list.append(data)
        return data_list

    def _get_img_list(self) -> list:
        """Get list of paths from annotation file or folder of dataset.

        Returns:
            list[dict]: A list of paths.
        """
        if self.use_ann_file:
            img_ids = self._get_img_list_from_ann()
        else:
            img_ids = self._get_img_list_from_folder()

        return img_ids

    def _get_img_list_from_ann(self) -> list:
        """Get list of images from annotation file.

        Returns:
            List: List of paths.
        """

        ann_ids = list_from_file(
            self.ann_file, file_client_args=self.file_client_args)
        img_ids = []
        for ann_id in ann_ids:
            # delete suffix to keep logic same
            img_id, suffix = osp.splitext(ann_id)
            if suffix not in IMG_EXTENSIONS:
                img_id = ann_id
            img_ids.append(img_id)

        return img_ids

    def _get_img_list_from_folder(self) -> list:
        """Get list of images from folder.

        Returns:
            List: List of paths.
        """

        img_ids = []
        folder = self.data_prefix[self.search_key]
        img_suffix = self.img_suffix[self.search_key]
        for img_path in self.file_client.list_dir_or_file(
                dir_path=folder,
                list_dir=False,
                suffix=img_suffix,
                recursive=self.recursive):
            # delete suffix to keep logic same
            img_id, _ = osp.splitext(img_path)
            img_ids.append(img_id)
        return img_ids
