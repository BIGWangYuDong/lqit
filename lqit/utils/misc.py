# Modified from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/utils/misc.py  # noqa: E501
import glob
import os
import os.path as osp
import urllib
import warnings
from typing import Optional, Union

import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import scandir

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def find_latest_checkpoint(path: str, suffix: str = 'pth') -> Optional[str]:
    """Find the latest checkpoint from the working directory.

    Args:
        path(str): The path to find checkpoints.
        suffix(str): File extension.
            Defaults to pth.

    Returns:
        latest_path(str | None): File path of the latest checkpoint.

    References:
        .. [1] https://github.com/microsoft/SoftTeacher
                  /blob/main/ssod/utils/patch.py
    """
    if not osp.exists(path):
        warnings.warn('The path of checkpoints does not exist.')
        return None
    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        warnings.warn('There are no checkpoints in the path.')
        return None
    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint
    return latest_path


def update_data_root(cfg: Config) -> None:
    """Update data root according to env MMDET_DATASETS.

    If set env LQIT_DATASETS, update cfg.data_root according to
    LQIT_DATASETS. Otherwise, using cfg.data_root as default.

    Args:
        cfg (:obj:`Config`): The model config need to modify
    """
    assert isinstance(cfg, Config), \
        f'cfg got wrong type: {type(cfg)}, expected mmengine.Config'

    if 'LQIT_DATASETS' in os.environ:
        dst_root = os.environ['LQIT_DATASETS']
        print_log(f'LQIT_DATASETS has been set to be {dst_root}.'
                  f'Using {dst_root} as data root.')
    else:
        return

    assert isinstance(cfg, Config), \
        f'cfg got wrong type: {type(cfg)}, expected mmengine.Config'

    def update(cfg, src_str, dst_str):
        for k, v in cfg.items():
            if isinstance(v, ConfigDict):
                update(cfg[k], src_str, dst_str)
            if isinstance(v, str) and src_str in v:
                cfg[k] = v.replace(src_str, dst_str)

    update(cfg.data, cfg.data_root, dst_root)
    cfg.data_root = dst_root


def get_test_pipeline_cfg(cfg: Union[str, ConfigDict]) -> ConfigDict:
    """Get the test dataset pipeline from entire config.

    Args:
        cfg (str or :obj:`ConfigDict`): the entire config. Can be a config
            file or a ``ConfigDict``.

    Returns:
        :obj:`ConfigDict`: the config of test dataset.
    """
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)

    def _get_test_pipeline_cfg(dataset_cfg):
        if 'pipeline' in dataset_cfg:
            return dataset_cfg.pipeline
        # handle dataset wrapper
        elif 'dataset' in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.dataset)
        # handle dataset wrappers like ConcatDataset
        elif 'datasets' in dataset_cfg:
            return _get_test_pipeline_cfg(dataset_cfg.datasets[0])

        raise RuntimeError('Cannot find `pipeline` in `test_dataloader`')

    return _get_test_pipeline_cfg(cfg.test_dataloader.dataset)


def auto_arrange_images(image_list: list, image_column: int = 2) -> np.ndarray:
    """Auto arrange image to image_column x N row.

    Args:
        image_list (list): cv2 image list.
        image_column (int): Arrange to N column. Default: 2.

    Return:
        (np.ndarray): image_column x N row merge image
    """
    img_count = len(image_list)
    if img_count <= image_column:
        # no need to arrange
        image_show = np.concatenate(image_list, axis=1)
    else:
        # arrange image according to image_column
        image_row = round(img_count / image_column)
        fill_img_list = [np.ones(image_list[0].shape, dtype=np.uint8) * 255
                         ] * (
                             image_row * image_column - img_count)
        image_list.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(image_row):
            start_col = image_column * i
            end_col = image_column * (i + 1)
            merge_col = np.hstack(image_list[start_col:end_col])
            merge_imgs_col.append(merge_col)

        # merge to one image
        image_show = np.vstack(merge_imgs_col)

    return image_show


def get_file_list(source_root: str) -> [list, dict]:
    """Get file list.

    Args:
        source_root (str): image or video source path

    Return:
        source_file_path_list (list): A list for all source file.
        source_type (dict): Source type: file or url or dir.
    """
    is_dir = os.path.isdir(source_root)
    is_url = source_root.startswith(('http:/', 'https:/'))
    is_file = os.path.splitext(source_root)[-1].lower() in IMG_EXTENSIONS

    source_file_path_list = []
    if is_dir:
        # when input source is dir
        for file in scandir(source_root, IMG_EXTENSIONS, recursive=True):
            source_file_path_list.append(os.path.join(source_root, file))
    elif is_url:
        # when input source is url
        filename = os.path.basename(
            urllib.parse.unquote(source_root).split('?')[0])
        file_save_path = os.path.join(os.getcwd(), filename)
        print(f'Downloading source file to {file_save_path}')
        torch.hub.download_url_to_file(source_root, file_save_path)
        source_file_path_list = [file_save_path]
    elif is_file:
        # when input source is single image
        source_file_path_list = [source_root]
    else:
        print('Cannot find image file.')

    source_type = dict(is_dir=is_dir, is_url=is_url, is_file=is_file)

    return source_file_path_list, source_type
