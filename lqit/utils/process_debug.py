import warnings
from typing import Union

from mmengine.config import ConfigDict
from mmengine.dataset import (ClassBalancedDataset, ConcatDataset,
                              DefaultSampler, InfiniteSampler, RepeatDataset)

ConfigType = Union[dict, ConfigDict]


def process_debug_mode(cfg: ConfigType) -> ConfigType:
    """Process config for debug mode.

    Args:
        cfg (dict or :obj:`ConfigDict`): Config dict.

    Returns:
        dict or :obj:`ConfigDict`: Config dict.
    """

    dataloader_list = ['train_dataloader', 'val_dataloader', 'test_dataloader']
    for dataloader_name in dataloader_list:
        dataset_type = cfg[dataloader_name]['dataset']['type']
        if dataset_type in \
                ['ConcatDataset', 'RepeatDataset', 'ClassBalancedDataset',
                 ConcatDataset, RepeatDataset, ClassBalancedDataset]:
            warnings.warn(f'{dataset_type} not support in debug mode, skip.')
        else:
            # set dataset.indices = 10
            cfg[dataloader_name]['dataset']['indices'] = 10

        # set num_workers = 0
        cfg[dataloader_name]['num_workers'] = 0
        cfg[dataloader_name]['persistent_workers'] = False

        # set shuffle = False
        if cfg[dataloader_name]['sampler']['type'] in \
                ['DefaultSampler', 'InfiniteSampler',
                 DefaultSampler, InfiniteSampler]:
            cfg[dataloader_name]['sampler']['shuffle'] = False
    # set seed = 0
    cfg['randomness']['seed'] = 0
    # set deterministic = True
    cfg['randomness']['deterministic'] = True

    # set log_level = 'DEBUG'
    cfg['log_level'] = 'DEBUG'

    # set max_keep_ckpts = 1
    cfg['default_hooks']['checkpoint']['max_keep_ckpts'] = 1

    return cfg
