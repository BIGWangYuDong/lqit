import os.path as osp
import warnings

from mmengine.config import Config
from mmengine.runner import EpochBasedTrainLoop, IterBasedTrainLoop

from ..engine.hooks.lark_hook import LarkHook


def process_lark_hook(cfg: Config, lark_file: str) -> list:
    """Process LarkHook in custom_hooks.

    Here are three cases:
    1. If `custom_hooks` is None, add a LarkHook.
    2. If `custom_hooks` has LarkHook, update it.
    3. If `custom_hooks` does not have LarkHook, add a LarkHook.

    Args:
        cfg (:obj:`Config`): Full config.
        lark_file (str): Lark config file.

    Returns:
        list[dict]: Custom hooks with processed `LarkHook`.
    """
    custom_hooks = cfg.get('custom_hooks', None)

    process_lark = True
    if not osp.exists(lark_file):
        warnings.warn(f'{lark_file} not exists, skip process lark hook.')
        process_lark = False
    else:
        lark_url = Config.fromfile(lark_file).get('lark', None)
        if lark_url is None:
            warnings.warn(f'{lark_file} does not have `lark`, '
                          'skip process lark hook.')
            process_lark = False

    if not process_lark:
        return custom_hooks

    train_cfg = cfg['train_cfg']
    train_cfg_type = cfg['train_cfg']['type']

    if train_cfg_type == 'EpochBasedTrainLoop' or \
            isinstance(train_cfg_type, EpochBasedTrainLoop):
        by_epoch = True
        # max_epoch = train_cfg['max_epochs']
        val_interval = train_cfg['val_interval']

    elif train_cfg['type'] == 'IterBasedTrainLoop' or \
            isinstance(train_cfg_type, IterBasedTrainLoop):
        by_epoch = False
        # max_iters = train_cfg['max_iters']
        val_interval = train_cfg['val_interval']

    else:
        raise NotImplementedError

    base_lark_hook = dict(
        type='lqit.LarkHook',
        url=lark_url,
        cfg_file=cfg.filename,
        user_name=None,
        interval=val_interval,
        by_epoch=by_epoch,
        silent=True,
        first_eta_iter=200,
    )
    if custom_hooks is None:
        # does not set custom hook,  custom hood
        new_custom_hooks = [base_lark_hook]
    else:
        assert isinstance(custom_hooks, list)
        has_lark_hook = False
        new_custom_hooks = []
        for hook in custom_hooks:
            hook_type = hook['type']
            if hook_type == 'LarkHook' or hook_type == 'lqit.LarkHook' or \
                    isinstance(hook_type, LarkHook):
                has_lark_hook = True
                if by_epoch != hook['by_epoch']:
                    warnings.warn('LarkHook `by_epoch` is different from '
                                  'train_cfg, this may cause error!')
                base_lark_hook.update(hook)
                new_custom_hooks.append(base_lark_hook)
            else:
                new_custom_hooks.append(hook)

        if not has_lark_hook:
            new_custom_hooks.append(base_lark_hook)
    return new_custom_hooks
