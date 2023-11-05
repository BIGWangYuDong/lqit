import argparse
import logging
import os
import os.path as osp
import warnings

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from lqit.common.utils.lark_manager import (context_monitor_manager,
                                            get_error_message,
                                            initialize_monitor_manager)
from lqit.common.utils.process_lark_hook import process_lark_hook
from lqit.utils import (print_colored_log, process_debug_mode,
                        setup_cache_size_limit_of_dynamo)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '-l',
        '--lark',
        help='Report the running status to lark bot',
        action='store_true',
        default=False)
    parser.add_argument(
        '--lark-file',
        default='configs/lark/lark.py',
        type=str,
        help='lark bot config file path')
    parser.add_argument(
        '--debug',
        default=False,
        action='store_true',
        help='Debug mode, used for code debugging, specifically, turning '
        'data processing into single process (`num_workers`), adding '
        '`indices=10` in datasets, and other debug-friendly settings.')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main(args):
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # process debug mode if args.debug is True
    if args.debug:
        # force set args.lark = False
        args.lark = False
        # set necessary params for debug mode
        cfg = process_debug_mode(cfg)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    if args.lark:
        custom_hooks = process_lark_hook(cfg=cfg, lark_file=args.lark_file)
        cfg.custom_hooks = custom_hooks

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    print_colored_log(f'Working directory: {cfg.work_dir}')
    print_colored_log(f'Log directiry: {runner._log_dir}')

    # start training
    runner.train()

    print_colored_log(f'Log saved under {runner._log_dir}')
    print_colored_log(f'Checkpoint saved under {cfg.work_dir}')


if __name__ == '__main__':
    args = parse_args()

    monitor_manager = None

    if args.lark:
        # report the running status to lark bot
        lark_file = args.lark_file
        if not osp.exists(lark_file):
            warnings.warn(f'{lark_file} not exists, skip.')
            lark_url = None
        else:
            lark = Config.fromfile(lark_file)
            lark_url = lark.get('lark', None)
            if lark_url is None:
                warnings.warn(f'{lark_file} does not have `lark`, skip.')
            else:
                monitor_interval_seconds = lark.get('monitor_interval_seconds',
                                                    300)
                user_name = lark.get('user_name', None)
                monitor_manager = initialize_monitor_manager(
                    cfg_file=args.config,
                    url=lark_url,
                    task_type='train',
                    user_name=user_name,
                    monitor_interval_seconds=monitor_interval_seconds)

    with context_monitor_manager(monitor_manager):
        try:
            main(args)
        except Exception:
            if monitor_manager is not None:
                monitor_manager.monitor_exception()
            else:
                get_error_message()
