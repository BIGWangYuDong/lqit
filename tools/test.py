import argparse
import os
import os.path as osp
import warnings

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from lqit.common.utils.lark_manager import (context_monitor_manager,
                                            initialize_monitor_manager)
from lqit.common.utils.process_lark_hook import process_lark_hook
from lqit.registry import RUNNERS
from lqit.utils import setup_cache_size_limit_of_dynamo


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='LQIT test (and eval) a detection model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
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
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def main(args):
    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

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

    # start testing
    runner.test()


if __name__ == '__main__':
    args = parse_args()

    monitor_manager = None

    if args.lark:
        lark_file = args.lark_file
        if not osp.exists(lark_file):
            warnings.warn(f'{lark_file} not exists, skip.')
            lark_url = None
        else:
            lark = Config.fromfile(lark_file)
            lark_url = lark.get('lark', None)
            if lark_url is None:
                warnings.warn(f'{lark_file} does not have `lark`, skip.')

            monitor_interval_seconds = lark.get('monitor_interval_seconds',
                                                None)
            if monitor_interval_seconds is None:
                monitor_interval_seconds = 300

            user_name = lark.get('user_name', None)

        monitor_manager = initialize_monitor_manager(
            cfg_file=args.config,
            url=lark_url,
            task_type='test',
            user_name=user_name,
            monitor_interval_seconds=monitor_interval_seconds,
            ckpt_path=args.checkpoint)
    with context_monitor_manager(monitor_manager):
        try:
            main(args)
        except Exception:
            monitor_manager.monitor_exception()
