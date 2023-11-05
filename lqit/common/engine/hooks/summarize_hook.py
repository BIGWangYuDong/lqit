import logging
import os.path as osp
from pathlib import Path
from typing import Dict, Optional, Union

import tabulate
from mmengine.dist import master_only
from mmengine.hooks import Hook
from mmengine.logging import print_log

from lqit.registry import HOOKS


@HOOKS.register_module()
class SummarizeHook(Hook):
    """Summarize Hook, saving the metrics into a csv file.

    Args:
        summary_file (str): The name of the summary file.
            Defaults to 'gather_results.csv'.
        out_dir (str): The output directory. If not specified, it will be set
            to the log directory of the runner. Defaults to None.
        by_epoch (bool): Whether to save the metrics by epoch or by iteration.
            Defaults to True.
    """
    priority = 'VERY_LOW'

    def __init__(self,
                 summary_file: str = 'gather_results.csv',
                 out_dir: Optional[Union[str, Path]] = None,
                 by_epoch: bool = True):
        if not summary_file.endswith('.csv'):
            summary_file += '.csv'

        if out_dir is not None and not isinstance(out_dir, (str, Path)):
            raise TypeError('out_dir must be a str or Path object')
        self.out_dir = out_dir

        if by_epoch:
            self.metric_tmpl = 'epoch_{}'
        else:
            self.metric_tmpl = 'iter_{}'

        self.summary_file = summary_file
        self.by_epoch = by_epoch
        self.header = None
        self.gather_results = dict()

    def before_run(self, runner) -> None:
        """Set the output directory to the log directory of the runner if
        `out_dir` is not specified."""
        if self.out_dir is None:
            self.out_dir = runner.log_dir

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        if self.by_epoch:
            name = self.metric_tmpl.format(runner.epoch)
        else:
            name = self.metric_tmpl.format(runner.iter)
        self.process_metrics(name, metrics)

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        # name set as the checkpoint name
        ckpt_path = runner._load_from
        name = osp.basename(ckpt_path).split('.')[0]
        self.process_metrics(name, metrics)

    def process_metrics(self, name, metrics: Dict[str, float]):
        if self.header is None:
            if len(metrics) > 0:
                self.header = [key for key in metrics.keys()]

        if len(metrics) > 0:
            row = [str(item) for item in metrics.values()]
        else:
            row = None

        if self.header is not None and row is not None:
            assert len(self.header) == len(row)

        self.gather_results[name] = row

    @master_only
    def summarize(self):
        csv_file = osp.join(self.out_dir, self.summary_file)
        txt_file = osp.join(self.out_dir,
                            self.summary_file.replace('.csv', '.txt'))
        table = []
        header = ['Architecture'] + self.header
        table.append(header)
        for key, row in self.gather_results.items():
            if row is None:
                row = ['-'] * len(header)
            table.append([key] + row)
        # output to screean
        print(tabulate.tabulate(table, headers='firstrow'))
        # output to txt file
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(tabulate.tabulate(table, headers='firstrow'))

        # output to csv file
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join([','.join(row) for row in table]) + '\n')

        print_log(
            f'Summary results have been saved to {csv_file}.',
            logger='current',
            level=logging.INFO)

    def after_run(self, runner) -> None:
        # save into a csv file
        if self.out_dir is None:
            print_log(
                '`SummarizeHook.out_dir` is not specified, cannot save '
                'the summary file.',
                logger='current',
                level=logging.WARNING)
        elif self.header is None:
            print_log(
                'No metrics have been gathered from the runner. '
                'Cannot save the summary file.',
                logger='current',
                level=logging.WARNING)
        else:
            self.summarize()
