# Modified from https://github.com/InternLM/opencompass/
# Modified from https://github.com/InternLM/InternLM/
import datetime
import os
import time
from typing import Dict, Optional

from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH

from lqit.common.utils.lark_manager import get_user_name, send_alert_message
from lqit.registry import HOOKS


def set_env_var(key, value):
    os.environ[str(key)] = str(value)


@HOOKS.register_module()
class LarkHook(Hook):
    """Hook that sends message to Lark.

    Args:
        url (str): The url of Lark webhook.
        interval (int): The interval of sending message. Default: 1.
    """

    priority = 'BELOW_NORMAL'

    def __init__(
        self,
        url: str,
        cfg_file: str,
        user_name: Optional[str] = None,
        interval: int = 1,
        by_epoch: bool = True,
        silent: bool = True,
        first_eta_iter: int = 200,
    ):
        self.url = url
        self.interval = interval
        self.by_epoch = by_epoch
        self.cfg_file = cfg_file
        if user_name is None:
            user_name = get_user_name()
            if user_name is None:
                user_name = 'lqit'
        self.user_name = user_name
        self.silent = silent
        # sent eta message after `first_eta_iter` iterations
        self.first_eta_log = True
        self.first_eta_iter = first_eta_iter
        self.metrics_str = None

    @staticmethod
    def get_eta_time(runner) -> str:
        eta = runner.message_hub.get_info('eta')
        if eta is None:
            return None
        else:
            eta_str = str(datetime.timedelta(seconds=int(eta)))
            return eta_str

    def get_metric_results(self, metrics) -> str:
        if len(metrics) == 0:
            metrics_str = 'Empty metrics'
        else:
            metrics_str = 'Results:\n'
            for key, value in metrics.items():
                metrics_str += f'{key}: {value}\n'
        self.metrics_str = metrics_str
        return metrics_str

    def get_train_msg(self, runner) -> str:
        if self.by_epoch:
            progress = f'Finished {runner.epoch + 1} / ' \
                       f'{runner.max_epochs} epochs'
        else:
            progress = f'Finished {runner.iter + 1} / ' \
                       f'{runner.max_iters} iterations'
        eta_str = self.get_eta_time(runner=runner)

        title = 'Task Progress Report'
        msg = f"{self.user_name}'s task\n" \
              f'Config file: {self.cfg_file}\n' \
              f'Training progress: {progress}\n'
        if eta_str is not None:
            msg += f'Estimated time of completion: {eta_str}\n'
        return msg, title

    def get_val_msg(self, runner, metrics_str) -> str:
        if self.by_epoch:
            # epoch based runner will add 1 before call validation
            progress = f'Finished {runner.epoch} / ' \
                       f'{runner.max_epochs} epochs'
        else:
            progress = f'Finished {runner.iter + 1} / ' \
                       f'{runner.max_iters} iterations'

        title = 'Task Progress Report'
        msg = f"{self.user_name}'s task\n" \
              f'Config file: {self.cfg_file}\n' \
              f'Training progress: {progress}\n'
        msg += metrics_str
        return msg, title

    def get_test_msg(self, runner, metrics_str) -> str:

        title = 'Task Progress Report'
        msg = f"{self.user_name}'s task\n" \
              f'Config file: {self.cfg_file}\n'

        msg += metrics_str
        return msg, title

    def get_first_eta_msg(self, runner) -> str:
        eta_str = self.get_eta_time(runner=runner)
        if eta_str is None:
            return None, None
        else:
            title = 'Task Progress Report'
            msg = f"{self.user_name}'s Training task\n" \
                  f'Config file: {self.cfg_file}\n' \
                  f'Estimated time of completion: {eta_str}\n'
            return msg, title

    def before_train(self, runner) -> None:
        if self.silent:
            return
        title = 'Task Initiation Report'
        content = f"{self.user_name}'s task has started training!\n" \
                  f'Config file: {self.cfg_file}\n' \
                  f'Output path: {runner.work_dir}' \
                  f'Total epoch: {runner.max_epochs}\n' \
                  f'Total iter: {runner.max_iters}'

        send_alert_message(url=self.url, content=content, title=title)

    def before_test(self, runner) -> None:
        if self.silent:
            return
        # TODO: Check
        title = 'Task Initiation Report'
        content = f"{self.user_name}'s task has started testing!\n" \
                  f'Config file: {self.cfg_file}\n' \
                  f'Output path: {runner.work_dir}'

        send_alert_message(url=self.url, content=content, title=title)

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return
        if self.silent:
            return
        if self.every_n_epochs(runner, self.interval):
            msg, title = self.get_train_msg(runner)
            if msg is not None:
                send_alert_message(url=self.url, content=msg, title=title)

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        metrics_str = self.get_metric_results(metrics)
        if self.silent:
            return

        msg, title = self.get_val_msg(runner, metrics_str)
        if msg is not None:
            send_alert_message(url=self.url, content=msg, title=title)

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        metrics_str = self.get_metric_results(metrics)
        if self.silent:
            return
        msg, title = self.get_test_msg(runner, metrics_str)
        if msg is not None:
            send_alert_message(url=self.url, content=msg, title=title)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        # set LAST_ACTIVE_TIMESTAMP in the environ, so that the monitor
        # manager can check if the process is stuck
        set_env_var(key='LAST_ACTIVE_TIMESTAMP', value=int(time.time()))

        if self.first_eta_log:
            if self.every_n_train_iters(runner, self.first_eta_iter):
                msg, title = self.get_first_eta_msg(runner)
                self.first_eta_log = False
                if msg is not None:
                    send_alert_message(url=self.url, content=msg, title=title)
        if self.by_epoch:
            return
        if not self.silent:
            return
        if not self.every_n_iters(runner, self.interval):
            msg, title = self.get_train_msg(runner)
            if msg is not None:
                send_alert_message(url=self.url, content=msg, title=title)

    def after_val_iter(self, *args, **kwargs) -> None:
        # set LAST_ACTIVE_TIMESTAMP in the environ, so that the monitor
        # manager can check if the process is stuck
        set_env_var(key='LAST_ACTIVE_TIMESTAMP', value=int(time.time()))

    def after_test_iter(self, *args, **kwargs) -> None:
        # set LAST_ACTIVE_TIMESTAMP in the environ, so that the monitor
        # manager can check if the process is stuck
        set_env_var(key='LAST_ACTIVE_TIMESTAMP', value=int(time.time()))

    def after_run(self, runner) -> None:
        if self.metrics_str is not None:
            set_env_var(key='LAST_METRIC_RESULTS', value=self.metrics_str)
