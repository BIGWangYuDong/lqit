# Modified from https://github.com/InternLM/opencompass/
# Modified from https://github.com/InternLM/InternLM/
import json
import os
import signal
import time
import traceback
from contextlib import contextmanager
from threading import Thread
from typing import Dict, List, Optional, Union

import requests
from func_timeout import FunctionTimedOut, func_set_timeout
from mmengine.dist import master_only


def get_user_name():
    for name in ('LOGNAME', 'USER', 'LNAME', 'USERNAME'):
        user = os.environ.get(name)
        if user:
            return user
    return None


def get_rank():
    rank = os.getenv('RANK')
    if rank is None:
        rank = os.getenv('SLURM_PROCID')
    return rank


@master_only
def send_alert_message(url: str,
                       content: Union[str, List[List[Dict]]],
                       title: Optional[str] = None):
    """Post a message to Lark.

    When title is None, message must be a str.
    otherwise msg can be in rich text format (see
    https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/im-v1/message/create_json#45e0953e
    for details).
    """
    if title is None:
        assert isinstance(content, str)
        msg = {'msg_type': 'text', 'content': {'text': content}}
    else:
        if isinstance(content, str):
            content = [[{'tag': 'text', 'text': content}]]
        msg = {
            'msg_type': 'post',
            'content': {
                'post': {
                    'zh_cn': {
                        'title': title,
                        'content': content
                    }
                }
            }
        }
    try:
        # avoid connection timeout
        func_set_timeout(5)(requests.post(url, data=json.dumps(msg)))
    except FunctionTimedOut as e:
        print(e)


class MonitorTracker(Thread):
    """Track job status and alert to Feishu during job training.

    Args:
        user_name (str): The user name of the job.
        cfg_file (str): The config file of the job.
        url (str): The Feishu webhook address for sending alerting messages.
        task_type (str): The type of the task, 'train' or 'test'.
            Defaults to 'train'.
        check_interval (int): The interval in seconds for monitoring checks.
            Defaults to 300.
    """

    def __init__(self,
                 user_name: str,
                 cfg_file: str,
                 url: str,
                 task_type: str = 'train',
                 check_interval: int = 300):
        super().__init__()
        self.user_name = user_name
        self.cfg_file = cfg_file
        self.url = url
        assert isinstance(check_interval, int) and check_interval > 0
        self.check_interval = check_interval

        assert task_type in ['train', 'test']
        if task_type == 'train':
            self.task_type = 'Training'
        elif task_type == 'test':
            self.task_type = 'Testing'
        else:
            raise NotImplementedError

        self.last_active_time = -1
        self.last_loss_value = -1
        self.stopped = False
        self.start()

    def run(self):
        """start the monitor tracker."""

        while not self.stopped:
            try:
                self._check_stuck()
            except Exception:
                continue
            # time.sleep(self.check_interval)
            for _ in range(self.check_interval):
                time.sleep(1)
                if self.stopped:
                    break

    def _check_stuck(self):
        """Check training status for potential stuck condition."""

        new_active_time = -1
        # LAST_ACTIVE_TIMESTAMP will be added in `LarkHook.after_XXX_iter`
        # using
        # `set_env_var(key="LAST_ACTIVE_TIMESTAMP", value=int(time.time()))`
        # to set LAST_ACTIVE_TIMESTAMP
        if os.getenv('LAST_ACTIVE_TIMESTAMP') is not None:
            new_active_time = os.getenv('LAST_ACTIVE_TIMESTAMP')
        if int(new_active_time) <= int(self.last_active_time) and \
                new_active_time != -1:
            title = 'Task Progress Report'
            content = f"{self.user_name}'s {self.task_type} task\n" \
                      f'Config file: {self.cfg_file}\n' \
                      f'Task may be in stuck status, please check it.'

            # the process is not main, cannot directly use `send_alert_message`
            # send_alert_message(
            #     url=self.url,
            #     content=content,
            #     title=title)
            msg = {
                'msg_type': 'post',
                'content': {
                    'post': {
                        'zh_cn': {
                            'title': title,
                            'content': [[{
                                'tag': 'text',
                                'text': content
                            }]]
                        }
                    }
                }
            }

            try:
                # avoid connection timeout
                func_set_timeout(5)(
                    requests.post(self.url, data=json.dumps(msg)))
            except FunctionTimedOut as e:
                print(e)
        self.last_active_time = new_active_time

    def stop(self):
        """Stop the monitor tracker."""

        self.stopped = True


class SingletonMeta(type):
    """Singleton Meta."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        else:
            assert (
                len(args) == 0 and len(kwargs) == 0
            ), f'{cls.__name__} is a singleton class and ' \
               'a instance has been created.'
        return cls._instances[cls]


class MonitorManager(metaclass=SingletonMeta):
    """Monitor Manager for managing monitor thread and monitoring training
    status."""

    def __init__(self) -> None:
        self.monitor_thread = None
        self.user_name = None
        self.cfg_file = None
        self.task_type = None
        self.url = None

    def monitor_exception(self) -> None:
        """Catch and format exception information, send alert message to
        Feishu."""

        assert self.url is not None, \
            'Please run `MonitorManager.start_monitor` first.'

        filtered_trace = traceback.format_exc().split('\n')[-10:]
        format_trace = ''
        for line in filtered_trace:
            format_trace += '\n' + line
        title = 'Task Error Report'
        content = f"{self.user_name}'s {self.task_type} task\n" \
                  f'Config file: {self.cfg_file}\n' \
                  f'Task got exception: {format_trace}.\n' \
                  'Please check it.'
        send_alert_message(url=self.url, content=content, title=title)

    def handle_sigterm(self):
        """Catch SIGTERM signal, and send alert message to Feishu."""
        assert self.url is not None, \
            'Please run `MonitorManager.start_monitor` first.'

        def sigterm_handler(sys_signal, frame):
            print('receive frame: ', frame)
            print('receive signal: ', sys_signal)
            title = 'Task Report'
            content = f"{self.user_name}'s {self.task_type} task\n" \
                      f'Config file: {self.cfg_file}\n' \
                      f'Process received signal {signal} and exited.'
            send_alert_message(url=self.url, content=content, title=title)

        signal.signal(signal.SIGTERM, sigterm_handler)

    def start_monitor(self,
                      user_name: str,
                      cfg_file: str,
                      url: str,
                      task_type: str,
                      monitor_interval_seconds: int = 300,
                      ckpt_path: Optional[str] = None) -> None:
        """Initialize and start monitor thread for checking training job status
        and other task.

        Args:
            user_name (str): The user name of the job.
            cfg_file (str): The config file of the job.
            url (str): The Feishu webhook address for sending alert messages.
            task_type (str): The type of the task, 'train' or 'test'.
            monitor_interval_seconds (int): The time of monitor interval
                in seconds. Defaults to 300.
        """
        # start a monitor thread, periodically check the training status
        self.monitor_thread = MonitorTracker(
            user_name=user_name,
            cfg_file=cfg_file,
            url=url,
            task_type=task_type,
            check_interval=monitor_interval_seconds,
        )
        # start a monitor thread, set the important information of the task
        if task_type == 'train':
            self.task_type = 'Training'
        elif task_type == 'test':
            self.task_type = 'Testing'
        else:
            raise NotImplementedError
        self.user_name = user_name
        self.cfg_file = cfg_file
        self.url = url
        title = 'Task Initiation Report'
        content = f"{self.user_name}'s {self.task_type} task has started!\n" \
                  f'Config file: {self.cfg_file}\n'
        if ckpt_path is not None:
            content += f'Checkpoint file: {ckpt_path}'
        rank = get_rank()
        if rank == '0' or rank == 0 or rank is None:
            send_alert_message(url=url, content=content, title=title)

    def stop_monitor(self) -> None:
        """Stop the monitor and alert thread."""
        assert self.url is not None, \
            'Please run `MonitorManager.start_monitor` first.'

        if self.monitor_thread is not None:
            self.monitor_thread.stop()
        title = 'Task Finish Report'

        content = f"{self.user_name}'s {self.task_type} task completed!\n" \
                  f'Config file: {self.cfg_file}\n'
        if os.getenv('LAST_METRIC_RESULTS') is not None:
            metric_content = os.getenv('LAST_METRIC_RESULTS')
            content += metric_content

        rank = get_rank()
        if rank == '0' or rank == 0 or rank is None:
            send_alert_message(url=self.url, content=content, title=title)


def initialize_monitor_manager(cfg_file: str,
                               url: str,
                               task_type: str,
                               user_name: Optional[str] = None,
                               monitor_interval_seconds: int = 300,
                               ckpt_path: Optional[str] = None) -> None:
    """Initialize and start monitor thread for checking training job status and
    other task.

    Args:
        user_name (str): The user name of the job.
        cfg_file (str): The config file of the job.
        url (str): The Feishu webhook address for sending alert messages.
        task_type (str): The type of the task, 'train' or 'test'.
        monitor_interval_seconds (int): The time of monitor interval
            in seconds. Defaults to 300.
    """
    if user_name is None:
        user_name = get_user_name()
        if user_name is None:
            user_name = 'lqit'
    monitor_manager = MonitorManager()
    monitor_manager.start_monitor(
        user_name=user_name,
        cfg_file=cfg_file,
        url=url,
        task_type=task_type,
        monitor_interval_seconds=monitor_interval_seconds,
        ckpt_path=ckpt_path)
    return monitor_manager


@contextmanager
def context_monitor_manager(monitor_manager: Optional[MonitorManager] = None):
    # `monitor_manager.start_monitor` should be called outside of the context
    if monitor_manager is not None and monitor_manager.url is not None:
        try:
            # start monitor should be called outside of the context
            # monitor_manager.start_monitor(job_name=job_name, url=url)
            monitor_manager.handle_sigterm()
            yield
        finally:
            monitor_manager.stop_monitor()
    else:
        yield
