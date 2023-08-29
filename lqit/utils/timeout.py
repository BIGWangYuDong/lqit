import signal


class TimeOutHandler:
    """Timeout handler.

    If the code gets stuck it will return TimeoutError.

    Args:
        waiting_time (int): waiting time. Defaults to 1.
        error_message (str): error message. Defaults to "TimeoutError".

    Examples:
    >>> with TimeOut(10):
    >>>     try:
    >>>         # do something
    >>>     except TimeoutError as e:
    >>>         print(e)
    """

    def __init__(self,
                 waiting_time: int = 1,
                 error_message: str = 'TimeoutError'):
        self.waiting_time = waiting_time
        self.error_message = error_message

    def timeout_handler(self, *args, **kwargs):
        raise TimeoutError(f'{self.error_message}, the code maybe stuck')

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.waiting_time)

    def __exit__(self, *args, **kwargs):
        signal.alarm(0)


TimeOut = TimeOutHandler(waiting_time=5, error_message='TimeoutError')
