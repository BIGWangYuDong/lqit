# Modified from https://github.com/open-mmlab/mmediting/blob/main/mmedit/utils/logger.py  # noqa: E501
# Modified from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/utils/logger.py  # noqa: E501
import inspect
import logging

from mmengine.logging import print_log
from termcolor import colored


def print_colored_log(msg: str,
                      level: int = logging.INFO,
                      color: str = 'magenta') -> None:
    """Print colored log with default logger.

    Args:
        msg (str): Message to log.
        level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.Log level,
            default to 'info'.
        color (str, optional): Color 'magenta'.
    """
    print_log(colored(msg, color), 'current', level)


def get_caller_name() -> str:
    """Get name of caller method."""
    # this_func_frame = inspect.stack()[0][0]  # i.e., get_caller_name
    # callee_frame = inspect.stack()[1][0]  # e.g., log_img_scale
    caller_frame = inspect.stack()[2][0]  # e.g., caller of log_img_scale
    caller_method = caller_frame.f_code.co_name
    try:
        caller_class = caller_frame.f_locals['self'].__class__.__name__
        return f'{caller_class}.{caller_method}'
    except KeyError:  # caller is a function
        return caller_method


def log_img_scale(img_scale: type,
                  shape_order: str = 'hw',
                  skip_square: bool = False) -> bool:
    """Log image size.

    Args:
        img_scale (tuple): Image size to be logged.
        shape_order (str, optional): The order of image shape.
            'hw' for (height, width) and 'wh' for (width, height).
            Defaults to 'hw'.
        skip_square (bool, optional): Whether to skip logging for square
            img_scale. Defaults to False.

    Returns:
        bool: Whether to have done logging.
    """
    if shape_order == 'hw':
        height, width = img_scale
    elif shape_order == 'wh':
        width, height = img_scale
    else:
        raise ValueError(f'Invalid shape_order {shape_order}.')

    if skip_square and (height == width):
        return False

    caller = get_caller_name()
    print_log(
        f'image shape: height={height}, width={width} in {caller}',
        logger='current')

    return True
