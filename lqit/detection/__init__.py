import warnings

from mmengine.utils import digit_version

try:
    import mmdet
    HAS_MMDET = True
except ImportWarning:
    HAS_MMDET = False

mmdet_minimum_version = '3.0.0'
mmdet_maximum_version = '3.1.0'
if HAS_MMDET:
    mmdet_version = digit_version(mmdet.__version__)
    assert (mmdet_version >= digit_version(mmdet_minimum_version)
            and mmdet_version < digit_version(mmdet_maximum_version)), \
        f'MMDetection=={mmdet.__version__} is used but incompatible. ' \
        f'Please install mmdet>={mmdet_minimum_version}, ' \
        f'<{mmdet_maximum_version}.'
    from .datasets import *  # noqa: F401,F403
    from .engine import *  # noqa: F401,F403
    from .models import *  # noqa: F401,F403
else:
    warnings.warn('Please install mmdet to import `lqit.detection`.')
