"""LQIT provides 17 registry nodes to support using modules across projects.
Each node is a child of the root registry in MMEngine.

If you want to use a codebase in OpenMMLab 2.0 series, it is suggested to
rigister your modules in the corresponding codebase registry node.
For example, if you want to use MMDetection and define a detection model,
you can register your model in MMDetection registry node. Such as:

>>> from mmdet.registry import MODELS
>>> @MODELS.register_module()
>>> class MaskRCNN:
>>>     pass

More details can be found at
https://mmengine.readthedocs.io/en/latest/tutorials/registry.html.
"""

from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import \
    RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import \
    WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry(
    'runner',
    parent=MMENGINE_RUNNERS,
    locations=[
        'lqit.common.engine', 'lqit.detection.engine', 'lqit.edit.engine'
    ])
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor',
    parent=MMENGINE_RUNNER_CONSTRUCTORS,
    locations=[
        'lqit.common.engine', 'lqit.detection.engine', 'lqit.edit.engine'
    ])
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry(
    'loop',
    parent=MMENGINE_LOOPS,
    locations=[
        'lqit.common.engine', 'lqit.detection.engine', 'lqit.edit.engine'
    ])
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook',
    parent=MMENGINE_HOOKS,
    locations=[
        'lqit.common.engine', 'lqit.detection.engine', 'lqit.edit.engine'
    ])

# manage data-related modules
DATASETS = Registry(
    'dataset',
    parent=MMENGINE_DATASETS,
    locations=[
        'lqit.common.datasets', 'lqit.detection.datasets', 'lqit.edit.datasets'
    ])
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=MMENGINE_DATA_SAMPLERS,
    locations=[
        'lqit.common.datasets', 'lqit.detection.datasets', 'lqit.edit.datasets'
    ])
TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=[
        'lqit.common.datasets.transforms',
        'lqit.detection.datasets.transforms', 'lqit.edit.datasets.transforms'
    ])

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry(
    'model',
    parent=MMENGINE_MODELS,
    locations=[
        'lqit.common.models', 'lqit.detection.models', 'lqit.edit.models'
    ])
# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMENGINE_MODEL_WRAPPERS,
    locations=[
        'lqit.common.models', 'lqit.detection.models', 'lqit.edit.models'
    ])
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMENGINE_WEIGHT_INITIALIZERS,
    locations=[
        'lqit.common.models', 'lqit.detection.models', 'lqit.edit.models'
    ])

# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMENGINE_OPTIMIZERS,
    locations=[
        'lqit.common.engine', 'lqit.detection.engine', 'lqit.edit.engine'
    ])
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optim_wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS,
    locations=[
        'lqit.common.engine', 'lqit.detection.engine', 'lqit.edit.engine'
    ])
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=[
        'lqit.common.engine', 'lqit.detection.engine', 'lqit.edit.engine'
    ])
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMENGINE_PARAM_SCHEDULERS,
    locations=[
        'lqit.common.engine', 'lqit.detection.engine', 'lqit.edit.engine'
    ])
# manage all kinds of metrics
METRICS = Registry(
    'metric',
    parent=MMENGINE_METRICS,
    locations=[
        'lqit.common.evaluation', 'lqit.detection.evaluation',
        'lqit.edit.evaluation'
    ])
# manage evaluator
EVALUATOR = Registry(
    'evaluator',
    parent=MMENGINE_EVALUATOR,
    locations=[
        'lqit.common.evaluation', 'lqit.detection.evaluation',
        'lqit.edit.evaluation'
    ])

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util',
    parent=MMENGINE_TASK_UTILS,
    locations=[
        'lqit.common.models', 'lqit.detection.models', 'lqit.edit.models'
    ])

# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    parent=MMENGINE_VISUALIZERS,
    locations=[
        'lqit.common.visualization', 'lqit.detection.visualization',
        'lqit.edit.visualization'
    ])
# manage visualizer backend
VISBACKENDS = Registry(
    'vis_backend',
    parent=MMENGINE_VISBACKENDS,
    locations=[
        'lqit.common.visualization', 'lqit.detection.visualization',
        'lqit.edit.visualization'
    ])

# manage logprocessor
LOG_PROCESSORS = Registry(
    'log_processor',
    parent=MMENGINE_LOG_PROCESSORS,
    locations=[
        'lqit.common.engine', 'lqit.detection.engine', 'lqit.edit.engine'
    ])
