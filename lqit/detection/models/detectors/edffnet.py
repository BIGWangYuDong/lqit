from typing import Optional

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from lqit.registry import MODELS
from .single_stage_enhance_head import SingleStageWithEnhanceHead


@MODELS.register_module()
class EDFFNet(SingleStageWithEnhanceHead):

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 enhance_head: OptConfigType = None,
                 vis_enhance: Optional[bool] = False,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            enhance_head=enhance_head,
            vis_enhance=vis_enhance,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
