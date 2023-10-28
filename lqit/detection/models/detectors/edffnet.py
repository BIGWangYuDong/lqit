from lqit.registry import MODELS
from lqit.utils import ConfigType, OptMultiConfig
from .detector_with_enhance_head import DetectorWithEnhanceHead


@MODELS.register_module()
class EDFFNet(DetectorWithEnhanceHead):
    """Implementation of EDFFNet.

    <https://link.springer.com/article/10.1007/s11760-022-02410-0>`_
    """

    def __init__(self,
                 detector: ConfigType,
                 edge_head: ConfigType,
                 process_gt_preprocessor: bool = False,
                 vis_enhance: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:
        assert not process_gt_preprocessor, \
            'process_gt_preprocessor is not supported in EDFFNet'
        super().__init__(
            detector=detector,
            enhance_head=edge_head,
            process_gt_preprocessor=process_gt_preprocessor,
            vis_enhance=vis_enhance,
            init_cfg=init_cfg)
