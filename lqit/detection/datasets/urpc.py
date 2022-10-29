from mmdet.datasets import CocoDataset, XMLDataset
from mmdet.registry import DATASETS

URPC_METAINFO = {
    'CLASSES': ('holothurian', 'echinus', 'scallop', 'starfish'),
    'PALETTE': [(235, 211, 70), (106, 90, 205), (160, 32, 240), (176, 23, 31)]
}


@DATASETS.register_module()
class URPCCocoDataset(CocoDataset):
    """Underwater Robot Professional Contest dataset `URPC.

    <https://arxiv.org/abs/2106.05681>`_
    """
    METAINFO = URPC_METAINFO


@DATASETS.register_module()
class URPCXMLDataset(XMLDataset):
    """"""
    METAINFO = URPC_METAINFO

    def __init__(self, **kwargs):
        # TODO: Currently not support
        super().__init__(**kwargs)
