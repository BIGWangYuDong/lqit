from mmdet.datasets import CocoDataset

from lqit.registry import DATASETS

DUO_METAINFO = {
    'classes': ('holothurian', 'echinus', 'scallop', 'starfish'),
    'palette': [(235, 211, 70), (106, 90, 205), (160, 32, 240), (176, 23, 31)]
}


@DATASETS.register_module()
class DUODataset(CocoDataset):
    """Detecting Underwater Objects dataset `DUO.

    <https://arxiv.org/abs/2106.05681>`_
    """
    METAINFO = DUO_METAINFO
