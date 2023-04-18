from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS

RUOD_METAINFO = {
    'classes': ('holothurian', 'echinus', 'scallop', 'starfish', 'fish',
                'corals', 'diver', 'cuttlefish', 'turtle', 'jellyfish'),
    'palette': [(235, 211, 70), (106, 90, 205), (160, 32, 240), (176, 23, 31),
                (142, 0, 0), (230, 0, 0), (106, 0, 228), (60, 100, 0),
                (80, 100, 0), (70, 0, 0)]
}


@DATASETS.register_module()
class RUODDataset(CocoDataset):
    """Real-world Underwater Object Detection dataset `RUOD.

    <https://www.sciencedirect.com/science/article/abs/pii/S0925231222013169>`_
    """
    METAINFO = RUOD_METAINFO
