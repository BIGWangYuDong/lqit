from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS

RTTS_METAINFO = {
    'CLASSES': ('bicycle', 'bus', 'car', 'motorbike', 'person'),
    'PALETTE': [(255, 97, 0), (0, 201, 87), (176, 23, 31), (138, 43, 226),
                (30, 144, 255)]
}


@DATASETS.register_module()
class RTTSCocoDataset(CocoDataset):
    """Foggy object detection dataset in RESIDE `RTSS.

    <https://arxiv.org/abs/1712.04143>`_
    """
    METAINFO = RTTS_METAINFO
