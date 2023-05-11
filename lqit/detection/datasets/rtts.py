from mmdet.datasets import CocoDataset

from lqit.registry import DATASETS

RTTS_METAINFO = {
    'classes': ('bicycle', 'bus', 'car', 'motorbike', 'person'),
    'palette': [(255, 97, 0), (0, 201, 87), (176, 23, 31), (138, 43, 226),
                (30, 144, 255)]
}


@DATASETS.register_module()
class RTTSCocoDataset(CocoDataset):
    """Foggy object detection dataset in RESIDE `RTSS.

    <https://arxiv.org/abs/1712.04143>`_
    """
    METAINFO = RTTS_METAINFO
