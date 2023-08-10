from mmdet.datasets import CocoDataset

from lqit.registry import DATASETS


URPC_METAINFO = {
    'classes': ('holothurian', 'echinus', 'starfish', 'scallop', 'waterweeds'),
    'palette': [(235, 211, 70), (106, 90, 205), (160, 32, 240), (176, 23, 31), (0, 0, 0)]
}


@DATASETS.register_module()
class URPCAAAICocoDataset(CocoDataset):
    """Underwater Robot Professional Contest dataset `URPC.
    <https://arxiv.org/abs/2106.05681>`_
    
    With waterweeds
    """
    METAINFO = URPC_METAINFO

from mmyolo.datasets.yolov5_coco import BatchShapePolicyDataset

@DATASETS.register_module()
class YOLOv5URPCAAAICocoDataset(BatchShapePolicyDataset, URPCAAAICocoDataset):
    """Dataset for YOLOv5 COCO Dataset.

    We only add `BatchShapePolicy` function compared with CocoDataset. See
    `mmyolo/datasets/utils.py#BatchShapePolicy` for details
    """
    pass