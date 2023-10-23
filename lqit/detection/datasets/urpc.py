from mmdet.datasets import CocoDataset

from lqit.registry import DATASETS
from .xml_dataset import XMLDatasetWithMetaFile

URPC_METAINFO = {
    'classes': ('holothurian', 'echinus', 'scallop', 'starfish'),
    'palette': [(235, 211, 70), (106, 90, 205), (160, 32, 240), (176, 23, 31)]
}


@DATASETS.register_module()
class URPCCocoDataset(CocoDataset):
    """Underwater Robot Professional Contest dataset `URPC.

    <https://openi.pcl.ac.cn/OpenOrcinus_orca/URPC_opticalimage_dataset/datasets>`_
    """
    METAINFO = URPC_METAINFO


@DATASETS.register_module()
class URPCXMLDataset(XMLDatasetWithMetaFile):
    """"""
    METAINFO = URPC_METAINFO
