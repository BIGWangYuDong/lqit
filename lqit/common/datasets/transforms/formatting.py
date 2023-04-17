import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, PixelData
from numpy import ndarray
from torch import Tensor

from lqit.common.structures import DataSample
from lqit.registry import TRANSFORMS

try:
    from mmdet.structures.bbox import BaseBoxes
    HAS_MMDET = True
except ImportError:
    HAS_MMDET = False


def image_to_tensor(img: ndarray) -> Tensor:
    """Trans image to tensor.

    Args:
        img (np.ndarray): The original image.

    Returns:
        Tensor: The output tensor.
    """
    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    tensor = to_tensor(img)

    return tensor


@TRANSFORMS.register_module()
class PackInputs(BaseTransform):
    """Pack the inputs data for the detection / semantic segmentation / image
    enhancement / salient object detection.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
    }

    def __init__(
        self,
        meta_keys: dict = ('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'flip', 'flip_direction')
    ) -> None:
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        data_sample = DataSample()
        # process input image
        img = results['img']
        img_tensor = image_to_tensor(img)
        packed_results['inputs'] = img_tensor

        # process multi-input image(s)
        multi_input = PixelData()
        if 'gt_img' in results:
            img = results['gt_img']
            img_tensor = image_to_tensor(img)
            multi_input.img = img_tensor

        if 'depth' in results:
            depth = results['depth']
            depth_tensor = image_to_tensor(depth)
            multi_input.depth = depth_tensor

        if 'salient' in results:
            salient = results['salient']
            salient_tensor = image_to_tensor(salient)
            multi_input.salient = salient_tensor

        data_sample.multi_input = multi_input

        # process detection and instance segmentation gts
        if 'gt_bboxes' in results:
            assert HAS_MMDET, \
                'Cannot import `mmdet.structures.bbox.BaseBoxes`,' \
                'please install `mmdet` first.'

            if 'gt_ignore_flags' in results:
                valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
                ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]
            else:
                valid_idx = np.arange(len(results['gt_bboxes']))
                ignore_idx = np.array([])

            instance_data = InstanceData()
            ignore_instance_data = InstanceData()
            for key in self.mapping_table.keys():
                if key not in results:
                    continue
                if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                    if 'gt_ignore_flags' in results:
                        instance_data[
                            self.mapping_table[key]] = results[key][valid_idx]
                        ignore_instance_data[
                            self.mapping_table[key]] = results[key][ignore_idx]
                    else:
                        instance_data[self.mapping_table[key]] = results[key]
                else:
                    if 'gt_ignore_flags' in results:
                        instance_data[self.mapping_table[key]] = to_tensor(
                            results[key][valid_idx])
                        ignore_instance_data[self.mapping_table[key]] = \
                            to_tensor(results[key][ignore_idx])
                    else:
                        instance_data[self.mapping_table[key]] = to_tensor(
                            results[key])

            data_sample.gt_instances = instance_data
            data_sample.ignored_instances = ignore_instance_data

        # process proposals
        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        # process semantic segmentation
        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                data=to_tensor(results['gt_seg_map'][None,
                                                     ...].astype(np.int64)))
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        # process ground truth image(s)
        gt_pixel = PixelData()
        if 'gt_img' in results:
            # TODO: use wrapper to set gt_image and image have same process.
            #  note: some transfor should only used on image.
            gt_img = results['gt_img']
            gt_img_tensor = image_to_tensor(gt_img)
            gt_pixel.img = gt_img_tensor

        if 'gt_frequency' in results:
            gt_frequency = results['gt_frequency']
            gt_frequency_tensor = image_to_tensor(gt_frequency)
            gt_pixel.frequency = gt_frequency_tensor

        if 'gt_salient' in results:
            gt_salient = results['gt_salient']
            gt_salient_tensor = image_to_tensor(gt_salient)
            gt_pixel.salient = gt_salient_tensor

        if 'gt_edge' in results:
            gt_edge = results['gt_edge']
            gt_edge_tensor = image_to_tensor(gt_edge)
            gt_pixel.edge = gt_edge_tensor
        data_sample.gt_pixel = gt_pixel

        # process image metas
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
