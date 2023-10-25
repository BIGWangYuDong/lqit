# Modified from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/test_time_augs/det_tta.py  # noqa
from typing import List, Tuple

import torch
from mmcv.ops import batched_nms
from mmdet.structures.bbox import bbox_flip
from mmengine.structures import InstanceData
from torch import Tensor

from lqit.common.structures import DataSample


def merge_preds(data_samples_list: List[List[DataSample]],
                merge_cfg: dict) -> List[DataSample]:
    """Merge batch predictions of enhanced data.

    Args:
        data_samples_list (List[List[DataSample]]): List of predictions
            of all enhanced data. The outer list indicates images, and the
            inner list corresponds to the different views of one image.
            Each element of the inner list is a ``DataSample``.
        merge_cfg (dict): Config of merge method.

    Returns:
        List[DataSample]: Merged batch prediction.
    """
    merged_data_samples = []
    for data_samples in data_samples_list:
        merged_data_samples.append(_merge_single_sample(data_samples))
    return merged_data_samples


def _merge_single_sample(data_samples: List[DataSample],
                         merge_cfg: dict) -> DataSample:
    """Merge predictions which come form the different views of one image to
    one prediction.

    Args:
        data_samples (List[DataSample]): List of predictions
        of enhanced data which come form one image.
        merge_cfg (dict): Config of merge method.

    Returns:
        List[DataSample]: Merged prediction.
    """
    aug_bboxes = []
    aug_scores = []
    aug_labels = []
    img_metas = []
    # TODO: support instance segmentation TTA
    assert data_samples[0].pred_instances.get('masks', None) is None, \
        'TTA of instance segmentation does not support now.'
    for data_sample in data_samples:
        aug_bboxes.append(data_sample.pred_instances.bboxes)
        aug_scores.append(data_sample.pred_instances.scores)
        aug_labels.append(data_sample.pred_instances.labels)
        img_metas.append(data_sample.metainfo)

    merged_bboxes, merged_scores = merge_aug_bboxes(aug_bboxes, aug_scores,
                                                    img_metas)
    merged_labels = torch.cat(aug_labels, dim=0)

    if merged_bboxes.numel() == 0:
        return data_samples[0]

    det_bboxes, keep_idxs = batched_nms(merged_bboxes, merged_scores,
                                        merged_labels, merge_cfg.nms)

    det_bboxes = det_bboxes[:merge_cfg.max_per_img]
    det_labels = merged_labels[keep_idxs][:merge_cfg.max_per_img]

    results = InstanceData()
    _det_bboxes = det_bboxes.clone()
    results.bboxes = _det_bboxes[:, :-1]
    results.scores = _det_bboxes[:, -1]
    results.labels = det_labels
    det_results = data_samples[0]
    det_results.pred_instances = results
    return det_results


def merge_aug_bboxes(aug_bboxes: List[Tensor], aug_scores: List[Tensor],
                     img_metas: List[str]) -> Tuple[Tensor, Tensor]:
    """Merge augmented detection bboxes and scores.

    Args:
        aug_bboxes (list[Tensor]): shape (n, 4*#class)
        aug_scores (list[Tensor] or None): shape (n, #class)
    Returns:
        tuple[Tensor]: ``bboxes`` with shape (n,4), where
        4 represent (tl_x, tl_y, br_x, br_y)
        and ``scores`` with shape (n,).
    """
    recovered_bboxes = []
    for bboxes, img_info in zip(aug_bboxes, img_metas):
        ori_shape = img_info['ori_shape']
        flip = img_info['flip']
        flip_direction = img_info['flip_direction']
        if flip:
            bboxes = bbox_flip(
                bboxes=bboxes, img_shape=ori_shape, direction=flip_direction)
        recovered_bboxes.append(bboxes)
    bboxes = torch.cat(recovered_bboxes, dim=0)
    if aug_scores is None:
        return bboxes
    else:
        scores = torch.cat(aug_scores, dim=0)
        return bboxes, scores
