# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional, Sequence

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.engine.hooks import DetVisualizationHook
from mmdet.structures import DetDataSample
from mmengine.fileio import FileClient
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist

from lqit.registry import HOOKS


@HOOKS.register_module()
class DetEnhanceVisualizationHook(DetVisualizationHook):
    """Detection and Enhancement Visualization Hook. Used to visualize
    validation and testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.
    4. If ``show_in_enhance`` is True, it means that the prediction results are
        visualized on enhanced image

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(
        self,
        draw: bool = False,
        interval: int = 50,
        score_thr: float = 0.3,
        show: bool = False,
        wait_time: float = 0.,
        test_out_dir: Optional[str] = None,
        file_client_args: dict = dict(backend='disk'),
        show_in_enhance: bool = False,
    ):
        super().__init__(
            draw=draw,
            interval=interval,
            score_thr=score_thr,
            show=show,
            wait_time=wait_time,
            test_out_dir=test_out_dir,
            file_client_args=file_client_args)

        self.show_in_enhance = show_in_enhance

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:

        if self.draw is False:
            return

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        for data_sample in outputs:
            self._test_index += 1
            img_path = data_sample.img_path
            out_file = None
            if self.test_out_dir is not None:
                out_file = osp.basename(img_path)
                out_file = osp.join(self.test_out_dir, out_file)

            pre_enhance_img = data_sample.pred_pixel.pred_img
            pre_enhance_img = torch.unsqueeze(pre_enhance_img, 0)
            pre_enhance_img = F.interpolate(
                pre_enhance_img, size=data_sample.ori_shape, mode='bilinear')
            enhance_img = torch.squeeze(pre_enhance_img)
            enhance_img = enhance_img.cpu().numpy().astype(np.uint8).transpose(
                1, 2, 0)

            img_bytes = self.file_client.get(img_path)
            raw_img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            if self.show_in_enhance:
                img = enhance_img
            else:
                img = raw_img

            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'test_img',
                img,
                data_sample=data_sample,
                draw_gt=True,
                draw_pred=True,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=out_file,
                step=self._test_index)
