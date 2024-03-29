# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional, Sequence

import mmcv
import numpy as np
from mmdet.engine.hooks import DetVisualizationHook
from mmdet.structures import DetDataSample
from mmengine.fileio import get
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist

from lqit.registry import HOOKS


@HOOKS.register_module()
class EnhanceDetVisualizationHook(DetVisualizationHook):
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
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        show_on_enhance (bool): Whether show the detection results on the
            enhanced image. Defaults to False
        draw_gt (bool): Whether to draw GT DetDataSample. Default to False.
        draw_pred (bool): Whether to draw Prediction DetDataSample.
            Defaults to True.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 score_thr: float = 0.3,
                 show: bool = False,
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 backend_args: dict = None,
                 show_on_enhance: bool = False,
                 draw_gt: bool = False,
                 draw_pred: bool = True) -> None:
        super().__init__(
            draw=draw,
            interval=interval,
            score_thr=score_thr,
            show=show,
            wait_time=wait_time,
            test_out_dir=test_out_dir,
            backend_args=backend_args)

        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self.show_on_enhance = show_on_enhance

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DetDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path = outputs[0].img_path
        if self.show_on_enhance:
            img = outputs[0].pred_pixel.pred_img
            # convert to rgb
            # TODO: Support check whether image is RGB or BGR
            img = img[[2, 1, 0], ...]
            img = img.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            h, w = outputs[0].ori_shape
            img = mmcv.imresize(img, size=(w, h))
        else:
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'val_img',
                img,
                data_sample=outputs[0],
                draw_gt=self.draw_gt,
                draw_pred=self.draw_pred,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:

        if self.draw is False:
            return

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.img_path
            out_file = None
            if self.test_out_dir is not None:
                out_file = osp.basename(img_path)
                out_file = osp.join(self.test_out_dir, out_file)

            if self.show_on_enhance:
                img = data_sample.pred_pixel.pred_img
                # convert to rgb
                # TODO: Support check whether image is RGB or BGR
                img = img[[2, 1, 0], ...]
                img = img.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
                h, w = data_sample.ori_shape
                img = mmcv.imresize(img, size=(w, h))
            else:
                img_bytes = get(img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'test_img',
                img,
                data_sample=data_sample,
                draw_gt=self.draw_gt,
                draw_pred=self.draw_pred,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=out_file,
                step=self._test_index)
