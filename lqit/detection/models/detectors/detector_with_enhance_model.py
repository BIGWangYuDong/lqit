import copy
from typing import Any, Dict, Optional, Tuple, Union

from mmengine.model import BaseModel
from mmengine.model.wrappers import MMDistributedDataParallel as MMDDP
from mmengine.utils import is_list_of
from torch import Tensor

from lqit.common.structures import SampleList
from lqit.detection.utils import merge_det_results
from lqit.edit.models.post_processor import add_pixel_pred_to_datasample
from lqit.registry import MODEL_WRAPPERS, MODELS
from lqit.utils import ConfigType, OptConfigType, OptMultiConfig

ForwardResults = Union[Dict[str, Tensor], SampleList, Tuple[Tensor], Tensor]


@MODELS.register_module()
class DetectorWithEnhanceModel(BaseModel):
    """Detector with enhance model.

    The `DetectorWithEnhanceModel` usually combines a detector and an enhance
    model. It has three train mode: `raw`, `enhance` and
    `both`. The `raw` mode only train the detector with raw image. The
    `enhance` mode only train the detector with enhance image. The `both` mode
    train the detector with both raw and enhance image.

    Args:
        detector (dict or ConfigDict): Config for detector.
        enhance_model (dict or ConfigDict, optional): Config for enhance model.
        loss_weight (list): Detection loss weight for raw and enhanced image.
            Only used when `train_mode` is `both`.
        vis_enhance (bool): Whether visualize enhanced image during inference.
            Defaults to False.
        train_mode (str): Train mode of detector, support `raw`, `enhance` and
            `both`. Defaults to `enhance`.
        pred_mode (str): Predict mode of detector, support `raw`, `enhance`,
            and `both`. Defaults to `enhance`.
        detach_enhance_img (bool): Whether stop the gradient of enhance image.
            Defaults to False.
        merge_cfg (dict or ConfigDict, optional): The config to control the
            merge process of raw and enhance image. Defaults to None.
        init_cfg (dict or ConfigDict, optional): The config to control the
           initialization. Defaults to None.
    """

    def __init__(self,
                 detector: ConfigType,
                 enhance_model: OptConfigType = None,
                 loss_weight: list = [0.5, 0.5],
                 vis_enhance: Optional[bool] = False,
                 train_mode: str = 'enhance',
                 pred_mode: str = 'enhance',
                 detach_enhance_img: bool = False,
                 merge_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        # if train_mode is `both`, the loss_weight should be a list of two
        # elements, which means the loss weight of raw and enhance image.
        if enhance_model is not None and train_mode == 'both':
            assert isinstance(loss_weight, list) and len(loss_weight) == 2

        assert pred_mode in ['raw', 'enhance', 'both']
        assert train_mode in ['raw', 'enhance', 'both']
        self.train_mode = train_mode
        self.pred_mode = pred_mode

        # build detector
        self.detector = MODELS.build(detector)
        # build enhance model
        if enhance_model is not None:
            self.enhance_model = MODELS.build(enhance_model)
        else:
            self.enhance_model = None

        self.detach_enhance_img = detach_enhance_img
        if vis_enhance:
            assert self.with_enhance_model
        self.vis_enhance = vis_enhance

        self.merge_cfg = merge_cfg
        if train_mode == 'both':
            # if train_mode is `both`, should have enhance_model.
            assert self.with_enhance_model
            assert merge_cfg is not None
            # The loss_weight should be a list of two elements, which means
            # the loss weight of raw and enhance image.
            assert isinstance(loss_weight, list) and len(loss_weight) == 2
            self.prefix_name = ['raw', 'enhance']
            self.loss_weight = loss_weight
        elif train_mode == 'enhance':
            # if train_mode is `enhance`, should have enhance_model.
            assert self.with_enhance_model
            self.prefix_name = ['enhance']
            self.loss_weight = [1.0]
        else:
            self.prefix_name = ['raw']
            self.loss_weight = [1.0]

    @property
    def with_enhance_model(self) -> bool:
        """Whether has a enhance model."""
        return (hasattr(self, 'enhance_model')
                and self.enhance_model is not None)

    def forward(self,
                data: dict,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        assert isinstance(data, dict)
        if mode == 'loss':
            return self.loss(data)
        elif mode == 'predict':
            return self.predict(data)
        elif mode == 'tensor':
            return self._forward(data)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def _run_forward(self, data: dict,
                     mode: str) -> Union[Dict[str, Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        assert isinstance(data, dict), \
            'The output of DataPreprocessor should be a dict, ' \
            'which only deal with `cast_data`. The data_preprocessor ' \
            'should process in forward.'
        results = self(data, mode=mode)

        return results

    def _preprocess_data(self, data: Union[dict, list, tuple]) -> tuple:
        """Preprocess data to a tuple of (batch_inputs, batch_data_samples)."""
        if isinstance(data, dict):
            batch_inputs = data['inputs']
            batch_data_samples = data['data_samples']
        elif isinstance(data, (list, tuple)):
            batch_inputs = data[0]
            batch_data_samples = data[1]
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            'list, tuple or dict, but got '
                            f'{type(data)}')
        return batch_inputs, batch_data_samples

    def _forward(self, data: dict) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            data (dict): Data sampled from dataloader, usually contains
                following keys.

                - inputs (list[Tensor]): A list of input image.
                - data_samples (:obj:`DataSample`): A list of DataSample.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()

        raw_det_data = self.detector.data_preprocessor(data, True)
        raw_batch_inputs, batch_det_data_samples = self._preprocess_data(
            raw_det_data)
        if self.pred_mode in ['enhance', 'both']:
            enhance_raw_data = self.enhance_model.data_preprocessor(data, True)

            # the enhance model should have `loss_and_predict` mode
            if isinstance(enhance_raw_data, dict):
                enhance_results = self.enhance_model(
                    **enhance_raw_data, mode='predict')
            elif isinstance(enhance_raw_data, (list, tuple)):
                enhance_results = self.enhance_model(
                    *enhance_raw_data, mode='predict')
            else:
                raise TypeError('Output of `data_preprocessor` should be '
                                'list, tuple or dict, but got '
                                f'{type(enhance_raw_data)}')
            enhance_img_list = [
                result.pred_pixel.pred_img for result in enhance_results
            ]
            # get enhance_batch_inputs of detector
            enhance_data = {'inputs': enhance_img_list}
            enhance_det_data = self.detector.data_preprocessor(
                enhance_data, True)
            enhance_batch_inputs, _ = self._preprocess_data(enhance_det_data)
            results = results + (enhance_batch_inputs, )

        if self.pred_mode == 'raw':
            raw_results = self.detector(
                raw_batch_inputs, batch_det_data_samples, mode='tensor')
            results = results + (raw_results, )
        elif self.pred_mode == 'enhance':
            enhance_results = self.detector(
                enhance_batch_inputs, batch_det_data_samples, mode='tensor')
            results = results + (enhance_results, )
        else:
            raw_results = self.detector(
                raw_batch_inputs, batch_det_data_samples, mode='tensor')
            results = results + (raw_results, )
            enhance_results = self.detector(
                enhance_batch_inputs, batch_det_data_samples, mode='tensor')
            results = results + (enhance_results, )
        return results

    def loss(self, data: dict) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            data (dict): Data sampled from dataloader, usually contains
                following keys.

                - inputs (list[Tensor]): A list of input image.
                - data_samples (:obj:`DataSample`): A list of DataSample.

        Returns:
            dict: A dictionary of loss components
        """
        losses = dict()

        # get batch_inputs and batch_data_samples of detector
        raw_det_data = self.detector.data_preprocessor(data, True)
        raw_batch_inputs, batch_det_data_samples = self._preprocess_data(
            raw_det_data)

        if self.train_mode in ['enhance', 'both']:
            # get batch_inputs and batch_data_samples of enhance_model
            enhance_raw_data = self.enhance_model.data_preprocessor(data, True)

            # the enhance model should have `loss_and_predict` mode
            if isinstance(enhance_raw_data, dict):
                results = self.enhance_model(
                    **enhance_raw_data, mode='loss_and_predict')
            elif isinstance(enhance_raw_data, (list, tuple)):
                results = self.enhance_model(
                    *enhance_raw_data, mode='loss_and_predict')
            else:
                raise TypeError('Output of `data_preprocessor` should be '
                                'list, tuple or dict, but got '
                                f'{type(enhance_raw_data)}')
            # results should have `enhance_loss` and `enhance_results`
            enhance_loss, enhance_results = results
            losses.update(enhance_loss)

            enhance_img_list = [
                result.pred_pixel.pred_img for result in enhance_results
            ]
            # get enhance_batch_inputs of detector
            enhance_data = {'inputs': enhance_img_list}
            enhance_det_data = self.detector.data_preprocessor(
                enhance_data, True)

            enhance_batch_inputs, _ = self._preprocess_data(enhance_det_data)
            if self.detach_enhance_img:
                # if self.detach_enhance_img is True, stop the gradient of
                # enhance image.
                enhance_batch_inputs = enhance_batch_inputs.detach()

        if self.train_mode == 'both':
            batch_inputs_list = [raw_batch_inputs, enhance_batch_inputs]
        elif self.train_mode == 'raw':
            batch_inputs_list = [raw_batch_inputs]
        else:
            batch_inputs_list = [enhance_batch_inputs]

        for i, batch_inputs in enumerate(batch_inputs_list):
            temp_losses = self.detector(
                batch_inputs, batch_det_data_samples, mode='loss')

            for name, value in temp_losses.items():
                if 'loss' in name:
                    if isinstance(value, Tensor):
                        value = value * self.loss_weight[i]
                    elif is_list_of(value, Tensor):
                        value = [_v * self.loss_weight[i] for _v in value]
                losses[f'{self.prefix_name[i]}_{name}'] = value
        return losses

    def predict(self, data: dict) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            data (dict): Data sampled from dataloader, usually contains
                following keys.

                - inputs (list[Tensor]): A list of input image.
                - data_samples (:obj:`DataSample`): A list of DataSample.

        Returns:
            list[:obj:`DataSample`]: Return the detection results of the
            input images. The returns value is DataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # get batch_inputs and batch_data_samples of detector
        raw_det_data = self.detector.data_preprocessor(data, True)
        raw_batch_inputs, batch_data_samples = self._preprocess_data(
            raw_det_data)

        if self.pred_mode in ['enhance', 'both']:
            enhance_raw_data = self.enhance_model.data_preprocessor(data, True)
            # get enhance image
            if isinstance(enhance_raw_data, dict):
                results = self.enhance_model(
                    **enhance_raw_data, mode='predict')
            elif isinstance(enhance_raw_data, (list, tuple)):
                results = self.enhance_model(*enhance_raw_data, mode='predict')
            else:
                raise TypeError('Output of `data_preprocessor` should be '
                                'list, tuple or dict, but got '
                                f'{type(enhance_raw_data)}')
            enhance_img_list = [
                result.pred_pixel.pred_img for result in results
            ]
            # add into batch_data_samples
            batch_data_samples = add_pixel_pred_to_datasample(
                data_samples=batch_data_samples, pixel_list=enhance_img_list)
            enhance_data = {'inputs': enhance_img_list}
            enhance_det_data = self.detector.data_preprocessor(
                enhance_data, True)
            enhance_batch_inputs, _ = self._preprocess_data(enhance_det_data)

        if self.pred_mode == 'raw':
            batch_data_samples = self.detector(
                raw_batch_inputs, batch_data_samples, mode='predict')
        elif self.pred_mode == 'enhance':
            batch_data_samples = self.detector(
                enhance_batch_inputs, batch_data_samples, mode='predict')
        else:
            raw_batch_data_samples = copy.deepcopy(batch_data_samples)
            raw_batch_data_samples = self.detector(
                raw_batch_inputs, raw_batch_data_samples, mode='predict')

            enhance_batch_data_samples = copy.deepcopy(batch_data_samples)
            enhance_batch_data_samples = self.detector(
                enhance_batch_inputs,
                enhance_batch_data_samples,
                mode='predict')

            batch_data_samples = []
            for raw_data_sample, enhance_data_sample in zip(
                    raw_batch_data_samples, enhance_batch_data_samples):
                batch_data_samples.append(
                    [raw_data_sample, enhance_data_sample])
            batch_data_samples = merge_det_results(batch_data_samples,
                                                   self.merge_cfg)
        return batch_data_samples


@MODEL_WRAPPERS.register_module()
class SelfEnhanceModelDDP(MMDDP):

    def _run_forward(self, data: Union[dict, tuple, list], mode: str) -> Any:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        assert isinstance(data, dict), \
            'The output of DataPreprocessor should be a dict, ' \
            'which only deal with `cast_data`. The data_preprocessor ' \
            'should process in forward.'
        results = self(data, mode=mode)

        return results
