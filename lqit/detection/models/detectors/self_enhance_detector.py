import copy
from typing import Any, Dict, Optional, Tuple, Union

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModel
from mmengine.model.wrappers import MMDistributedDataParallel as MMDDP
from mmengine.registry import MODEL_WRAPPERS
from mmengine.utils import is_list_of
from torch import Tensor

from lqit.common.structures import SampleList
from lqit.edit.models.post_processor import add_pixel_pred_to_datasample

ForwardResults = Union[Dict[str, Tensor], SampleList, Tuple[Tensor], Tensor]


@MODELS.register_module()
class SelfEnhanceDetector(BaseModel):
    """Base class for two-stage detectors with enhance head.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 detector: ConfigType,
                 enhance_model: OptConfigType = None,
                 loss_weight: list = [0.5, 0.5],
                 vis_enhance: Optional[bool] = False,
                 train_mode: str = 'enhance',
                 pred_mode: str = 'enhance',
                 detach_enhance_img: bool = False,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(loss_weight, list) and len(loss_weight) == 2
        assert pred_mode in ['raw', 'enhance', 'both']
        assert train_mode in ['raw', 'enhance', 'both']

        self.detector = MODELS.build(detector)
        if enhance_model is not None:
            self.enhance_model = MODELS.build(enhance_model)
        else:
            self.enhance_model = None

        self.detach_enhance_img = detach_enhance_img
        if pred_mode in ['enhance', 'both']:
            self.vis_enhance = True
        else:
            self.vis_enhance = vis_enhance
        self.pred_mode = pred_mode

        self.train_mode = train_mode

        if self.with_enhance_model:
            if self.train_mode == 'both':
                self.prefix_name = ['raw', 'enhance']
                self.loss_weight = loss_weight
            elif self.train_mode == 'enhance':
                self.prefix_name = ['enhance']
                self.loss_weight = [1.0]
            else:
                raise KeyError('enhance_mode is not None, train_mode only '
                               'support `both` or `enhance`')
        else:
            if self.train_mode == 'both':
                self.prefix_name = ['raw_1', 'raw_2']
                self.loss_weight = loss_weight
            elif self.train_mode == 'raw':
                self.prefix_name = ['raw']
                self.loss_weight = [1.0]
            else:
                raise KeyError('enhance_mode is None, train_mode only '
                               'support `both` or `raw`')

    @property
    def with_enhance_model(self) -> bool:
        """bool: whether the detector has a Enhance Model"""
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

        import mmengine

        batch_data_samples = mmengine.load('work_dirs/batch_data_samples.pkl')

        det_data_raw = {'inputs': [data[0]]}
        results = ()
        det_data = self.detector.data_preprocessor(det_data_raw, True)
        det_data['data_samples'] = batch_data_samples
        if isinstance(det_data, dict):
            det_tensor = self.detector(**det_data, mode='tensor')
        elif isinstance(det_data, (list, tuple)):
            det_tensor = self.detector(*det_data, mode='tensor')
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            'list, tuple or dict, but got '
                            f'{type(det_data)}')

        if not isinstance(det_tensor, tuple):
            assert isinstance(det_tensor, Tensor)
            results = (det_tensor, )
        elif isinstance(det_tensor, tuple):
            results = results + det_tensor
        else:
            raise TypeError('The output of detector should be Tensor or '
                            f'Tuple, but got {type(det_tensor)}')

        enhance_data_raw = {
            'inputs': [data[0]],
            'data_samples': batch_data_samples
        }
        enhance_data = self.enhance_model.data_preprocessor(
            enhance_data_raw, True)
        enhance_data['data_samples'] = batch_data_samples
        if isinstance(enhance_data, dict):
            enhance_tensor = self.enhance_model(**enhance_data, mode='predict')
        elif isinstance(enhance_data, (list, tuple)):
            enhance_tensor = self.enhance_model(*enhance_data, mode='predict')
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            'list, tuple or dict, but got '
                            f'{type(enhance_data)}')
        enhance_tensor = enhance_tensor[0].pred_pixel.pred_img
        if not isinstance(enhance_tensor, tuple):
            assert isinstance(enhance_tensor, Tensor)
            results = results + (enhance_tensor, )
        elif isinstance(enhance_tensor, tuple):
            results = results + enhance_tensor
        else:
            raise TypeError('The output of detector should be Tensor or '
                            f'Tuple, but got {type(det_tensor)}')

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
        if isinstance(raw_det_data, dict):
            raw_batch_inputs = raw_det_data['inputs']
            batch_det_data_samples = raw_det_data['data_samples']
        elif isinstance(raw_det_data, (list, tuple)):
            raw_batch_inputs = raw_det_data[0]
            batch_det_data_samples = raw_det_data[1]
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            'list, tuple or dict, but got '
                            f'{type(raw_det_data)}')

        if self.with_enhance_model:
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

            if isinstance(enhance_det_data, dict):
                enhance_batch_inputs = enhance_det_data['inputs']
            elif isinstance(enhance_det_data, (list, tuple)):
                enhance_batch_inputs = enhance_det_data[0]
            else:
                raise TypeError('Output of `data_preprocessor` should be '
                                'list, tuple or dict, but got '
                                f'{type(enhance_det_data)}')
            # if stop the gradient
            if self.detach_enhance_img:
                enhance_batch_inputs = enhance_batch_inputs.detach()
        else:
            # the enhance_batch_inputs same as raw_batch_inputs
            enhance_batch_inputs = copy.deepcopy(raw_batch_inputs)

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
        import time
        print('-' * 20)
        t1 = time.time()
        raw_det_data = self.detector.data_preprocessor(data, True)
        t2 = time.time()
        print(t2 - t1)
        if isinstance(raw_det_data, dict):
            raw_batch_inputs = raw_det_data['inputs']
            batch_data_samples = raw_det_data['data_samples']
        elif isinstance(raw_det_data, (list, tuple)):
            raw_batch_inputs = raw_det_data[0]
            batch_data_samples = raw_det_data[1]
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            'list, tuple or dict, but got '
                            f'{type(raw_det_data)}')
        t3 = time.time()
        print(t3 - t2)
        if self.vis_enhance:
            assert self.with_enhance_model

            enhance_raw_data = self.enhance_model.data_preprocessor(data, True)
            t4 = time.time()
            print(t4 - t3)
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
            t5 = time.time()
            print(t5 - t4)

            enhance_img_list = [
                result.pred_pixel.pred_img for result in results
            ]
            # add into batch_data_samples
            batch_data_samples = add_pixel_pred_to_datasample(
                data_samples=batch_data_samples, pixel_list=enhance_img_list)
            t6 = time.time()
            print(t6 - t5)
        if self.pred_mode in ['enhance', 'both']:
            # get enhance_batch_inputs of detector
            enhance_data = {'inputs': enhance_img_list}
            enhance_det_data = self.detector.data_preprocessor(
                enhance_data, True)
            if isinstance(enhance_det_data, dict):
                enhance_batch_inputs = enhance_det_data['inputs']
            elif isinstance(enhance_det_data, (list, tuple)):
                enhance_batch_inputs = enhance_det_data[0]
            else:
                raise TypeError('Output of `data_preprocessor` should be '
                                'list, tuple or dict, but got '
                                f'{type(raw_det_data)}')

        if self.pred_mode == 'raw':
            batch_data_samples = self.detector(
                raw_batch_inputs, batch_data_samples, mode='predict')
            return batch_data_samples
        elif self.pred_mode == 'enhance':
            batch_data_samples = self.detector(
                enhance_batch_inputs, batch_data_samples, mode='predict')
            t7 = time.time()
            print(t7 - t6)
            print('-' * 20)
            return batch_data_samples
        else:
            # TODO: support aug_test
            raise NotImplementedError
            # batch_data_samples = self.detector(
            #     raw_batch_inputs, batch_data_samples, mode='predict')
            # cp_batch_data_samples = copy.deepcopy(batch_data_samples)
            # cp_batch_data_samples = self.detector(
            #     enhance_batch_inputs, cp_batch_data_samples, mode='predict')


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
