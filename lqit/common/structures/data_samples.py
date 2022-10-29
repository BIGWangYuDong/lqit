from typing import List, Optional

from mmengine.structures import BaseDataElement, InstanceData, PixelData


class DataSample(BaseDataElement):
    """A data structure interface of LQIT. They are used as interfaces between
    different components.

    The attributes in ``DataSample`` are divided into several parts:

        - `multi_input`` (PixelData): The other input image, which deal with
          multi-input task (e.g. the depth of rgb-d). `multi_input` usually
          contains `depth`, `salient`, `fft_filter_img`, etc.
        - ``proposals`` (InstanceData): Region proposals used in two-stage
          detectors.
        - ``gt_instances`` (InstanceData): Ground truth of instance
          annotations.
        - ``pred_instances`` (InstanceData): Instances of model predictions.
        - ``ignored_instances`` (InstanceData): Instances to be ignored during
          training/testing.
        - ``gt_pixel`` (PixelData): Ground truth image(s), usually
          contains `img`, `sem_seg`, `salient`, `depth`, etc.
        - ``pred_pixel`` (PixelData): Image(s) of model predictions, usually
          contains `img`, `sem_seg`, `salient`, `depth`, etc.
        - ``gt_sem_seg``(PixelData): Ground truth of semantic segmentation.
        - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
        - ``seg_logits``(PixelData): Predicted logits of semantic segmentation.
    """

    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData):
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def ignored_instances(self) -> InstanceData:
        return self._ignored_instances

    @ignored_instances.setter
    def ignored_instances(self, value: InstanceData):
        self.set_field(value, '_ignored_instances', dtype=InstanceData)

    @ignored_instances.deleter
    def ignored_instances(self):
        del self._ignored_instances

    @property
    def gt_pixel(self) -> PixelData:
        return self._gt_pixel

    @gt_pixel.setter
    def gt_pixel(self, value: PixelData):
        self.set_field(value, '_gt_pixel', dtype=PixelData)

    @gt_pixel.deleter
    def gt_pixel(self):
        del self._gt_pixel

    @property
    def pred_pixel(self) -> PixelData:
        return self._pred_pixel

    @pred_pixel.setter
    def pred_pixel(self, value: PixelData):
        self.set_field(value, '_pred_pixel', dtype=PixelData)

    @pred_pixel.deleter
    def pred_pixel(self):
        del self._pred_pixel

    @property
    def multi_input(self) -> PixelData:
        return self._multi_input

    @multi_input.setter
    def multi_input(self, value: PixelData):
        self.set_field(value, '_multi_input', dtype=PixelData)

    @multi_input.deleter
    def multi_input(self):
        del self._multi_input

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_gt_sem_seg', dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self) -> None:
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_pred_sem_seg', dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self) -> None:
        del self._pred_sem_seg

    @property
    def seg_logits(self) -> PixelData:
        return self._seg_logits

    @seg_logits.setter
    def seg_logits(self, value: PixelData) -> None:
        self.set_field(value, '_seg_logits', dtype=PixelData)

    @seg_logits.deleter
    def seg_logits(self) -> None:
        del self._seg_logits


SampleList = List[DataSample]
OptSampleList = Optional[SampleList]
