from typing import List

from lqit.edit.models.base_models import BaseGenerator
from lqit.edit.structures import BatchPixelData
from lqit.registry import MODELS
from lqit.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class UNetGenerator(BaseGenerator):

    def __init__(self,
                 model: ConfigType,
                 pixel_loss: ConfigType,
                 perceptual_loss: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            model=model,
            pixel_loss=pixel_loss,
            perceptual_loss=perceptual_loss,
            init_cfg=init_cfg)

    def loss(self, loss_input: BatchPixelData, batch_img_metas: List[dict]):
        """Calculate the loss based on the outputs of generator."""
        batch_outputs = loss_input.output
        batch_gt_pixel = loss_input.gt
        pixel_loss = self.pixel_loss(batch_outputs, batch_gt_pixel)

        return dict(pixel_loss=pixel_loss)
