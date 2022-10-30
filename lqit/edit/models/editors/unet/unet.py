from typing import List

from lqit.edit.models.base_models import BaseGenerator
from lqit.edit.structures import BatchPixelData
from lqit.registry import MODELS
from lqit.utils.typing import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class UNet(BaseGenerator):

    def __init__(self,
                 unet: ConfigType,
                 pixel_loss: ConfigType,
                 perceptual_loss: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            pixel_loss=pixel_loss,
            perceptual_loss=perceptual_loss,
            init_cfg=init_cfg)
        self.model = MODELS.build(unet)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return self.model(x)

    def loss(self, loss_input: BatchPixelData, batch_img_metas: List[dict]):
        """Calculate the loss based on the outputs of generator."""
        batch_outputs = loss_input.output
        batch_gt_pixel = loss_input.gt
        pixel_loss = self.pixel_loss(batch_outputs, batch_gt_pixel)

        return dict(pixel_loss=pixel_loss)
