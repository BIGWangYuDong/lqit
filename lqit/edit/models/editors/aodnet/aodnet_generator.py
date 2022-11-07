from typing import List

from lqit.edit.models.base_models import BaseGenerator
from lqit.edit.structures import BatchPixelData
from lqit.registry import MODELS
from lqit.utils.typing import ConfigType, OptMultiConfig


@MODELS.register_module()
class AODNetGenerator(BaseGenerator):

    def __init__(self,
                 model: ConfigType,
                 pixel_loss: ConfigType = dict(
                     type='MSELoss', loss_weight=1.0),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(model=model, pixel_loss=pixel_loss, init_cfg=init_cfg)

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
        batch_gt = loss_input.gt

        pixel_loss = self.pixel_loss(batch_outputs, batch_gt)

        losses = dict(pixel_loss=pixel_loss)
        return losses

    def post_precess(self, outputs):
        # ZeroDCE return enhance loss and curve at the same time.
        assert outputs.dim() in [3, 4]
        in_channels = self.model.in_channels
        if outputs.dim() == 4:
            enhance_img = outputs[:, :in_channels, :, :]
        else:
            enhance_img = outputs[:in_channels, :, :]
        return enhance_img
