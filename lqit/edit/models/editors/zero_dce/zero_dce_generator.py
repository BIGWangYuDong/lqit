from typing import List

from lqit.edit.models.base_models import BaseGenerator
from lqit.edit.structures import BatchPixelData
from lqit.registry import MODELS
from lqit.utils.typing import ConfigType, OptMultiConfig


@MODELS.register_module()
class ZeroDCEGenerator(BaseGenerator):

    def __init__(self,
                 zero_dce: ConfigType,
                 color_loss: ConfigType = dict(
                     type='ColorLoss', loss_weight=5.0),
                 spacial_loss: ConfigType = dict(
                     type='SpatialLoss', loss_weight=1.0),
                 tv_loss: ConfigType = dict(
                     type='MaskedTVLoss', loss_mode='mse', loss_weight=200.0),
                 exposure_loss: ConfigType = dict(
                     type='ExposureLoss',
                     patch_size=16,
                     mean_val=0.6,
                     loss_weight=10.),
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        # build network
        self.model = MODELS.build(zero_dce)

        # build losses
        self.color_loss = MODELS.build(color_loss)
        self.spacial_loss = MODELS.build(spacial_loss)
        self.tv_loss = MODELS.build(tv_loss)
        self.exposure_loss = MODELS.build(exposure_loss)

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
        batch_inputs = loss_input.input

        # ZeroDCE return enhance loss and curve at the same time.
        in_channels = self.model.in_channels
        batch_enhance_img = batch_outputs[:, :in_channels, :, :]
        batch_curve = batch_outputs[:, in_channels:, :, :]

        tv_loss = self.tv_loss(batch_curve)
        spacial_loss = self.spacial_loss(batch_enhance_img, batch_inputs)
        color_loss = self.color_loss(batch_enhance_img)
        exposure_loss = self.exposure_loss(batch_enhance_img)

        losses = dict(
            tv_loss=tv_loss,
            spacial_loss=spacial_loss,
            color_loss=color_loss,
            exposure_loss=exposure_loss)

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
