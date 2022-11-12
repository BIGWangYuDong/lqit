from typing import List

from lqit.edit.models.base_models import BaseGenerator
from lqit.edit.structures import BatchPixelData
from lqit.registry import MODELS
from lqit.utils.typing import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class SelfEnhanceGenerator(BaseGenerator):

    def __init__(self,
                 model: ConfigType,
                 spacial_loss: ConfigType = dict(
                     type='SpatialLoss', loss_weight=1.0),
                 tv_loss: ConfigType = dict(
                     type='MaskedTVLoss', loss_mode='mse', loss_weight=10.0),
                 perceptual_loss: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super().__init__(
            model=model, perceptual_loss=perceptual_loss, init_cfg=init_cfg)

        # build losses
        self.spacial_loss = MODELS.build(spacial_loss)
        self.tv_loss = MODELS.build(tv_loss)

    def loss(self, loss_input: BatchPixelData, batch_img_metas: List[dict]):
        """Calculate the loss based on the outputs of generator."""
        losses = dict()

        batch_outputs = loss_input.output
        batch_inputs = loss_input.input

        in_channels = self.model.in_channels
        batch_enhance_img = batch_outputs[:, :in_channels, :, :]
        batch_enhance_structure = batch_outputs[:, in_channels:, :, :]

        tv_loss = self.tv_loss(batch_enhance_structure)
        spacial_loss = self.spacial_loss(batch_enhance_img, batch_inputs)

        losses['tv_loss'] = tv_loss
        losses['spacial_loss'] = spacial_loss

        if self.spacial_loss is not None:
            de_batch_outputs = loss_input.de_output
            de_batch_inputs = loss_input.de_input
            if de_batch_outputs.shape[1] > 3:
                de_batch_outputs = de_batch_outputs[:, :in_channels, :, :]
            # norm to 0-1
            de_batch_outputs = de_batch_outputs / 255
            de_batch_inputs = de_batch_inputs / 255
            loss_percep, loss_style = self.perceptual_loss(
                de_batch_outputs, de_batch_inputs)
            if loss_percep is not None:
                losses['perceptual_loss'] = loss_percep
            if loss_style is not None:
                losses['style_loss'] = loss_style

        return losses

    def post_precess(self, outputs):
        assert outputs.dim() in [3, 4]
        in_channels = self.model.in_channels
        if outputs.dim() == 4:
            enhance_img = outputs[:, :in_channels, :, :]
        else:
            enhance_img = outputs[:in_channels, :, :]
        return enhance_img


@MODELS.register_module()
class SelfEnhanceUNetGenerator(BaseGenerator):

    def __init__(self,
                 model: ConfigType,
                 spacial_loss: ConfigType = dict(
                     type='SpatialLoss', loss_weight=1.0),
                 tv_loss: ConfigType = dict(
                     type='MaskedTVLoss', loss_mode='mse', loss_weight=10.0),
                 perceptual_loss: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super().__init__(
            model=model, perceptual_loss=perceptual_loss, init_cfg=init_cfg)

        # build losses
        self.spacial_loss = MODELS.build(spacial_loss)
        self.tv_loss = MODELS.build(tv_loss)

    def loss(self, loss_input: BatchPixelData, batch_img_metas: List[dict]):
        """Calculate the loss based on the outputs of generator."""
        losses = dict()

        batch_outputs = loss_input.output
        batch_inputs = loss_input.input

        tv_loss = self.tv_loss(batch_outputs)
        spacial_loss = self.spacial_loss(batch_outputs, batch_inputs)

        losses['tv_loss'] = tv_loss
        losses['spacial_loss'] = spacial_loss

        if self.spacial_loss is not None:
            de_batch_outputs = loss_input.de_output
            de_batch_inputs = loss_input.de_input
            # norm to 0-1
            de_batch_outputs = de_batch_outputs / 255
            de_batch_inputs = de_batch_inputs / 255
            loss_percep, loss_style = self.perceptual_loss(
                de_batch_outputs, de_batch_inputs)
            if loss_percep is not None:
                losses['perceptual_loss'] = loss_percep
            if loss_style is not None:
                losses['style_loss'] = loss_style

        return losses
