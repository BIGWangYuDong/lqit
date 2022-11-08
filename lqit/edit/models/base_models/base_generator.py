from abc import ABCMeta, abstractmethod
from typing import List

from mmengine.model import BaseModule

from lqit.edit.structures import BatchPixelData
from lqit.registry import MODELS
from lqit.utils.typing import ConfigType, OptConfigType, OptMultiConfig


class BaseGenerator(BaseModule, metaclass=ABCMeta):
    """Base class for EnhanceHead."""

    def __init__(self,
                 model: ConfigType,
                 pixel_loss: OptConfigType = None,
                 perceptual_loss: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        # build network
        self.model = MODELS.build(model)

        # build necessary loss
        self.pixel_loss = MODELS.build(pixel_loss) if pixel_loss else None
        self.perceptual_loss = MODELS.build(
            perceptual_loss) if perceptual_loss else None

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return self.model(x)

    @abstractmethod
    def loss(self, loss_input: BatchPixelData, batch_img_metas: List[dict]):
        """Calculate the loss based on the outputs of generator."""
        pass
