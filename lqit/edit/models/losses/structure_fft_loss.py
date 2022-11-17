import torch
import torch.nn as nn
import torch.nn.functional as F

from lqit.registry import MODELS
from .utils import mask_reduce_loss


@MODELS.register_module()
class StructureFFTLoss(nn.Module):

    def __init__(self,
                 radius: int = 8,
                 shape: str = 'cycle',
                 channel_mean: bool = False,
                 loss_weight=1.0):
        super().__init__()
        if shape == 'cycle':
            self.center_mask = self._cycle_mask(radius)
        elif shape == 'square':
            self.center_mask = self._square_mask(radius)
        else:
            raise NotImplementedError('Only support `cycle` and `square`, '
                                      f'but got {shape}')
        self.channel_mean = channel_mean
        self.radius = radius
        self.loss_weight = loss_weight

    def _cycle_mask(self, radius):
        x = torch.arange(0, 2 * radius)[None, :]
        y = torch.arange(2 * radius - 1, -1, -1)[:, None]
        cycle_mask = ((x - radius)**2 + (y - radius)**2) <= (radius - 1)**2
        return cycle_mask

    def _square_mask(self, radius):
        square_mask = torch.ones((radius * 2, radius * 2), dtype=torch.bool)
        return square_mask

    def _get_mask(self, img):
        device = img.device
        center_mask = self.center_mask.to(device)
        hw_img = img[0, ...]
        mask = torch.zeros_like(hw_img, dtype=torch.bool)
        height, width = mask.shape[0], mask.shape[1]
        x_c, y_c = width // 2, height // 2

        mask[y_c - self.radius:y_c + self.radius,
             x_c - self.radius:x_c + self.radius] = center_mask
        mask = ~mask
        return mask

    def forward(self, pred, target, batch_img_metas, **kwargs):
        """Forward function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.

        Returns:
            Tensor: Calculated loss.
        """
        assert pred.shape == target.shape
        if self.channel_mean:
            pred = torch.mean(pred, dim=1, keepdim=True)
            target = torch.mean(target, dim=1, keepdim=True)

        losses = []

        for _pred, _target, img_meta in zip(pred, target, batch_img_metas):
            assert len(_pred.shape) == len(_target.shape) == 3
            h, w = img_meta['img_shape']
            no_padding_pred = _pred[:, :h, :w]
            no_padding_target = _target[:, :h, :w]
            mask = self._get_mask(no_padding_pred)
            high_pass_pred = self.get_high_pass_img(no_padding_pred, mask)
            high_pass_target = self.get_high_pass_img(no_padding_target, mask)

            norm_high_pass_pred = high_pass_pred / 255
            norm_high_pass_target = high_pass_target / 255

            loss = F.l1_loss(
                norm_high_pass_pred, norm_high_pass_target, reduction='mean')
            losses.append(loss)
        total_loss = sum(_loss.mean() for _loss in losses)
        return total_loss * self.loss_weight

    @staticmethod
    def get_high_pass_img(img, mask):
        channel_img_list = []
        for i in range(img.size(0)):
            channel_img = img[i, ...]
            f = torch.fft.fft2(channel_img)
            fshift = torch.fft.fftshift(f)
            filter_fshift = fshift * mask

            ishift = torch.fft.ifftshift(filter_fshift)
            high_pass_img = torch.fft.ifft2(ishift)
            high_pass_img = torch.abs(high_pass_img).clip_(min=0, max=255)
            channel_img_list.append(high_pass_img[None, ...])
        result_img = torch.cat(channel_img_list, dim=0)
        return result_img


@MODELS.register_module()
class HighPassFFTLoss(StructureFFTLoss):

    def __init__(self,
                 radius: int = 16,
                 shape: str = 'cycle',
                 channel_mean: bool = False,
                 norm_input: bool = False,
                 loss_weight=1.0):
        super().__init__(
            radius=radius,
            shape=shape,
            channel_mean=channel_mean,
            loss_weight=loss_weight)
        self.norm_input = norm_input

    @staticmethod
    def get_real(complex_img):
        real, imaginary = complex_img.real, complex_img.imag
        return real, imaginary

    @staticmethod
    def high_pass_fft(img, mask):
        channel_img_list = []
        for i in range(img.size(0)):
            channel_img = img[i, ...]
            f = torch.fft.fft2(channel_img)
            fshift = torch.fft.fftshift(f)
            filter_fshift = fshift * mask

            channel_img_list.append(filter_fshift[None, ...])
        result_img = torch.cat(channel_img_list, dim=0)
        return result_img

    def fft_loss(self, pred, target, weight=None, reduction='mean'):
        pred_real, pred_imaginary = self.get_real(pred)
        gt_real, gt_imaginary = self.get_real(target)

        real_loss = torch.pow((pred_real - gt_real), exponent=2)
        imaginary_loss = torch.pow((pred_imaginary - gt_imaginary), exponent=2)
        loss = torch.sqrt((real_loss + imaginary_loss) + 1e-10)
        if weight is not None:
            assert weight.ndim == loss.ndim
            assert len(weight) == len(pred)
        loss = mask_reduce_loss(loss, weight=weight, reduction=reduction)
        return loss

    def forward(self, pred, target, batch_img_metas, weight=None, **kwargs):
        """Forward function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.

        Returns:
            Tensor: Calculated loss.
        """
        assert pred.shape == target.shape
        if self.channel_mean:
            pred = torch.mean(pred, dim=1, keepdim=True)
            target = torch.mean(target, dim=1, keepdim=True)

        losses = []

        for _pred, _target, img_meta in zip(pred, target, batch_img_metas):
            assert len(_pred.shape) == len(_target.shape) == 3
            h, w = img_meta['img_shape']
            no_padding_pred = _pred[:, :h, :w]
            no_padding_target = _target[:, :h, :w]
            mask = self._get_mask(no_padding_pred)

            high_pass_pred = self.high_pass_fft(no_padding_pred, mask)
            high_pass_target = self.high_pass_fft(no_padding_target, mask)
            if self.norm_input:
                high_pass_pred = high_pass_pred / 255
                high_pass_target = high_pass_target / 255

            loss = self.fft_loss(
                high_pass_pred, high_pass_target, weight,
                reduction='mean') * self.loss_weight
            losses.append(loss)
        return losses
