from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lqit.registry import MODELS
from lqit.utils import OptConfigType
from .utils import mask_reduce_loss


@MODELS.register_module()
class StructureFFTLoss(nn.Module):

    def __init__(self,
                 radius: int = 64,
                 pass_type: str = 'high',
                 shape: str = 'cycle',
                 channel_mean: bool = False,
                 loss_type: str = 'mse',
                 guid_filter: OptConfigType = None,
                 loss_weight=1.0):
        super().__init__()
        assert pass_type in ['high', 'low']
        self.pass_type = pass_type
        if shape == 'cycle':
            self.center_mask = self._cycle_mask(radius)
        elif shape == 'square':
            self.center_mask = self._square_mask(radius)
        else:
            raise NotImplementedError('Only support `cycle` and `square`, '
                                      f'but got {shape}')

        assert loss_type in ['l1', 'mse']
        self.loss_type = loss_type
        self.channel_mean = channel_mean
        self.radius = radius
        self.loss_weight = loss_weight

        if guid_filter is not None:
            self.guid_filter = MODELS.build(guid_filter)
        else:
            self.guid_filter = None

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
        if self.pass_type == 'high':
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

            high_pass_target = self.get_pass_img(no_padding_target, mask)

            if self.guid_filter is not None:
                high_pass_target = self.guid_filter(
                    high_pass_target[None, ...], no_padding_target[None,
                                                                   ...])[0]
                high_pass_target = high_pass_target.clip_(min=1e-7, max=255)

            high_pass_pred = self.get_pass_img(no_padding_pred, mask)
            norm_high_pass_pred = high_pass_pred / 255
            norm_high_pass_target = high_pass_target / 255
            if self.loss_type == 'l1':
                loss = F.l1_loss(
                    norm_high_pass_pred,
                    norm_high_pass_target,
                    reduction='mean')
            else:
                loss = F.mse_loss(
                    norm_high_pass_pred,
                    norm_high_pass_target,
                    reduction='mean')
            losses.append(loss)
        total_loss = sum(_loss.mean() for _loss in losses)
        return total_loss * self.loss_weight

    @staticmethod
    def get_pass_img(img, mask):
        channel_img_list = []
        for i in range(img.size(0)):
            channel_img = img[i, ...]
            f = torch.fft.fft2(channel_img)
            fshift = torch.fft.fftshift(f)
            filter_fshift = fshift * mask

            ishift = torch.fft.ifftshift(filter_fshift)
            high_pass_img = torch.fft.ifft2(ishift)
            high_pass_img = torch.abs(high_pass_img).clip_(min=1e-7, max=255)
            channel_img_list.append(high_pass_img[None, ...])
        result_img = torch.cat(channel_img_list, dim=0)
        return result_img


@MODELS.register_module()
class GuidedFilter2d(nn.Module):

    def __init__(self,
                 radius: int = 30,
                 eps: float = 1e-4,
                 fast_s: Optional[int] = None,
                 channel_wise: bool = True):
        super().__init__()
        self.r = radius
        self.eps = eps
        self.fast_s = fast_s
        self.channel_wise = channel_wise

    def forward(self, x, guide):
        if guide.shape[1] == 3:
            if self.channel_wise:
                assert x.shape == guide.shape
                channel_result = []
                for i in range(3):
                    result = self.guidedfilter2d_gray(guide[:, i:i + 1, ...],
                                                      x[:, i:i + 1,
                                                        ...], self.r, self.eps,
                                                      self.fast_s)
                    channel_result.append(result)

                results = torch.cat(channel_result, dim=1)
                return results
            else:
                return self.guidedfilter2d_color(guide, x, self.r, self.eps,
                                                 self.fast_s)
        elif guide.shape[1] == 1:
            return self.guidedfilter2d_gray(guide, x, self.r, self.eps,
                                            self.fast_s)
        else:
            raise NotImplementedError

    def guidedfilter2d_color(self, guide, src, radius, eps, scale=None):
        """guided filter for a color guide image.

        Parameters
        -----
        guide: (B, 3, H, W)-dim torch.Tensor
            guide image
        src: (B, C, H, W)-dim torch.Tensor
            filtering image
        radius: int
            filter radius
        eps: float
            regularization coefficient
        """
        assert guide.shape[1] == 3
        if src.ndim == 3:
            src = src[:, None]
        if scale is not None:
            guide_sub = guide.clone()
            src = F.interpolate(src, scale_factor=1. / scale, mode='nearest')
            guide = F.interpolate(
                guide, scale_factor=1. / scale, mode='nearest')
            radius = radius // scale

        # b x 1 x H x W
        guide_r, guide_g, guide_b = torch.chunk(guide, 3, 1)
        ones = torch.ones_like(guide_r)
        N = self.boxfilter2d(ones, radius)

        # b x 3 x H x W
        mean_I = self.boxfilter2d(guide, radius) / N
        mean_p = self.boxfilter2d(src, radius) / N
        # b x 1 x H x W
        mean_I_r, mean_I_g, mean_I_b = torch.chunk(mean_I, 3, 1)

        # b x C x H x W
        mean_Ip_r = self.boxfilter2d(guide_r * src, radius) / N
        mean_Ip_g = self.boxfilter2d(guide_g * src, radius) / N
        mean_Ip_b = self.boxfilter2d(guide_b * src, radius) / N

        # b x C x H x W
        cov_Ip_r = mean_Ip_r - mean_I_r * mean_p
        cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
        cov_Ip_b = mean_Ip_b - mean_I_b * mean_p

        # b x 1 x H x W
        var_I_rr = self.boxfilter2d(guide_r * guide_r, radius) / N \
            - mean_I_r * mean_I_r + eps
        var_I_rg = self.boxfilter2d(guide_r * guide_g, radius) / N \
            - mean_I_r * mean_I_g
        var_I_rb = self.boxfilter2d(guide_r * guide_b, radius) / N \
            - mean_I_r * mean_I_b
        var_I_gg = self.boxfilter2d(guide_g * guide_g, radius) / N \
            - mean_I_g * mean_I_g + eps
        var_I_gb = self.boxfilter2d(guide_g * guide_b, radius) / N \
            - mean_I_g * mean_I_b
        var_I_bb = self.boxfilter2d(guide_b * guide_b, radius) / N \
            - mean_I_b * mean_I_b + eps

        # determinant, b x 1 x H x W
        cov_det = var_I_rr * var_I_gg * var_I_bb \
            + var_I_rg * var_I_gb * var_I_rb \
            + var_I_rb * var_I_rg * var_I_gb \
            - var_I_rb * var_I_gg * var_I_rb \
            - var_I_rg * var_I_rg * var_I_bb \
            - var_I_rr * var_I_gb * var_I_gb

        # inverse, b x 1 x H x W
        inv_var_I_rr = (var_I_gg * var_I_bb - var_I_gb * var_I_gb) / cov_det
        inv_var_I_rg = -(var_I_rg * var_I_bb - var_I_rb * var_I_gb) / cov_det
        inv_var_I_rb = (var_I_rg * var_I_gb - var_I_rb * var_I_gg) / cov_det
        inv_var_I_gg = (var_I_rr * var_I_bb - var_I_rb * var_I_rb) / cov_det
        inv_var_I_gb = -(var_I_rr * var_I_gb - var_I_rb * var_I_rg) / cov_det
        inv_var_I_bb = (var_I_rr * var_I_gg - var_I_rg * var_I_rg) / cov_det

        # b x 3 x 3 x H x W
        inv_sigma = torch.stack([
            torch.stack([inv_var_I_rr, inv_var_I_rg, inv_var_I_rb], 1),
            torch.stack([inv_var_I_rg, inv_var_I_gg, inv_var_I_gb], 1),
            torch.stack([inv_var_I_rb, inv_var_I_gb, inv_var_I_bb], 1)
        ], 1).squeeze(-3)

        # b x 3 x C x H x W
        cov_Ip = torch.stack([cov_Ip_r, cov_Ip_g, cov_Ip_b], 1)

        a = torch.einsum('bichw,bijhw->bjchw', (cov_Ip, inv_sigma))
        # b x C x H x W
        b = mean_p - a[:, 0] * mean_I_r - \
            a[:, 1] * mean_I_g - \
            a[:, 2] * mean_I_b

        mean_a = torch.stack(
            [self.boxfilter2d(a[:, i], radius) / N for i in range(3)], 1)
        mean_b = self.boxfilter2d(b, radius) / N

        if scale is not None:
            guide = guide_sub
            mean_a = torch.stack([
                F.interpolate(mean_a[:, i], guide.shape[-2:], mode='bilinear')
                for i in range(3)
            ], 1)
            mean_b = F.interpolate(mean_b, guide.shape[-2:], mode='bilinear')

        q = torch.einsum('bichw,bihw->bchw', (mean_a, guide)) + mean_b

        return q

    def guidedfilter2d_gray(self, guide, src, radius, eps, scale=None):
        """guided filter for a gray scale guide image.

        Parameters
        -----
        guide: (B, 1, H, W)-dim torch.Tensor
            guide image
        src: (B, C, H, W)-dim torch.Tensor
            filtering image
        radius: int
            filter radius
        eps: float
            regularization coefficient
        """
        if guide.ndim == 3:
            guide = guide[:, None]
        if src.ndim == 3:
            src = src[:, None]

        if scale is not None:
            guide_sub = guide.clone()
            src = F.interpolate(src, scale_factor=1. / scale, mode='nearest')
            guide = F.interpolate(
                guide, scale_factor=1. / scale, mode='nearest')
            radius = radius // scale

        ones = torch.ones_like(guide)
        N = self.boxfilter2d(ones, radius)

        mean_I = self.boxfilter2d(guide, radius) / N
        mean_p = self.boxfilter2d(src, radius) / N
        mean_Ip = self.boxfilter2d(guide * src, radius) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.boxfilter2d(guide * guide, radius) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = self.boxfilter2d(a, radius) / N
        mean_b = self.boxfilter2d(b, radius) / N

        if scale is not None:
            guide = guide_sub
            mean_a = F.interpolate(mean_a, guide.shape[-2:], mode='bilinear')
            mean_b = F.interpolate(mean_b, guide.shape[-2:], mode='bilinear')

        q = mean_a * guide + mean_b
        return q

    def boxfilter2d(self, src, radius):
        return self._diff_y(self._diff_x(src, radius), radius)

    @staticmethod
    def _diff_x(src, r):
        cum_src = src.cumsum(-2)

        left = cum_src[..., r:2 * r + 1, :]
        middle = cum_src[..., 2 * r + 1:, :] - \
            cum_src[..., :-2 * r - 1, :]
        right = cum_src[..., -1:, :] - \
            cum_src[..., -2 * r - 1:-r - 1, :]

        output = torch.cat([left, middle, right], -2)
        return output

    @staticmethod
    def _diff_y(src, r):
        cum_src = src.cumsum(-1)

        left = cum_src[..., r:2 * r + 1]
        middle = cum_src[..., 2 * r + 1:] - \
            cum_src[..., :-2 * r - 1]
        right = cum_src[..., -1:] - \
            cum_src[..., -2 * r - 1:-r - 1]

        output = torch.cat([left, middle, right], -1)
        return output


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
            f = torch.fft.fft2(channel_img, norm='ortho')
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

    def fft_domain_loss(self, pred, target, *args, **kwargs):
        pred_real, pred_imaginary = self.get_real(pred)
        gt_real, gt_imaginary = self.get_real(target)

        pred_amp = torch.sqrt(pred_real**2 + pred_imaginary**2 + 1e-10)
        pred_pha = torch.atan2(pred_imaginary, pred_real)

        gt_amp = torch.sqrt(gt_real**2 + gt_imaginary**2 + 1e-10)
        gt_pha = torch.atan2(gt_imaginary, gt_real)

        loss_amp = F.l1_loss(pred_amp, gt_amp)
        loss_pha = F.l1_loss(pred_pha, gt_pha)

        loss = loss_amp + loss_pha
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
