import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from lqit.registry import MODELS


def gaussian(window_size=11, sigma=1.5):
    x = torch.arange(window_size)
    gauss = (-(x - window_size // 2)**2 / (2 * sigma**2)).exp_()
    return gauss / gauss.sum()


def create_window(window_size=11, channel=3, sigma=1.5):
    _1D_window = gaussian(window_size=window_size, sigma=sigma)[:, None]
    _2D_window = _1D_window.mm(_1D_window.t())[None, None, ...]
    window = _2D_window.expand(channel, 1, window_size,
                               window_size).contiguous()
    window = nn.Parameter(data=window, requires_grad=False)
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(
        img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(
        img1 * img2, window, padding=window_size // 2,
        groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


def ssim_loss(pred, target, window, window_size, channel, size_average):
    loss = 1 - _ssim(
        pred,
        target,
        window=window,
        window_size=window_size,
        channel=channel,
        size_average=size_average)
    return loss


@MODELS.register_module()
class SSIMLoss(nn.Module):

    def __init__(self,
                 window_size=11,
                 sigma=1.5,
                 channel=3,
                 size_average=True,
                 loss_weight=1.0):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.sigma = sigma
        self.channel = channel
        self.window = create_window(
            window_size=self.window_size,
            channel=self.channel,
            sigma=self.sigma)
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        assert pred.size() == target.size()

        img_channel = pred.size(1)
        if img_channel != self.channel:
            warnings.warn(f'SSIMLoss.channel ({self.channel}) is not equal '
                          f'to input image channel: {img_channel}. '
                          'Recreate a new window based on input image channel')
            self.channel = img_channel
            self.window = create_window(
                window_size=self.window_size,
                channel=self.channel,
                sigma=self.sigma)

        device = pred.device
        window = self.window.to(device)

        loss = self.loss_weight * ssim_loss(
            pred=pred,
            target=target,
            window=window,
            window_size=self.window_size,
            channel=self.channel,
            size_average=self.size_average)
        return loss
