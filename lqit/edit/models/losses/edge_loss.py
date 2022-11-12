import torch
import torch.nn as nn

from lqit.registry import MODELS

# import torch.nn.functional as F

_reduction_modes = ['none', 'mean', 'sum']


@MODELS.register_module()
class EdgeeLoss(nn.Module):
    """Edge loss.

    Args:
        loss_weight (float): The weight of loss.
    """

    def __init__(self,
                 method='sobel',
                 threshold_value: int = 127,
                 loss_weight=1.0):
        super().__init__()
        # TODO: Add more method to get edge
        assert method in ['sobel']
        self.threshold_value = threshold_value
        self.loss_weight = loss_weight
        gaussian_kernel = torch.FloatTensor([[-1, -1, -1], [-1, 8, -1],
                                             [-1, 8, -1]])[None, None, ...]
        self.gaussian_kernel = nn.Parameter(
            data=gaussian_kernel, requires_grad=False)

    def forward(self, pred, target, *args, **kwargs):
        """Forward function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.

        Returns:
            Tensor: Calculated loss.
        """
        raise NotImplementedError
        # device = target.device
        # gaussian_kernel = self.gaussian_kernel.to(device)
        #
        # pred_edge = F.conv2d(pred, gaussian_kernel, padding=1)
        # target_edge = F.conv2d(target, target, padding=1)
        #
        # get_edge_func = getattr(self, self.method)
        # pred_mean = torch.mean(pred, 1, keepdim=True)
        #
        # # TODO not implement yet
        # return loss * self.loss_weight


@MODELS.register_module()
class HighPassFFTLoss(nn.Module):

    def __init__(self,
                 radius: int = 32,
                 shape: str = 'cycle',
                 loss_weight=1.0):
        super().__init__()
        if shape == 'cycle':
            self.center_mask = self._cycle_mask(radius)
        elif shape == 'square':
            self.center_mask = self._square_mask(radius)
        else:
            raise NotImplementedError('Only support `cycle` and `square`, '
                                      f'but got {shape}')
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

    def _get_total_mask(self, pred):
        device = pred.device
        self.center_mask = self.center_mask.to(device)

        mask = torch.zeros_like(pred, dtype=torch.bool)
        height, width = mask.shape[0], mask.shape[1]
        x_c, y_c = width // 2, height // 2

        mask[..., y_c - self.radius:y_c + self.radius,
             x_c - self.radius:x_c + self.radius] = self.center_mask
        return mask

    def forward(self, pred, target, *args, **kwargs):
        """Forward function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.

        Returns:
            Tensor: Calculated loss.
        """
        assert pred.shape == target.shape
        # mask = self._get_total_mask(pred)

    # def fft(self, img):
    #     # Fourier transform
    #     f = torch.fft.fft2(channel_img, norm='ortho')
    #     # Shift the spectrum to the central location
    #     fshift = torch.fft.fftshift(f)
    #     return fshift
    #
    # def ifft(self, fft_img):
    #     iff
