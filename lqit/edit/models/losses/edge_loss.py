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
