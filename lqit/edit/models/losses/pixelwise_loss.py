# Modified from https://github.com/open-mmlab/mmediting/tree/1.x/
import torch
import torch.nn as nn
import torch.nn.functional as F

from lqit.registry import MODELS
from .utils import masked_loss

_reduction_modes = ['none', 'mean', 'sum']


@masked_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated L1 loss.
    """
    return F.l1_loss(pred, target, reduction='none')


@masked_loss
def mse_loss(pred, target):
    """MSE loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated MSE loss.
    """
    return F.mse_loss(pred, target, reduction='none')


@masked_loss
def charbonnier_loss(pred, target, eps=1e-12):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    return torch.sqrt((pred - target)**2 + eps)


@MODELS.register_module()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduce loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@MODELS.register_module()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred,
            target,
            weight,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@MODELS.register_module()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable variant
    of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 sample_wise=False,
                 eps=1e-12):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """Forward Function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=self.reduction,
            sample_wise=self.sample_wise)


@MODELS.register_module()
class MaskedTVLoss(nn.Module):
    """Masked TV loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        sample_wise (bool): Whether calculate the loss sample-wise. This
            argument only takes effect when `reduction` is 'mean' and `weight`
            (argument of `forward()`) is not None. It will first reduces loss
            with 'mean' per-sample, and then it means over all the samples.
            Default: False.
    """

    def __init__(self,
                 loss_weight=1.0,
                 reduction='mean',
                 sample_wise=False,
                 loss_mode='l1'):
        super().__init__()
        if loss_mode not in ['l1', 'mse']:
            raise ValueError(f'Unsupported loss mode: {loss_mode}. '
                             f'Supported ones are: {["l1", "mse"]}')
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_mode = loss_mode
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise

    def forward(self, pred, *args, mask=None, **kwargs):
        """Forward function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            mask (torch.Tensor, optional): of shape (N, 1, H, W) or
                (N, C, H, W). Defaults to None.

        Returns:
            Tensor: Calculated loss.
        """
        if self.loss_mode == 'l1':
            loss_func = l1_loss
        else:
            loss_func = mse_loss

        if mask is None:
            mask = torch.ones_like(pred)

        y_diff = loss_func(
            pred[:, :, :-1, :],
            pred[:, :, 1:, :],
            weight=mask[:, :, :-1, :],
            reduction=self.reduction,
            sample_wise=self.sample_wise)
        x_diff = loss_func(
            pred[:, :, :, :-1],
            pred[:, :, :, 1:],
            weight=mask[:, :, :, :-1],
            reduction=self.reduction,
            sample_wise=self.sample_wise)

        loss = x_diff + y_diff

        return loss


@MODELS.register_module()
class SpatialLoss(nn.Module):
    """Spatial consistency loss.

    Modified from
    https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py

    Args:
        loss_weight (float): The weight of loss.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0],
                                         [0, 0, 0]])[None, None, ...]
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1],
                                          [0, 0, 0]])[None, None, ...]
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0],
                                       [0, 0, 0]])[None, None, ...]
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0],
                                         [0, -1, 0]])[None, None, ...]

        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(
            data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, pred, target, *args, **kwargs):
        """Forward function.

        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.

        Returns:
            Tensor: Calculated loss.
        """
        device = target.device
        weight_up = self.weight_up.to(device)
        weight_down = self.weight_down.to(device)
        weight_left = self.weight_left.to(device)
        weight_right = self.weight_right.to(device)

        target_mean = torch.mean(target, 1, keepdim=True)
        pred_mean = torch.mean(pred, 1, keepdim=True)

        target_pool = self.pool(target_mean)
        pred_pool = self.pool(pred_mean)

        target_left = F.conv2d(target_pool, weight_left, padding=1)
        target_right = F.conv2d(target_pool, weight_right, padding=1)
        target_up = F.conv2d(target_pool, weight_up, padding=1)
        target_down = F.conv2d(target_pool, weight_down, padding=1)

        pred_left = F.conv2d(pred_pool, weight_left, padding=1)
        pred_right = F.conv2d(pred_pool, weight_right, padding=1)
        pred_up = F.conv2d(pred_pool, weight_up, padding=1)
        pred_down = F.conv2d(pred_pool, weight_down, padding=1)

        left = torch.pow(target_left - pred_left, 2)
        right = torch.pow(target_right - pred_right, 2)
        up = torch.pow(target_up - pred_up, 2)
        down = torch.pow(target_down - pred_down, 2)
        loss = torch.mean(left + right + up + down)

        return loss * self.loss_weight


@MODELS.register_module()
class ColorLoss(nn.Module):
    """Color constancy loss.

    Modified from
    https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py

    Args:
        loss_weight (float): The weight of loss.
    """

    def __init__(self, loss_mode='l2', loss_weight=1.0):
        super().__init__()
        assert loss_mode in ['l1', 'l2']
        self.loss_mode = loss_mode
        self.loss_weight = loss_weight

    def forward(self, pred, *args, **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            Tensor: Calculated loss.
        """
        assert pred.dim() == 4
        mean_rgb = torch.mean(pred, [2, 3], keepdim=True)

        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)

        distance_rg = torch.pow(mr - mg, 2)
        distance_rb = torch.pow(mr - mb, 2)
        distance_gb = torch.pow(mb - mg, 2)
        if self.loss_mode == 'l2':
            loss = torch.sqrt(
                torch.pow(distance_rg, 2) + torch.pow(distance_rb, 2) +
                torch.pow(distance_gb, 2) + 1e-6)
        else:
            loss = distance_rg + distance_rb + distance_gb
        loss = loss.mean()
        return loss * self.loss_weight


@MODELS.register_module()
class ExposureLoss(nn.Module):
    """Exposure control loss.

    Modified from
    https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py

    Args:
        loss_weight (float): The weight of loss.
    """

    def __init__(self, patch_size=16, mean_val=0.6, loss_weight=1.0):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        assert isinstance(mean_val, float)
        self.mean_val = mean_val
        self.loss_weight = loss_weight

    def forward(self, pred, *args, **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            Tensor: Calculated loss.
        """
        pred_mean = torch.mean(pred, 1, keepdim=True)

        pred_mean = self.pool(pred_mean)

        loss = ((pred_mean - self.mean_val)**2).mean()
        return loss * self.loss_weight
