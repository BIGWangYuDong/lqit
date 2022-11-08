# Modified from https://github.com/open-mmlab/mmediting/tree/1.x/
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.logging import MMLogger
from mmengine.runner import load_checkpoint

from lqit.registry import MODELS


def init_weights(module, init_type='normal', init_gain=0.02):
    """Default initialization of network weights for image generation.

    By default, we use normal init, but xavier and kaiming might work
    better for some applications.

    Args:
        module (nn.Module): Module to be initialized.
        init_type (str): The name of an initialization method:
            normal | xavier | kaiming | orthogonal. Defaults to 'normal'.
        init_gain (float): Scaling factor for normal, xavier and
            orthogonal. Defaults to 0.02.
    """
    from mmengine.model.weight_init import (kaiming_init, normal_init,
                                            xavier_init)
    from torch.nn import init

    def init_func(m):
        """Initialization function.

        Args:
            m (nn.Module): Module to be initialized.
        """
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                normal_init(m, 0.0, init_gain)
            elif init_type == 'xavier':
                xavier_init(m, gain=init_gain, distribution='normal')
            elif init_type == 'kaiming':
                kaiming_init(
                    m,
                    a=0,
                    mode='fan_in',
                    nonlinearity='leaky_relu',
                    distribution='normal')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_gain)
                init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError(
                    f"Initialization method '{init_type}' is not implemented")
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix;
            # only normal distribution applies.
            normal_init(m, 1.0, init_gain)

    module.apply(init_func)


class UnetSkipConnectionBlock(nn.Module):
    """Construct a Unet submodule with skip connections, with the following.

    structure: downsampling - `submodule` - upsampling.

    Args:
        outer_channels (int): Number of channels at the outer conv layer.
        inner_channels (int): Number of channels at the inner conv layer.
        in_channels (int): Number of channels in input images/features. If is
            None, equals to `outer_channels`. Defaults to None.
        submodule (UnetSkipConnectionBlock): Previously constructed submodule.
            Defaults to None.
        is_outermost (bool): Whether this module is the outermost module.
            Defaults to False.
        is_innermost (bool): Whether this module is the innermost module.
            Defaults to False.
        norm_cfg (dict): Config dict to build norm layer. Defaults to
            `dict(type='BN')`.
        use_dropout (bool): Whether to use dropout layers. Defaults to False.
    """

    def __init__(self,
                 outer_channels,
                 inner_channels,
                 in_channels=None,
                 submodule=None,
                 is_outermost=False,
                 is_innermost=False,
                 norm_cfg=dict(type='BN'),
                 use_dropout=False):
        super().__init__()
        # cannot be both outermost and innermost
        assert not (is_outermost and is_innermost), (
            "'is_outermost' and 'is_innermost' cannot be True"
            'at the same time.')
        self.is_outermost = is_outermost
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"
        # We use norm layers in the unet skip connection block.
        # Only for IN, use bias since it does not have affine parameters.
        use_bias = norm_cfg['type'] == 'IN'

        kernel_size = 4
        stride = 2
        padding = 1
        if in_channels is None:
            in_channels = outer_channels
        down_conv_cfg = dict(type='Conv2d')
        down_norm_cfg = norm_cfg
        down_act_cfg = dict(type='LeakyReLU', negative_slope=0.2)
        up_conv_cfg = dict(type='Deconv')
        up_norm_cfg = norm_cfg
        up_act_cfg = dict(type='ReLU')
        up_in_channels = inner_channels * 2
        up_bias = use_bias
        middle = [submodule]
        upper = []

        if is_outermost:
            down_act_cfg = None
            down_norm_cfg = None
            up_bias = True
            up_norm_cfg = None
            upper = [nn.Tanh()]
        elif is_innermost:
            down_norm_cfg = None
            up_in_channels = inner_channels
            middle = []
        else:
            upper = [nn.Dropout(0.5)] if use_dropout else []

        down = [
            ConvModule(
                in_channels=in_channels,
                out_channels=inner_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=use_bias,
                conv_cfg=down_conv_cfg,
                norm_cfg=down_norm_cfg,
                act_cfg=down_act_cfg,
                order=('act', 'conv', 'norm'))
        ]
        up = [
            ConvModule(
                in_channels=up_in_channels,
                out_channels=outer_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=up_bias,
                conv_cfg=up_conv_cfg,
                norm_cfg=up_norm_cfg,
                act_cfg=up_act_cfg,
                order=('act', 'conv', 'norm'))
        ]

        model = down + middle + up + upper

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        if self.is_outermost:
            return self.model(x)

        # add skip connections
        return torch.cat([x, self.model(x)], 1)


@MODELS.register_module()
class Pix2PixUnet(nn.Module):
    """Construct the Unet-based generator from the innermost layer to the
    outermost layer, which is a recursive process.

    Args:
        in_channels (int): Number of channels in input images.
        out_channels (int): Number of channels in output images.
        num_down (int): Number of downsamplings in Unet. If `num_down` is 8,
            the image with size 256x256 will become 1x1 at the bottleneck.
            Defaults to 8.
        base_channels (int): Number of channels at the last conv layer.
            Defaults to 64.
        norm_cfg (dict): Config dict to build norm layer. Defaults to
            `dict(type='BN')`.
        use_dropout (bool): Whether to use dropout layers. Defaults to False.
        init_cfg (dict): Config dict for initialization.
            `type`: The name of our initialization method.
            Defaults to 'normal'.
            `gain`: Scaling factor for normal, xavier and orthogonal.
            Defaults to 0.02.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_down=8,
                 base_channels=64,
                 norm_cfg=dict(type='BN'),
                 use_dropout=False,
                 init_cfg=dict(type='normal', gain=0.02)):
        super().__init__()
        # We use norm layers in the unet generator.
        assert isinstance(norm_cfg, dict), ("'norm_cfg' should be dict, but"
                                            f'got {type(norm_cfg)}')
        assert 'type' in norm_cfg, "'norm_cfg' must have key 'type'"

        # add the innermost layer
        unet_block = UnetSkipConnectionBlock(
            base_channels * 8,
            base_channels * 8,
            in_channels=None,
            submodule=None,
            norm_cfg=norm_cfg,
            is_innermost=True)
        # add intermediate layers with base_channels * 8 filters
        for _ in range(num_down - 5):
            unet_block = UnetSkipConnectionBlock(
                base_channels * 8,
                base_channels * 8,
                in_channels=None,
                submodule=unet_block,
                norm_cfg=norm_cfg,
                use_dropout=use_dropout)
        # gradually reduce the number of filters
        # from base_channels * 8 to base_channels
        unet_block = UnetSkipConnectionBlock(
            base_channels * 4,
            base_channels * 8,
            in_channels=None,
            submodule=unet_block,
            norm_cfg=norm_cfg)
        unet_block = UnetSkipConnectionBlock(
            base_channels * 2,
            base_channels * 4,
            in_channels=None,
            submodule=unet_block,
            norm_cfg=norm_cfg)
        unet_block = UnetSkipConnectionBlock(
            base_channels,
            base_channels * 2,
            in_channels=None,
            submodule=unet_block,
            norm_cfg=norm_cfg)
        # add the outermost layer
        self.model = UnetSkipConnectionBlock(
            out_channels,
            base_channels,
            in_channels=in_channels,
            submodule=unet_block,
            is_outermost=True,
            norm_cfg=norm_cfg)

        self.init_type = 'normal' if init_cfg is None else init_cfg.get(
            'type', 'normal')
        self.init_gain = 0.02 if init_cfg is None else init_cfg.get(
            'gain', 0.02)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        return self.model(x)

    def init_weights(self, pretrained=None, strict=True):
        """Initialize weights for the model.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (bool, optional): Whether to allow different params for the
                model and checkpoint. Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            init_weights(
                self, init_type=self.init_type, init_gain=self.init_gain)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')
