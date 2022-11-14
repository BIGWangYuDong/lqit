_base_ = '.self_enhance_base_loss.py'

enhance_model = dict(
    generator=dict(
        spacial_loss=dict(type='SpatialLoss', loss_weight=1.0),
        tv_loss=dict(type='MaskedTVLoss', loss_mode='mse', loss_weight=10.0),
        perceptual_loss=dict(
            type='PerceptualLoss',
            vgg_type='vgg16',
            pretrained='torchvision://vgg16',
            layer_weights={'21': 1.},
            perceptual_weight=1.0,
            style_weight=0)))
