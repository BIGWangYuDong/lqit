enhance_model = dict(
    _scope_='lqit',
    type='BaseEditModel',
    destruct_gt=True,
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        gt_name='img'),
    generator=dict(
        type='SelfEnhanceGenerator',
        model=dict(type='SelfEnhance'),
        spacial_loss=dict(type='SpatialLoss', loss_weight=1.0),
        tv_loss=dict(type='MaskedTVLoss', loss_mode='mse', loss_weight=10.0),
        structure_loss=dict(type='StructureFFTLoss', loss_weight=1.0),
        perceptual_loss=dict(
            type='PerceptualLoss',
            vgg_type='vgg16',
            pretrained='torchvision://vgg16',
            layer_weights={'21': 1.},
            perceptual_weight=1.0,
            style_weight=0)))
