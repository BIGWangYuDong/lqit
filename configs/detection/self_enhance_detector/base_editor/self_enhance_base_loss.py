enhance_model = dict(
    _scope_='lqit',
    type='BaseEditModel',
    destruct_gt=True,
    data_preprocessor=dict(
        type='EditDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=False,
        gt_name='img'),
    generator=dict(
        type='SelfEnhanceGenerator',
        model=dict(
            type='SelfEnhanceLight',
            in_channels=3,
            feat_channels=64,
            out_channels=3,
            num_blocks=4,
            expand_ratio=0.5,  # TODO: try 0.5 if oom
            kernel_size=[1, 3, 5, 7],
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU'),
            use_depthwise=True),
        spacial_loss=dict(type='SpatialLoss', loss_weight=1.0),
        tv_loss=dict(type='MaskedTVLoss', loss_mode='mse', loss_weight=10.0)))
