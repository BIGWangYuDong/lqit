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
        type='TIENetGenerator',
        model=dict(
            type='TIENetEnhanceModel',
            in_channels=3,
            feat_channels=64,
            out_channels=3,
            num_blocks=3,
            expand_ratio=0.5,
            kernel_size=[1, 3, 5],
            output_weight=[1.0, 1.0],
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU'),
            use_depthwise=True),
        spacial_pred='structure',
        structure_pred='structure',
        spacial_loss=dict(type='SpatialLoss', loss_weight=1.0),
        tv_loss=dict(type='MaskedTVLoss', loss_mode='mse', loss_weight=10.0),
        structure_loss=dict(
            type='StructureFFTLoss',
            radius=4,
            pass_type='high',
            channel_mean=True,
            loss_type='mse',
            guid_filter=dict(
                type='GuidedFilter2d', radius=32, eps=1e-4, fast_s=2),
            loss_weight=0.1)))
