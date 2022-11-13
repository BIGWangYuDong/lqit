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
        type='SelfEnhanceUNetGenerator',
        model=dict(
            type='BaseUNet',
            in_channels=3,
            out_channels=3,
            base_channels=64,
            num_stages=4,
            strides=(1, 1, 1, 1),
            enc_num_convs=(2, 2, 2, 2),
            dec_num_convs=(2, 2, 2),
            downsamples=(True, True, True),
            enc_dilations=(1, 1, 1, 1),
            dec_dilations=(1, 1, 1),
            with_cp=False,
            conv_cfg=None,
            norm_cfg=dict(type='BN', requires_grad=False),
            act_cfg=dict(type='LeakyReLU'),
            upsample_cfg=dict(type='InterpConv'),
            norm_eval=False),
        spacial_loss=dict(type='SpatialLoss', loss_weight=1.0),
        tv_loss=dict(type='MaskedTVLoss', loss_mode='mse', loss_weight=1.0),
        structure_loss=dict(type='StructureFFTLoss', loss_weight=1.0)))
