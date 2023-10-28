enhance_head = dict(
    _scope_='lqit',
    type='BasicEnhanceHead',
    in_channels=256,
    feat_channels=256,
    num_convs=5,
    loss_enhance=dict(type='L1Loss', loss_weight=0.5),
    gt_preprocessor=dict(
        type='GTPixelPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        element_name='img'))
