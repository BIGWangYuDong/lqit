_base_ = 'retinanet_r50_fpn_1x_urpc2020.py'

model = dict(
    type='SingleStageWithEnhanceModel',
    backbone=dict(norm_eval=False),
    loss_weight=[0.8, 0.2],
    enhance_model=dict(
        _scope_='lqit',
        type='lqit.BaseEditModel',
        destruct_gt=True,
        gt_preprocessor=dict(
            type='lqit.GTPixelPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32,
            element_name='img'),
        generator=dict(
            type='SelfEnhanceGenerator',
            model=dict(type='SelfEnhance'),
            spacial_loss=dict(type='SpatialLoss', loss_weight=1.0),
            tv_loss=dict(
                type='MaskedTVLoss', loss_mode='mse', loss_weight=10.0),
            perceptual_loss=dict(
                type='PerceptualLoss',
                layer_weights={'21': 1.},
                perceptual_weight=1.0,
                style_weight=0))))

# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='lqit.SetInputImageAsGT'),
    dict(type='lqit.PackInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
