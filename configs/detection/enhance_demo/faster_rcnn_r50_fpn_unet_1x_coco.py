# TODO: delete after fully support
_base_ = 'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

model = dict(
    type='TwoStageWithEnhanceModel',
    backbone=dict(norm_eval=False),
    enhance_model=dict(
        type='lqit.BaseEditModel',
        gt_preprocessor=dict(
            type='lqit.GTPixelPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32,
            element_name='img'),
        generator=dict(
            _scope_='lqit',
            type='UNet',
            unet=dict(type='BaseUNet'),
            pixel_loss=dict(type='L1Loss', loss_weight=1.0),
            perceptual_loss=dict(
                type='PerceptualLoss',
                layer_weights={'21': 1.},
                perceptual_weight=1.,
                style_weight=0))))
# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='lqit.SetInputImageAsGT'),
    dict(type='lqit.PackInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False)
test_dataloader = val_dataloader

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
