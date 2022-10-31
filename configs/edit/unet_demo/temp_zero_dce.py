# TODO: delete after fully support editor metric and datasets.
_base_ = 'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

model = dict(
    _delete_=True,
    type='lqit.BaseEditModel',
    data_preprocessor=dict(
        type='lqit.EditDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        gt_name='img'),
    generator=dict(
        _scope_='lqit',
        type='ZeroDCEGenerator',
        zero_dce=dict(type='ZeroDCE'),
        color_loss=dict(type='ColorLoss', loss_weight=5.0),
        spacial_loss=dict(type='SpatialLoss', loss_weight=1.0),
        tv_loss=dict(type='MaskedTVLoss', loss_mode='mse', loss_weight=200.0),
        exposure_loss=dict(
            type='ExposureLoss', patch_size=16, mean_val=0.6,
            loss_weight=10.0)))
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
