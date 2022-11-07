_base_ = '../edffnet/atss_r50_fpn_1x.py'

model = dict(
    type='EDFFNet',
    backbone=dict(norm_eval=True),
    neck=dict(
        type='DFFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        shape_level=2,
        num_outs=5),
    enhance_head=dict(
        type='lqit.EdgeHead',
        in_channels=256,
        feat_channels=256,
        num_convs=5,
        loss_enhance=dict(type='mmdet.L1Loss', loss_weight=0.7),
        gt_preprocessor=dict(
            type='lqit.GTPixelPreprocessor',
            mean=[128],
            std=[57.12],
            pad_size_divisor=32,
            element_name='edge')))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='lqit.GetEdgeGTFromImage', method='scharr'),
    dict(
        type='lqit.TransBroadcaster',
        src_key='img',
        dst_key='gt_edge',
        transforms=[
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5)
        ]),
    dict(type='lqit.PackInputs', )
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001))
