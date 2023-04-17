_base_ = '../edffnet/atss_r50_fpn_1x.py'

model = dict(
    type='EDFFNet',
    backbone=dict(norm_eval=False),
    neck=dict(
        type='DFFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
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
            mean=[123.675],
            std=[58.395],
            pad_size_divisor=32,
            element_name='edge')),
)

# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='lqit.GetEdgeGTFromImage', method='scharr'),
    dict(type='lqit.PackInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
