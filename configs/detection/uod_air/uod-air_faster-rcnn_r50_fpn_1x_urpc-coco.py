_base_ = [
    './base_detector/faster-rcnn_r50_fpn_1x_urpc-coco.py',
    './base_ehance_head/enhance_head.py'
]

# model settings
model = dict(
    _delete_=True,
    type='lqit.DetectorWithEnhanceHead',
    detector={{_base_.model}},
    enhance_head={{_base_.enhance_head}},
    vis_enhance=False)

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='lqit.SetInputImageAsGT'),
    dict(type='lqit.PackInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
