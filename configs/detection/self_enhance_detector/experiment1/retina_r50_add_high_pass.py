_base_ = ['../base_detector/retinanet_r50_fpn_1x_urpc2020.py']

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='lqit.AddHighPassImg', radius=16),
    dict(type='lqit.PackInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
