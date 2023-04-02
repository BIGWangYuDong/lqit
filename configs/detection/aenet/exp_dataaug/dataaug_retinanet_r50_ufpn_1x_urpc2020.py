_base_ = '../base_detector/retinanet_r50_fpn_1x_urpc2020.py'

model = dict(neck=dict(type='UFPN'))

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='FFTFilter',
        pass_type='soft',
        radius=[32, 256],
        get_gt=True,
        w_high=[0.8, 1.2],
        w_low=[0.8, 1.2]),
    dict(type='lqit.PackInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

optim_wrapper = dict(clip_grad=dict(max_norm=35, norm_type=2))
