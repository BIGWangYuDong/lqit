_base_ = '../base_detector/faster_rcnn_r50_fpn_1x_rtts.py'

# model settings
model = dict(
    type='CycleTwoStageWithEnhanceHead',
    loss_weight=[1.5, 0.5],
    neck=dict(type='UFPN'),
    enhance_head=dict(
        _scope_='lqit',
        type='AENetEnhanceHead',
        in_channels=256,
        upscale_factor=4,
        num_convs=2,
        gt_preprocessor=dict(
            type='GTPixelPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32,
            element_name='img'),
        spacial_loss=dict(type='SpatialLoss', loss_weight=1.0),
        structure_loss=dict(
            type='StructureFFTLoss',
            radius=4,
            pass_type='high',
            channel_mean=True,
            loss_type='l1',
            guid_filter=None,
            loss_weight=0.1),
        enhance_loss=dict(type='L1Loss', loss_weight=1.0),
    ))

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
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#     dict(
#         type='FFTFilter',
#         pass_type='soft',
#         radius=[32, 256],
#         get_gt=True,
#         w_high=[0.5, 1.5],
#         w_low=[0.5, 1.5]),
#     # If you don't have a gt annotation, delete the pipeline
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor', 'img'))
# ]
# val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
# test_dataloader = val_dataloader
# show_dir = 'work_dirs/a_aedet_vis/rtts/faster'
#
# default_hooks = dict(
#     visualization=dict(
#         type='EnhanceDetVisualizationHook',
#         draw=True,
#         test_out_dir=show_dir,
#         show_on_enhance=False,
#         draw_gt=False,
#         draw_pred=True))
