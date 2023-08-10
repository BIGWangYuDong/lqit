_base_ = [
    '../base_tienet_enhancer.py',
    '../base_detector/fcos_r50-caffe_fpn_gn-head_1x_urpc.py'
]

model = dict(
    _delete_=True,
    type='lqit.SelfEnhanceDetector',
    detector={{_base_.model}},
    enhance_model={{_base_.enhance_model}})

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.01),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))  # loss may NaN without clip_grad

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='lqit.SetInputImageAsGT'),
    dict(type='lqit.PackInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

model_wrapper_cfg = dict(
    type='lqit.SelfEnhanceModelDDP',
    broadcast_buffers=False,
    find_unused_parameters=False)

# default_hooks = dict(
#     visualization=dict(
#         type='EnhanceDetVisualizationHook',
#         draw=True,
#         show_on_enhance=True,
#         test_out_dir='tmp'))
