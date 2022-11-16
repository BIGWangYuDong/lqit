_base_ = [
    '../base_editor/self_enhance_base_loss.py',
    '../base_detector/retinanet_r50_fpn_1x_rtts.py'
]

model = dict(
    _delete_=True,
    type='SelfEnhanceDetector',
    detector={{_base_.model}},
    enhance_model={{_base_.enhance_model}})

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

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

model_wrapper_cfg = dict(
    type='SelfEnhanceModelDDP',
    broadcast_buffers=False,
    find_unused_parameters=False)
