# default scope is mmdet
_base_ = [
    './base_editor/tienet_enhance_model.py',
    './base_detector/tood_r50_fpn_1x_rtts-coco.py'
]

model = dict(
    _delete_=True,
    type='lqit.DetectorWithEnhanceModel',
    detector={{_base_.model}},
    enhance_model={{_base_.enhance_model}},
    train_mode='enhance',
    pred_mode='enhance',
    detach_enhance_img=False)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

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

model_wrapper_cfg = dict(
    type='lqit.SelfEnhanceModelDDP',
    broadcast_buffers=False,
    find_unused_parameters=False)
