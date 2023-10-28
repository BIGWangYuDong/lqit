_base_ = ['./atss_r50_dffpn_1x_rtts-coco_lr002.py']

model = dict(
    _delete_=True,
    type='lqit.EDFFNet',
    detector={{_base_.model}},
    edge_head=dict(
        _scope_='lqit',
        type='EdgeHead',
        in_channels=256,
        feat_channels=256,
        num_convs=5,
        loss_enhance=dict(type='L1Loss', loss_weight=0.7),
        gt_preprocessor=dict(
            type='GTPixelPreprocessor',
            mean=[128],
            std=[57.12],
            pad_size_divisor=32,
            element_name='edge')),
    vis_enhance=False)

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
