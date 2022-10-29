_base_ = 'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

model = dict(
    type='TwoStageWithEnhanceHead',
    backbone=dict(norm_eval=False),
    enhance_head=dict(
        type='lqit.BasicEnhanceHead',
        in_channels=256,
        feat_channels=256,
        num_convs=5,
        loss_enhance=dict(type='mmdet.L1Loss', loss_weight=0.1),
        gt_preprocessor=dict(
            type='lqit.GTPixelPreprocessor',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            bgr_to_rgb=True,
            pad_size_divisor=32,
            element_name='img')),
)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(
        type='lqit.LoadGTImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='lqit.TransBroadcaster',
        src_key='img',
        dst_key='gt_img',
        transforms=[
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
        ]),
    dict(type='lqit.PackInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type='lqit.DatasetWithGTImageWrapper',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/instances_train2017.json',
            data_prefix=dict(img='train2017/', gt_img_path='train2017/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline)))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
