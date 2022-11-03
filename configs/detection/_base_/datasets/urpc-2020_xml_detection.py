# dataset settings
dataset_type = 'URPCXMLDataset'
data_root = 'data/UWDetData/'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/train.txt',
        meta_file='ImageMetas/train-image-metas.pkl',
        img_subdir='0_uwdet_RAW',
        ann_subdir='annotations_xml',
        data_prefix=dict(sub_data_root=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/val.txt',
        meta_file='ImageMetas/val-image-metas.pkl',
        img_subdir='0_uwdet_RAW',
        ann_subdir='annotations_xml',
        data_prefix=dict(sub_data_root=''),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='area')
test_evaluator = val_evaluator
