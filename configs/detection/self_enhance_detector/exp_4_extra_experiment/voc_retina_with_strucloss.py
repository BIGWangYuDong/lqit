_base_ = [
    '../base_editor/self_enhance_with_high_strcloss_gf.py',
    './voc_base_retina.py'
]

model = dict(
    _delete_=True,
    type='SelfEnhanceDetector',
    detector={{_base_.model}},
    enhance_model={{_base_.enhance_model}})

# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args={{_base_.file_client_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='lqit.SetInputImageAsGT'),
    dict(type='lqit.PackInputs')
]

dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='ConcatDataset',
            # VOCDataset will add different `DATASET_TYPE` in dataset.metainfo,
            # which will get error if using ConcatDataset. Adding
            # `ignore_keys` can avoid this error.
            ignore_keys=['DATASET_TYPE'],
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='VOC2007/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=train_pipeline),
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='VOC2012/ImageSets/Main/trainval.txt',
                    data_prefix=dict(sub_data_root='VOC2012/'),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=train_pipeline)
            ])))

model_wrapper_cfg = dict(
    type='SelfEnhanceModelDDP',
    broadcast_buffers=False,
    find_unused_parameters=False)
