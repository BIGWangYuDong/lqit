_base_ = [
    '../../detection/_base_/schedules/schedule_1x.py', '../../detection/_base_/default_runtime.py'
]

model = dict(
    type='lqit.BaseEditModel',
    data_preprocessor=dict(
        type='lqit.EditDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        gt_name='img'),
    generator=dict(
        _scope_='lqit',
        type='AODNetGenerator',
        aodnet=dict(type='AODNet'),
        pixel_loss=dict(type='MSELoss', loss_weight=1.0)
       ))

# dataset settings
dataset_type = 'CityscapesDataset'
data_root = '/home/tju531/hwr/Datasets/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='lqit.LoadGTImageFromFile'),
    dict(
        type='lqit.TransBroadcaster',
        src_key='img',
        dst_key='gt_img',
        transforms=[
            dict(type='Resize', scale=(256, 256), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5)
        ]),
    dict(type='lqit.PackInputs', )
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='lqit.DatasetWithClearImageWrapper',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='cityscape_foggy/annotations_json/instancesonly_filtered_gtFine_train.json',
            data_prefix=dict(img='cityscape_foggy/train/', gt_img_path='cityscape/train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline),
    suffix='png'
    ))

val_dataloader = None
val_cfg = None
test_cfg = None

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.0001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=10,
        by_epoch=True,
        milestones=[6, 9],
        gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))