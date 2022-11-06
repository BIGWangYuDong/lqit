# dataset settings
dataset_type = 'BasicImageDataset'
data_root = 'data/uw_enhance/'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadGTImageFromFile', file_client_args=file_client_args),
    dict(
        type='TransBroadcaster',
        src_key='img',
        dst_key='gt_img',
        transforms=[
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
        ]),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadGTImageFromFile', file_client_args=file_client_args),
    dict(
        type='TransBroadcaster',
        src_key='img',
        dst_key='gt_img',
        transforms=[dict(type='Resize', scale=(512, 512), keep_ratio=True)]),
    dict(
        type='PackInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(
            dataset_type='underwater_enhancement', task_name='enhancement'),
        data_prefix=dict(img='input_train/', gt_img='gt_train/'),
        search_key='img',
        img_suffix=dict(img='png', gt_img='png'),
        file_client_args=file_client_args,
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
        test_mode=True,
        metainfo=dict(
            dataset_type='underwater_enhancement', task_name='enhancement'),
        data_prefix=dict(img='input_test/', gt_img='gt_test/'),
        search_key='img',
        img_suffix=dict(img='png', gt_img='png'),
        file_client_args=file_client_args,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='MAE', gt_key='img', pred_key='pred_img'),
    dict(type='MSE', gt_key='img', pred_key='pred_img'),
    dict(type='PSNR', gt_key='img', pred_key='pred_img'),
    dict(type='SSIM', gt_key='img', pred_key='pred_img'),
]
test_evaluator = val_evaluator
