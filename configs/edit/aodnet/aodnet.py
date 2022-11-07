_base_ = [
    '../_base_/datasets/cityscape_enhancement_with_txt.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
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