_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/datasets/rtts_coco.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=5))

# 4bs * 4GPUs
train_dataloader = dict(batch_size=4, num_workers=4)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# show_dir = 'work_dirs/a_tienet_vis_new/rtts/retina'
#
# default_hooks = dict(
#     visualization=dict(
#         type='EnhanceDetVisualizationHook',
#         draw=True,
#         test_out_dir=show_dir + '/baseline',
#         show_on_enhance=False,
#         draw_gt=False,
#         draw_pred=True))
#