_base_ = 'faster-rcnn_r50_fpn_1x_ruod.py'

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))

# add WandbVisBackend
# vis_backends = [
#     dict(type='LocalVisBackend'),
#     dict(type='WandbVisBackend',
#          init_kwargs=dict(
#             project='RUOD_detection',
#             name='faster-rcnn_x101-64x4d_fpn_1x_ruod',
#             entity='lqit',
#             )
#         )
# ]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
