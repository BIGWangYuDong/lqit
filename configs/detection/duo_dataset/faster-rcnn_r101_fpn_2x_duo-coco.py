_base_ = 'faster-rcnn_r50_fpn_2x_duo-coco.py'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

# add WandbVisBackend
# vis_backends = [
#     dict(type='LocalVisBackend'),
#     dict(type='WandbVisBackend',
#          init_kwargs=dict(
#             project='DUO_detection',
#             name='faster-rcnn_r101_fpn_2x_duo',
#             entity='lqit',
#             )
#         )
# ]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
