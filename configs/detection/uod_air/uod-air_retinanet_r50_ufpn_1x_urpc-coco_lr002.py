_base_ = [
    './uod-air_retinanet_r50_ufpn_1x_urpc-coco.py',
]
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))  # loss may NaN without clip_grad
