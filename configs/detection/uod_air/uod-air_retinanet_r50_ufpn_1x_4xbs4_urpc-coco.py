_base_ = [
    './uod-air_retinanet_r50_ufpn_1x_urpc-coco.py',
]
train_dataloader = dict(batch_size=4, num_workers=4)
