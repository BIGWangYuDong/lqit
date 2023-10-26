_base_ = ['./tienet_retinanet_r50_fpn_1x_urpc-coco.py']

train_dataloader = dict(batch_size=4, num_workers=4)
