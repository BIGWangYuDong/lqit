_base_ = '../0_raw_image_detectors/faster-rcnn_x101-32x4d_fpn_1x_urpc-coco.py'

train_dataloader = dict(dataset=dict(data_prefix=dict(img='1_uwdet_HE/')))
val_dataloader = dict(dataset=dict(data_prefix=dict(img='1_uwdet_HE/')))
test_dataloader = val_dataloader
