_base_ = '../0_raw_image_detectors/faster-rcnn_x101-64x4d_fpn_1x_urpc-coco.py'

train_dataloader = dict(
    dataset=dict(data_prefix=dict(img='6_uwdet_DMIL_HDP/')))
val_dataloader = dict(dataset=dict(data_prefix=dict(img='6_uwdet_DMIL_HDP/')))
test_dataloader = val_dataloader
