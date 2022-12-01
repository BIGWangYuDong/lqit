_base_ = '../0_raw_image_detectors/faster-rcnn_x101-64x4d_fpn_2x_urpc-coco.py'

train_dataloader = dict(dataset=dict(data_prefix=dict(img='3_uwdet_WB/')))
val_dataloader = dict(dataset=dict(data_prefix=dict(img='3_uwdet_WB/')))
test_dataloader = val_dataloader
