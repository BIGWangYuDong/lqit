_base_ = '../0_raw_image_detectors/faster-rcnn_x101-32x4d_fpn_2x_urpc-coco.py'

train_dataloader = dict(dataset=dict(data_prefix=dict(img='9_uwdet_UGAN/')))
val_dataloader = dict(dataset=dict(data_prefix=dict(img='9_uwdet_UGAN/')))
test_dataloader = val_dataloader
