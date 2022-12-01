_base_ = '../0_raw_image_detectors/faster-rcnn_r50_fpn_1x_urpc-coco.py'

train_dataloader = dict(dataset=dict(data_prefix=dict(img='9_uwdet_UGAN/')))
val_dataloader = dict(dataset=dict(data_prefix=dict(img='9_uwdet_UGAN/')))
test_dataloader = val_dataloader
