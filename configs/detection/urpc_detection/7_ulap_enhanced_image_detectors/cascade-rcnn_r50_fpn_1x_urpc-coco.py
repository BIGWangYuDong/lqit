_base_ = '../0_raw_image_detectors/cascade-rcnn_r50_fpn_1x_urpc-coco.py'

train_dataloader = dict(dataset=dict(data_prefix=dict(img='7_uwdet_ULAP/')))
val_dataloader = dict(dataset=dict(data_prefix=dict(img='7_uwdet_ULAP/')))
test_dataloader = val_dataloader
