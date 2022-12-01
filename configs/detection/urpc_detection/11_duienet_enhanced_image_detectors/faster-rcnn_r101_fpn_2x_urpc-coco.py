_base_ = '../0_raw_image_detectors/faster-rcnn_r101_fpn_2x_urpc-coco.py'

train_dataloader = dict(
    dataset=dict(data_prefix=dict(img='11_uwdet_DUIENet/')))
val_dataloader = dict(dataset=dict(data_prefix=dict(img='11_uwdet_DUIENet/')))
test_dataloader = val_dataloader