# TODO: delete after fully support editor metric and datasets.
_base_ = [
    '../_base_/models/unet.py',
    '../_base_/datasets/underwater_enhancement.py',
    # '../_base_/datasets/underwater_enhancement_with_ann.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(num_workers=0, persistent_workers=False)
val_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False)

test_dataloader = val_dataloader
