_base_ = ['./atss_r50_fpn_1x_rtts-coco_lr002.py']

# model settings
model = dict(
    neck=dict(
        type='lqit.DFFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        shape_level=2,
        num_outs=5))
