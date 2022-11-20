_base_ = './urpc_faster_r50_with_struc_loss_base.py',

model = dict(enhance_model=dict(kernel_size=[3, 3, 3]))
