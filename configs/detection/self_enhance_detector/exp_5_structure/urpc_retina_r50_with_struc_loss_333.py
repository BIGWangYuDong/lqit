_base_ = './urpc_retina_r50_with_struc_loss_base.py',

model = dict(enhance_model=dict(kernel_size=[3, 3, 3]))
