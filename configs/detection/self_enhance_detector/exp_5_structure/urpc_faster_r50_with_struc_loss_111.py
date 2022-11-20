_base_ = 'urpc_faster_base.py'

model = dict(
    enhance_model=dict(generator=dict(model=dict(kernel_size=[1, 1, 1]))))
