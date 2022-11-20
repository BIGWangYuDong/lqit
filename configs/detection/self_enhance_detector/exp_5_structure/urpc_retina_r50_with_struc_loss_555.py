_base_ = 'urpc_retina_base.py'

model = dict(
    enhance_model=dict(generator=dict(model=dict(kernel_size=[5, 5, 5]))))
