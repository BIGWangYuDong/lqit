_base_ = '.self_enhance_base_loss.py'

enhance_model = dict(
    generator=dict(
        spacial_loss=dict(type='SpatialLoss', loss_weight=1.0),
        tv_loss=dict(type='MaskedTVLoss', loss_mode='mse', loss_weight=10.0),
        structure_loss=dict(type='StructureFFTLoss', loss_weight=1.0)))
