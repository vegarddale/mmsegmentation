iters = 400000
# optimizer
optimizer = dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)
# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=optimizer,
    # clip_grad=dict(max_norm=0.01, norm_type=2),
    # paramwise_cfg=dict(
    #     custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=1.0,
        begin=0,
        end=iters,
        by_epoch=False)
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=iters, val_interval=25000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=25000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# auto_scale_lr = dict(enable=False, base_batch_size=8)