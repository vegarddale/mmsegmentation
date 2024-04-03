
# optimizer
optimizer = dict(
        type='AdamW', lr=9.3439e-08, betas=(0.9, 0.999), eps=1e-8, weight_decay=3.129999999999976e-12) # type='AdamW', lr=9.3439e-08, betas=(0.9, 0.999), eps=1e-8, weight_decay=3.129999999999976e-10)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=9e-08,
        power=1.0,
        begin=0,
        end=500,
        by_epoch=False)
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=500, val_interval=25)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=25, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=25),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))