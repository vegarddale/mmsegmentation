# optimizer
optimizer = dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=5e-5,
        power=1.0,
        begin=0,
        end=80000,
        by_epoch=False),
    dict(
        type='LinearParamScheduler',
         param_name='lr',  # adjust the 'lr' in `optimizer.param_groups`
         start_factor=0.15,
         by_epoch=False,
         begin=80000,
         end=80001),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=80001,
        end=240000,
        by_epoch=False),
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=240000, val_interval=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))


