_base_ = [
    '../_base_/models/ham_mscan.py',
    '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_custom_160k_1e-4.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
train_dataloader=dict(batch_size=4)
model = dict(
     backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[4, 4, 4, 4],
        depths=[3, 5, 27, 3],
    ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6,
                     in_channels=[128, 320, 512],
                     channels=1024))


