_base_ = [
    '../_base_/models/upernet_swin_b.py',
    '../_base_/datasets/shorelines.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_custom_160k_1e-4.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
train_dataloader=dict(batch_size=4)
model = dict(
    # backbone=dict(
    #     embed_dims=128,
    #     depths=[2, 2, 18, 2]),
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        # in_channels=[128, 256, 512, 1024],
        num_classes=8),
    auxiliary_head=dict(
        # in_channels=512,
        num_classes=8)
)