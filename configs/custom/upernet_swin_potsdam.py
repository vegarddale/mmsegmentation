_base_ = [
    '../_base_/models/upernet_swin.py',
    '../_base_/datasets/shorelines.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_25k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
train_dataloader=dict(batch_size=4)
model = dict(
    # backbone=dict(depths=[3, 4, 18, 5]),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=9),
    auxiliary_head=dict(num_classes=9)
)