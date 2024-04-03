_base_ = [
    '../_base_/models/upernet_swin.py',
    '../_base_/datasets/beachtypes_c.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_25k.py'
]

crop_size = (500, 1500)
data_preprocessor = dict(size=crop_size)
train_dataloader=dict(batch_size=4)
model = dict(
    data_preprocessor=data_preprocessor,
decode_head=dict(num_classes=6))

