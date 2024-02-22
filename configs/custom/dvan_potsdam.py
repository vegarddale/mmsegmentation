_base_ = [
    '../_base_/models/upernet_dvan.py',
    '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_custom_160k_1e-4.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    backbone=dict(
        attn_module="DCNv3KA",
        kernel_size=[5, 7],
        pad=[2,3]
    ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6))
