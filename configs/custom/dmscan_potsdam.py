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
        attn_module="DCNv3_SW_KA",
        kernel_size=[3, [1, 5], [1, 7], [1, 11]],
        pad=[2, [0, 3], [0, 5], [0, 10]],
        embed_dims=[8, 16, 32, 64],
        mlp_ratios=[4, 4, 4, 4]
    ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6))
