_base_ = [
    '../_base_/models/ham_dvan.py',
    '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_custom_160k_1e-4.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
train_dataloader=dict(batch_size=4)
model = dict(
    backbone=dict(
        attn_module="DCNv3_SW_KA",
        kernel_size=[5, [1, 7], [1, 11], [1, 21]],
        pad=[2, [0, 3], [0, 5], [0, 10]],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        embed_dims=[32, 64, 160, 256],
        # mlp_ratios=[8, 8, 4, 4],
        groups=[3, 6, 12, 24]
        # depths=[3, 4, 18, 5],
    ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6))
