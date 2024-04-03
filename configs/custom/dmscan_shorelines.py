_base_ = [
    '../_base_/models/ham_dvan.py',
    '../_base_/datasets/shorelines.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_25k.py'
]

crop_size = (512,512)

data_preprocessor = dict(size=crop_size)
train_dataloader=dict(batch_size=4)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        attn_module="DCNv3_SW_KA",
        kernel_size=[5, [1, 11]],
        pad=[2, [0, 5]],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        groups=[3, 6, 12, 24],
        depths=[4, 4, 18, 4],
    ),
    decode_head=dict(num_classes=9,)
                     # in_channels=[64, 128, 256, 512],
                     # channels=512),
    # auxiliary_head=dict(
    #     in_channels=256,
    #     channels=512,
    #     num_classes=9
    # )
)


