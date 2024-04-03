
# checkpoint_file = "./work_dirs/weights/iter_100_75.61.pth"

_base_ = [
    '../_base_/models/mask2former_dmscan.py',
    '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_custom_160k_1e-4.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
train_dataloader=dict(batch_size=4)
model = dict(
    # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    backbone=dict(
        attn_module="DCNv3_SW_KA",
        kernel_size=[5, [1, 11]],
        pad=[2, [0, 5]],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        groups=[3, 6, 12, 24],
        depths=[5, 5, 22, 5],
    ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=6))
