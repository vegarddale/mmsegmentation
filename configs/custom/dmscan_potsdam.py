
checkpoint_file = "./work_dirs/dmscan_potsdam/20240322_225633/iter_160000.pth"

_base_ = [
    '../_base_/models/ham_dvan.py',
    '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_pretrained_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
train_dataloader=dict(batch_size=4)
model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    backbone=dict(
        attn_module="DCNv3_SW_KA",
        kernel_size=[5, [1, 13]],
        pad=[2, [0, 6]],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        embed_dims=[64, 128, 256, 512],
        mlp_ratios=[8, 8, 4, 4],
        groups=[3, 6, 12, 24],
        depths=[3, 4, 18, 5],
    ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6,
in_channels=[128, 256, 512], channels=512, ham_channels=512))
