checkpoint_file = "./work_dirs/dmscan_isaid/iter_22500.pth"

_base_ = [
    '../_base_/models/mask2former_dscan.py',
    '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_custom_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
train_dataloader=dict(batch_size=4)
model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    backbone=dict(
        attn_module="DCNv3_SW_KA",
        kernel_size=[5, [1, 7]],
        pad=[2, [0, 3]],
        norm_cfg=dict(eps=1e-06, requires_grad=True, type='LN'),
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        groups=[3, 6, 12, 24],
        depths=[6, 6, 24, 6],
    ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6,
                     # channels=512,
                     # ham_channels=512,
in_channels=[32, 64, 160, 256]))


