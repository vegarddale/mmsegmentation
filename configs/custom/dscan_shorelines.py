_base_ = [
    '../_base_/models/ham_dscan.py',
    '../_base_/datasets/shorelines.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_pretrained_160k.py'
]
# checkpoint_file = "./work_dirs/dmscan_shorelines/iter_240000.pth"

crop_size = (768,768)

data_preprocessor = dict(size=crop_size)
train_dataloader=dict(batch_size=4)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        attn_module="DCNv3_SW_KA",
        kernel_size=[5, [1, 7]],
        pad=[2, [0, 3]],
        norm_cfg=dict(eps=1e-06, requires_grad=True, type='LN'),
        embed_dims=[32,
            64,
            160,
            256,],
        mlp_ratios=[8, 8, 4, 4],
        groups=[3, 6, 12, 24],
        depths=[6, 6, 24, 6],
    ),
    decode_head=dict(num_classes=8,
                    channels=512,
                    ham_channels=512,
                     in_channels=[64,
            160,
            256,]),
    # auxiliary_head=dict(
    #     in_channels=256,
    # #     channels=512,
    #     num_classes=8
    # )
)
