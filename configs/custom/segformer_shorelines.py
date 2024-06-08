_base_ = ['../_base_/models/segformer_mit-b0.py',
         '../_base_/datasets/shorelines.py',
        '../_base_/default_runtime.py',
        '../_base_/schedules/schedule_custom_160k_1e-4.py'
         ]

# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'  # noqa
crop_size = (512,512)

data_preprocessor = dict(size=crop_size)
train_dataloader=dict(batch_size=4)
# model settings
model = dict(
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 18, 3]),
    data_preprocessor=data_preprocessor,
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=8))
