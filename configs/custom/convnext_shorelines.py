_base_ = [
    '../_base_/models/upernet_convnext.py',
    '../_base_/datasets/shorelines.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_custom_160k_1e-4.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
train_dataloader=dict(batch_size=4)
# checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='nano',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.3,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False),
        # init_cfg=dict(
        #     type='Pretrained', checkpoint=checkpoint_file,
        #     prefix='backbone.')),
    decode_head=dict(
        in_channels=[80,
            160,
            320,
            640,],
        num_classes=8,
    ),
    auxiliary_head=dict(in_channels=320, num_classes=8),
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)
