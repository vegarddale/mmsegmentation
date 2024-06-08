
# checkpoint_file = "./work_dirs/dmscan_potsdam/iter_50.pth"

# _base_ = [
#     '../_base_/models/ham_dvan.py',
#     '../_base_/datasets/shorelines.py',
#     '../_base_/default_runtime.py',
#     '../_base_/schedules/schedule_custom_160k_1e-4.py'
# ]
# crop_size = (768, 768)
# data_preprocessor = dict(size=crop_size)
# train_dataloader=dict(batch_size=4)
# model = dict(
#     # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
#     backbone=dict(
#         attn_module="DCNv3_SW_KA",
#         kernel_size=[5, [1, 11]],
#         pad=[2, [0,5]],
#         norm_cfg=dict(eps=1e-06, requires_grad=True, type='LN'),
#         embed_dims=[32,64,160,256],
#         mlp_ratios=[8, 8, 4, 4],
#         groups=[3, 6, 12, 24],
#         depths=[4, 4, 18, 4],
#     ),
#     data_preprocessor=data_preprocessor,
#     decode_head=dict(num_classes=8,
#                      channels=512,
#                      ham_channels=512,
# in_channels=[64,160,256]))

############# up old, this for imagenet pretrain


checkpoint_file = "./work_dirs/dmscan_potsdam/iter_110000.pth"
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
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    backbone=dict(
        type='DVAN',
        attn_module="DCNv3_SW_KA",
        kernel_size=[5, [1, 11]],
        pad=[2, [0, 5]],
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 3],
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(eps=1e-06, requires_grad=True, type='LN'),
        channel_attention=True),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=16,
                     # channels=1024,
                     # ham_channels=1024,
in_channels=[64, 160, 256]))

