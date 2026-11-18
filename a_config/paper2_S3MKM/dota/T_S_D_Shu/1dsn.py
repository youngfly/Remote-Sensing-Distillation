dataset_type = 'DotaDataset'
data_root = '/home/airstudio/data/dota_data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train1024/DOTA_train1024.json',
        img_prefix=data_root + 'train1024/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val1024/DOTA_val1024.json',
        img_prefix=data_root + 'val1024/images/',
        pipeline=test_pipeline),
    test1=dict(
        type=dataset_type,
        ann_file=data_root + 'val1024/DOTA_val1024.json',
        img_prefix=data_root + 'val1024/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test1024/DOTA_test1024.json',
        img_prefix=data_root + 'test1024/images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[20, 22])
total_epochs = 24

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# # work_dir = './work_dirs/d_dota_cate_d12'
# # work_dir = './work_dirs/d_dota_geo'
# # work_dir = './work_dirs/d_dota_rela'
#
# # work_dir = './work_dirs/d_dota_cls'
# # work_dir = './work_dirs/d_dota_reg'
# # work_dir = './work_dirs/d_dota_fea'
# # work_dir = './work_dirs/d_dota_fea_cls'
# # work_dir = './work_dirs/d_dota_fea_reg'

# work_dir = './work_dirs/d_dota_fea_reg'
# work_dir = './work_dirs/d_dota_fea_cls_geo_mob'

work_dir = './work_dirs/shufflenet_dota'