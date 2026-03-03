_base_ = [
    '../../_base_/models/faster_rcnn_r50_fpn_intern.py',
    '../../_base_/schedules/schedule.py', 
    '../../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['gvbenchmark'],
    allow_failed_imports=False
)

CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
]

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='torchvision://resnet50'
        )
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=15
        )
    )
)

dataset_type = 'CocoDataset'
data_root = 'data_use/DOTA/'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'train_split/train_coco.json',
        img_prefix=data_root + 'train_split/images/',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'val_split/val_coco.json',
        img_prefix=data_root + 'val_split/images/',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=data_root + 'val_split/val_coco.json',
        img_prefix=data_root + 'val_split/images/',
        pipeline=test_pipeline
    )
)

optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

fp16 = dict(loss_scale=512.)