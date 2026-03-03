_base_ = [
    '../_base_/models/release/central_resnet50.py',
    '../_base_/datasets/cifar100_10p.py', '../_base_/schedules/mmlab.py',
    '../_base_/default_runtime.py', '../_base_/custom_import.py'
]

# 1. 覆盖类别数量
dataset_num_classes = 15
model = dict(
    head=dict(
        num_classes=dataset_num_classes
    )
)

# 2. 覆盖类别名称 (DOTA 的 15 个类别)
CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
]

# 3. 覆盖数据集路径和 Batch Size (极其关键，防止 3060 爆显存)
data = dict(
    samples_per_gpu=2,  # 显存限制，1024的大图只能跑 2
    workers_per_gpu=0,  # 内存限制，关掉多进程
    train=dict(
        type='DOTADataset',    # 假定目标检测类名为 DOTA
        data_prefix='data_use/DOTA/train_split',
        classes=CLASSES,
        ann_file='data_use/DOTA/train_split/labelTxt' # 暂时指向这里，看框架怎么读
    ),
    val=dict(
        type='DOTADataset',
        data_prefix='data_use/DOTA/val_split',
        classes=CLASSES,
        ann_file='data_use/DOTA/val_split/labelTxt'
    ),
    test=dict(
        type='DOTADataset',
        data_prefix='data_use/DOTA/val_split',
        classes=CLASSES,
        ann_file='data_use/DOTA/val_split/labelTxt'
    )
)