import os
import sys
from mim.cli import cli

os.environ["PYTHONPATH"] = '.:' + os.environ.get("PYTHONPATH", "")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

desired_directory = 'F:/0/Signature Work/Code/SC-main/'
os.chdir(desired_directory)

sys.path.insert(0, desired_directory)

sys.argv = [
    'mim',
    'test',
    'mmcls',

    'F:/0/Signature Work/Code/SC-main/configs/cls/linear_probe/r50_Up-G-C_pretrain_cifar100_10p.py',

    '--checkpoint', "F:/0/Signature Work/Code/SC-main/work_dirs/pth-cls/best.pth",

    '--metrics', 'accuracy'
]

cli()
