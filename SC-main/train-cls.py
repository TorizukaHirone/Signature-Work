import os
import sys
from mim.cli import cli

# 1. 设置你本地的绝对路径
desired_directory = r'F:\0\Signature Work\Code\SC-main'
os.chdir(desired_directory)
sys.path.insert(0, desired_directory)

# 2. 【关键修复】Windows 环境变量分隔符是分号 (;)，并使用绝对路径
os.environ["PYTHONPATH"] = desired_directory + ';' + os.environ.get("PYTHONPATH", "")

# 3. 指定使用第一张显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 4. 构造 mim 训练命令
sys.argv = [
    'mim',
    'train',
    'mmcls',
    'configs/cls/linear_probe/r50_Up-G-C_dota.py', # 确保指向这里
]

cli()