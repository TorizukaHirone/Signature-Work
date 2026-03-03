import os
import sys
from mim.cli import cli

# 1. 指定绝对路径
desired_directory = r'F:\0\Signature Work\Code\SC-main'
os.chdir(desired_directory)
sys.path.insert(0, desired_directory)

# 2. 修复 Windows 环境变量分号问题
os.environ["PYTHONPATH"] = desired_directory + ';' + os.environ.get("PYTHONPATH", "")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 3. 指向我们刚写好的 DOTA 协同配置文件
if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    # 如果你在终端忘了加路径，默认跑你的 SC 模型兜底
    config_file = 'configs/det/linear_probe/faster_rcnn/central_r50_fpn_dota.py'

# --- 传给 MMDetection 核心的参数 ---
sys.argv = [
    'main',
    'train',
    'mmdet',
    config_file, # <--- 这里变成了动态变量
]

cli()