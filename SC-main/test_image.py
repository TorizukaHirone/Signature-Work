from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import os

# 1. 指定你的配置文件和刚刚训练好的最终权重文件路径
config_file = 'configs/det/linear_probe/faster_rcnn/central_r50_fpn_dota.py'
# latest.pth 是框架自动生成的快捷链接，指向最后一个 epoch
checkpoint_file = 'work_dirs/central_r50_fpn_dota/latest.pth' 

# 2. 从 DOTA 验证集中随便挑一张图片来测试 (请替换为实际存在的文件名)
img_path = 'data_use/DOTA/val_split/images/P0053__1__385___0.png' # 记得检查这个路径！

# 3. 初始化模型 (这一步会自动读取配置里的 custom_imports)
print("正在加载模型，请稍候...")
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 4. 进行推理预测
print("正在进行目标检测...")
result = inference_detector(model, img_path)

# 5. 可视化结果并保存
# score_thr=0.5 表示只显示置信度大于 50% 的框
print("正在绘制彩色检测框并保存...")
show_result_pyplot(
    model, 
    img_path, 
    result, 
    score_thr=0.5, 
    out_file='result_visualization.jpg' # 结果图片会保存在这里
)
print("大功告成！快去打开 result_visualization.jpg 看看效果吧！")