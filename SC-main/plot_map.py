import json
import os
import glob
import matplotlib.pyplot as plt

def main():
    # 1. 自动定位最新的日志文件
    log_dir = 'work_dirs/central_r50_fpn_dota'
    log_files = glob.glob(os.path.join(log_dir, '*.log.json'))
    
    if not log_files:
        print("未找到 .log.json 文件，请检查路径是否正确。")
        return
        
    # 按文件修改时间排序，拿到最新的一份日志
    log_files.sort(key=os.path.getmtime)
    latest_log = log_files[-1]
    print(f"正在解析日志文件: {latest_log}")

    epochs = []
    map_50_95 = []  # 严苛标准 mAP (IoU=0.5:0.95)
    map_50 = []     # 常用标准 mAP_50 (IoU=0.50)

    # 2. 逐行读取 JSON 格式的日志，提取验证集(val)的成绩
    with open(latest_log, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # 寻找包含验证集成绩的行 (mode='val')
                if data.get('mode') == 'val' and 'bbox_mAP' in data:
                    epochs.append(data.get('epoch', len(epochs) + 1))
                    map_50_95.append(data['bbox_mAP'])
                    map_50.append(data['bbox_mAP_50'])
            except json.JSONDecodeError:
                continue

    if not epochs:
        print("日志中没有找到验证集(val)的 mAP 数据，请确认模型是否完成了 epoch 评估。")
        return

    # 3. 开始绘制符合学术规范的双曲线图表
    plt.figure(figsize=(8, 5)) # 设置画布比例 (宽8, 高5)
    
    # 绘制 mAP_50 曲线 (橙色，代表基础检出率)
    plt.plot(epochs, map_50, marker='o', markersize=6, label='mAP@0.50', color='#ff7f0e', linewidth=2)
    
    # 绘制严苛的 mAP 曲线 (蓝色，代表高精度贴合率)
    plt.plot(epochs, map_50_95, marker='s', markersize=6, label='mAP (IoU=0.50:0.95)', color='#1f77b4', linewidth=2)

    # 纯净的排版和字体设置
    plt.title('Validation mAP over Epochs', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean Average Precision (mAP)', fontsize=12)
    
    # 设置 X 轴刻度为整数 (因为 epoch 是整数)
    plt.xticks(epochs)
    
    # 添加辅助网格线
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=11)

    # 去除顶部和右侧的多余边框线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 4. 保存为高分辨率图片 (300 DPI)
    save_path = 'paper_figure_mAP_curve.png'
    plt.savefig(save_path, dpi=300)
    print(f"大功告成！高精度 mAP 曲线图已保存为: {save_path}")
    
    # 在屏幕上显示一下
    plt.show()

if __name__ == '__main__':
    main()