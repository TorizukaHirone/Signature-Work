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

    steps = []
    losses = []

    # 2. 逐行读取 JSON 格式的日志
    with open(latest_log, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # 只提取训练阶段的 loss
                if data.get('mode') == 'train' and 'loss' in data:
                    steps.append(len(steps) * 50) # 你的配置是每 50 步记录一次
                    losses.append(data['loss'])
            except json.JSONDecodeError:
                continue

    # 3. 开始绘制符合学术规范的图表
    plt.figure(figsize=(8, 5)) # 设置画布比例 (宽8, 高5)
    
    # 绘制折线，使用经典的学术蓝色，稍微增加透明度让曲线看起来更平滑
    plt.plot(steps, losses, label='Training Loss', color='#1f77b4', linewidth=1.5, alpha=0.85)

    # 纯净的排版和字体设置
    plt.title('Training Loss over Iterations', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    
    # 添加辅助网格线，方便读者对照数值
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=11)

    # 去除顶部和右侧的多余边框线 (更符合现代学术期刊的审美)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 4. 保存为高分辨率图片 (300 DPI 是学术论文的标准要求)
    save_path = 'paper_figure_loss_curve.png'
    plt.savefig(save_path, dpi=300)
    print(f"大功告成！高精度曲线图已保存为: {save_path}")
    
    # 在屏幕上显示一下
    plt.show()

if __name__ == '__main__':
    main()