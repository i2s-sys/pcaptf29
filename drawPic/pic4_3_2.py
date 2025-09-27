import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import argparse
import time
import numpy as np

# 数据
feature_nums = [1, 2, 4, 8, 16, 24, 32, 72]
microF1 = [0.7951, 0.8634, 0.9152, 0.9491, 0.917, 0.9782, 0.9819, 0.9802]
macroF1 = [0.2744, 0.4994, 0.671, 0.79, 0.831, 0.8591, 0.8787, 0.8625]

# 设置字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"

# 获取当前时间戳
current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

# 处理命令行参数
parser = argparse.ArgumentParser(description='Generate F1 score plots.')
parser.add_argument('--dataset', type=str, default='unsw-iot', help='Dataset name (default: unsw-iot)')
args = parser.parse_args()

# 生成等距的x轴位置
x_positions = np.arange(len(feature_nums))

# 画 microF1 图
plt.figure()
plt.plot(x_positions, microF1, marker='o', linestyle='-', color='#1f77b4')  # 浅蓝色
plt.xticks(x_positions, feature_nums)  # 设置x轴刻度和标签
plt.xlabel('Number of selected features (K)', fontweight='bold')
plt.ylabel('Acc (%)', fontweight='bold')
plt.title('UNSW-IOT', fontweight='bold')
# 显示具体精度值，以百分比形式
for i, txt in enumerate(microF1):
    plt.annotate(f'{txt * 100:.2f}', (x_positions[i], microF1[i]), textcoords="offset points", xytext=(0,5), ha='center')
plt.grid(False)
plt.savefig(f'{current_time}_{args.dataset}_microF1_vs_features.png')
plt.show()

# 画 macroF1 图
plt.figure()
plt.plot(x_positions, macroF1, marker='o', linestyle='-', color='#FFA07A')  # 浅橙色
plt.xticks(x_positions, feature_nums)  # 设置x轴刻度和标签
plt.xlabel('Number of selected features (K)', fontweight='bold')
plt.ylabel('Acc (%)', fontweight='bold')
plt.title('UNSW-IOT', fontweight='bold')
# 显示具体精度值，以百分比形式
for i, txt in enumerate(macroF1):
    plt.annotate(f'{txt * 100:.2f}', (x_positions[i], macroF1[i]), textcoords="offset points", xytext=(0,5), ha='center')
plt.grid(False)
plt.savefig(f'{current_time}_{args.dataset}_macroF1_vs_features.png')
plt.show()
