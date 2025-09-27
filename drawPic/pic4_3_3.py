import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import argparse
import time

# 数据

# 处理命令行参数
parser = argparse.ArgumentParser(description='Generate F1 score plots.')
parser.add_argument('--dataset', type=str, default='NSL-KDD', help='Dataset name (default: unsw-iot)')
args = parser.parse_args()

# unsw-iot
if args.dataset == 'UNSW-IOT':
    feature_nums = [1, 2, 4, 8, 16, 24, 32, 72]
    microF1 = [0.7951, 0.8634, 0.9152, 0.9491, 0.917, 0.9782, 0.9819, 0.9867]
    macroF1 = [0.2744, 0.4994, 0.671, 0.79, 0.831, 0.8591, 0.8787, 0.8835]
elif args.dataset == 'UNSW-NB':
    feature_nums = [1, 2, 4, 8, 16, 24, 32, 42]
    microF1 = [0.5497, 0.7931, 0.8692, 0.9148, 0.9089, 0.9012, 0.9161, 0.9331]
    macroF1 = [0.3547, 0.772, 0.8653, 0.9129, 0.9068, 0.8983, 0.9161, 0.9317]
elif args.dataset == 'NSL-KDD':
    feature_nums = [1, 2, 4, 8, 16, 24, 32, 41]
    microF1 = [0.4323, 0.7019, 0.6772, 0.7296, 0.8006, 0.7833, 0.7876, 0.8026]
    macroF1 = [0.3018, 0.6956, 0.6565, 0.7282, 0.8004, 0.7824, 0.7870, 0.8024]

# 设置字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 14  # 放大一个字号

# 获取当前时间戳
current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

# 创建均匀分布的 x 轴刻度
x_ticks = np.linspace(1, len(feature_nums), len(feature_nums))

# 画 microF1 图
plt.figure()
plt.plot(x_ticks, microF1, marker='o', linestyle='-', color='#1f77b4')  # 浅蓝色
plt.xlabel('Number of selected features (K)', fontweight='bold')
plt.ylabel('Acc (%)', fontweight='bold')
plt.title(f'{args.dataset}: Resnet Model', fontweight='bold')
plt.xticks(x_ticks, feature_nums)  # 保持等间距且显示正确的刻度标签
plt.grid(False)

# 在点上显示具体的精度值
for i, txt in enumerate(microF1):
    plt.annotate(f'{txt:.4f}', (x_ticks[i], microF1[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)

# 保存并显示图像，裁剪白边并设置更高的 DPI
plt.savefig(f'{current_time}_{args.dataset}_microF1_vs_features.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.show()

# 画 macroF1 图
# plt.figure()
# plt.plot(x_ticks, macroF1, marker='o', linestyle='-', color='#FFA07A')  # 浅橙色
# plt.xlabel('Number of selected features (K)', fontweight='bold')
# plt.ylabel('Acc (%)', fontweight='bold')
# plt.title(f'{args.dataset}: Resnet Model', fontweight='bold')
# plt.xticks(x_ticks, feature_nums)  # 保持等间距且显示正确的刻度标签
# plt.grid(False)

# 在点上显示具体的精度值
# for i, txt in enumerate(macroF1):
#     plt.annotate(f'{txt:.4f}', (x_ticks[i], macroF1[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)

# 保存并显示图像，裁剪白边并设置更高的 DPI
# plt.savefig(f'{current_time}_{args.dataset}_macroF1_vs_features.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
# plt.show()
