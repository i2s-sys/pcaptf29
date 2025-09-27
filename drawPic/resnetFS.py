import matplotlib.pyplot as plt
import numpy as np
import argparse
import time

# 数据

# 处理命令行参数
parser = argparse.ArgumentParser(description='Generate F1 score plots.')
parser.add_argument('--dataset', type=str, default='UNSW-NB', help='Dataset name (default: unsw-iot)')
args = parser.parse_args()

# unsw-iot
if args.dataset == 'UNSW-IoT':
    feature_nums = [1, 2, 4, 8, 16, 24, 32, 72]
    microF1 = [0.4662, 0.7213, 0.9325, 0.9871, 0.9980, 0.9986, 0.9982, 0.9967] # mamba的
    macroF1 = [0.2965, 0.6925, 0.9301, 0.9865, 0.9978, 0.9986, 0.9983, 0.9964]

elif args.dataset == 'UNSW-NB':
    feature_nums = [1, 2, 4, 8, 16, 24, 32, 42]
    microF1 = [0.5506, 0.6935, 0.8191, 0.8486, 0.9044, 0.9423, 0.9446, 0.9266] # mamba
    macroF1 = [0.3910, 0.6868, 0.8185, 0.8483, 0.9040, 0.9423, 0.9442, 0.9257]

elif args.dataset == 'NSL-KDD':
    feature_nums = [1, 2, 4, 8, 16, 24, 32, 41]
    microF1 = [0.4308, 0.7489, 0.8981, 0.8457, 0.8404, 0.8325, 0.8259, 0.8610]
    macroF1 = [0.2594, 0.7435, 0.8985, 0.8454, 0.8400, 0.8308, 0.8250, 0.8610]

# 设置字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 14  # 放大一个字号

# 获取当前时间戳
current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

# 创建均匀分布的 x 轴刻度
x_ticks = np.linspace(1, len(feature_nums), len(feature_nums))

# 将 microF1 和 macroF1 乘以 100
microF1_percent = np.array(microF1) * 100
macroF1_percent = np.array(macroF1) * 100

# 画 microF1 图
plt.figure()
plt.plot(x_ticks, microF1_percent, marker='o', linestyle='-', color='#1f77b4')  # 浅蓝色
plt.xlabel('Number of selected features (K)', fontweight='bold')
plt.ylabel('MicroF1 (%)', fontweight='bold')
plt.title(f'{args.dataset}: NetMamba Model', fontweight='bold')
plt.xticks(x_ticks, feature_nums)  # 保持等间距且显示正确的刻度标签
plt.grid(False)

# 在点上显示具体的精度值
for i, txt in enumerate(microF1_percent):
    plt.annotate(f'{txt:.2f}', (x_ticks[i], microF1_percent[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)

# 保存并显示图像，裁剪白边并设置更高的 DPI
plt.savefig(f'{current_time}_{args.dataset}_microF1_vs_features.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.show()

# 画 macroF1 图
plt.figure()
plt.plot(x_ticks, macroF1_percent, marker='o', linestyle='-', color='#FFA07A')  # 浅橙色
plt.xlabel('Number of selected features (K)', fontweight='bold')
plt.ylabel('MacroF1 (%)', fontweight='bold')
plt.title(f'{args.dataset}: NetMamba Model', fontweight='bold')
plt.xticks(x_ticks, feature_nums)  # 保持等间距且显示正确的刻度标签
plt.grid(False)

# 在点上显示具体的精度值
for i, txt in enumerate(macroF1_percent):
    plt.annotate(f'{txt:.2f}', (x_ticks[i], macroF1_percent[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)

# 保存并显示图像，裁剪白边并设置更高的 DPI
plt.savefig(f'{current_time}_{args.dataset}_macroF1_vs_features.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.show()
