import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置字体为Times New Roman并加粗
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 14  # 字号为14

# 处理命令行参数
parser = argparse.ArgumentParser(description='Generate TSM-epoch plots.')
parser.add_argument('--model', type=str, default='NetMamba', help='')
parser.add_argument('--dataset', type=str, default='UNSW-IOT', help='Dataset name (default: unsw-iot)')
parser.add_argument('--K', type=int, default=24, help='')
args = parser.parse_args()

# unsw-iot
data = []
if args.model == 'Resnet':
    if args.dataset == 'UNSW-IOT':
        if args.K == 16:
            data = [0, 13, 12, 14, 13, 14, 15, 15, 14, 15, 15, 13, 14, 15, 15, 15, 15, 15, 16, 16, 16, 15, 14, 15, 14, 15, 14, 14, 15, 15]
        elif args.K == 24:
            data = [0, 22, 21, 21, 21, 24, 23, 23, 23, 22, 22, 23, 23, 22, 22, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 23, 23]
        elif args.K == 32:
            data = [0, 29, 27, 28, 29, 30, 30, 30, 30, 31, 31, 31, 30, 30, 31, 32, 32, 32, 32, 31, 31, 32, 31, 31, 31, 30, 31, 31, 31, 31]
        else:
            data = [0, 37, 36, 38, 37, 37, 38, 38, 37, 37, 39, 39, 38, 38, 38, 39, 40, 38, 38, 40, 40, 40, 40, 39, 39, 40, 39, 39, 40, 39]
    elif args.dataset == 'UNSW-NB':
        data = []
    elif args.dataset == 'NSL-KDD':
        data = []
elif args.model == 'NetMamba':
    if args.dataset == 'UNSW-IOT':
        if args.K == 8:
            data = [0, 7, 6, 6, 7, 7, 6, 5, 6, 7, 7, 8, 7, 7, 8, 7, 7, 8, 8, 7, 7, 7, 7, 8, 8, 7, 7, 8, 8, 8, 8, 7, 7, 8, 7, 7, 8, 7, 7, 7, 6, 7, 8, 7, 7, 7, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 6, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        elif args.K == 16:
            data = [0, 13, 9, 12, 15, 15, 14, 11, 11, 12, 14, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 15, 14, 15, 14, 14, 16, 16, 16, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 16, 15, 15, 16, 16, 16]
        elif args.K == 24:
            data = [0, 16, 12, 18, 14, 15, 20, 20, 19, 23, 23, 21, 22, 22, 23, 22, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 23, 22, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 23, 23, 24, 24, 24, 24, 24, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]
        elif args.K == 32:
            data = [0, 22, 18, 25, 22, 24, 28, 27, 28, 30, 30, 31, 32, 31, 31, 31, 32, 31, 30, 31, 30, 31, 31, 31, 31, 30, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 31, 31, 31, 31, 32, 31, 31, 32, 32, 32, 32, 32, 31, 31, 32, 31, 31, 32, 31, 31, 32, 32, 32, 31, 30, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 31, 31, 32, 32, 32, 31, 31, 31, 31, 32, 32, 32, 32, 32, 31, 31, 32, 32, 32, 32, 32, 32, 31, 31, 32, 32, 32, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32]

# 数据数组

# 找到第一次出现K值的位置
first_occurrence = data.index(args.K)

# 创建图形
plt.figure(dpi=600)  # 设置dpi为600
plt.plot(data, marker='o', markersize=2, markeredgewidth=1.5, linestyle='-', linewidth=1)
# plt.plot(data, marker='o', linestyle='-', linewidth=2)

# 标记第一次出现K值的地方
plt.axvline(x=first_occurrence, color='red', linestyle='--', linewidth=2, label=f'EarlyStop at epoch {first_occurrence + 1}')

# 设置标签
plt.xlabel('Epoch', weight='bold')
plt.ylabel('TSM', weight='bold')
plt.title(f'{args.dataset}: NetMamba Model', fontweight='bold')
plt.legend()

# 保存并显示图形，裁剪白边
plt.savefig(f'{args.dataset}_TSM_vs_epoch_K{args.K}.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
plt.show()
