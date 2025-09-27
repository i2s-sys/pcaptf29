import matplotlib.pyplot as plt
import numpy as np

# 定义数据
x_labels = [1, 2,  4, 8, 16, 24, 32, 72]
y_values = [0.5813, 0.7650, 0.9392, 0.9674, 0.9810, 0.9868, 0.9880, 0.9917]

# 处理 y_values，将其乘以100
y_values = [y * 100 for y in y_values]

# 创建新的x轴坐标，保持等间距
x_positions = np.arange(len(x_labels))
# 设置字体
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"

# 绘制图形
plt.figure(figsize=(8, 6))
plt.plot(x_positions, y_values, marker='o')

# 设置新的x轴刻度标签
plt.xticks(x_positions, x_labels)

# 在每个数据点上标出值
for i, value in enumerate(y_values):
    plt.text(x_positions[i], value, f'{value:.2f}', ha='right', va='bottom')

# 设置图形标题和标签
plt.title('Resnet')
plt.xlabel('Number of features(K)')
plt.ylabel('Acc(%)')

# 设置y轴范围
plt.ylim(min(y_values) - 10, max(y_values) + 10)

# 显示图形
plt.show()
