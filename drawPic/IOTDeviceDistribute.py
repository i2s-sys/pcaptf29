import matplotlib.pyplot as plt

# 数据
devices = list(range(1, 30))  # 生成包含1到29的列表
percentages = [46.61, 17.61, 9.54, 8.94, 5.77, 2.61, 2.24, 1.89, 1.30, 0.94, 0.61, 0.35, 0.31, 0.01, 0.27, 0.24, 0.01, 0.01, 0.08, 0.01, 0.26, 0.11, 0.11, 0.07, 0.11, 0.04, 0.01, 0.01, 0.02]

# 创建条形图
plt.figure(figsize=(12, 6), dpi=600)
plt.bar(devices, percentages, color='lightblue')

# 添加每个条形图上的数值标签
for i, v in enumerate(percentages):
    plt.text(devices[i], v + 0.5, str(devices[i]), ha='center', fontsize=8, fontname='Times New Roman')

# 添加标签
plt.xlabel('IoT Device ID', fontsize=12, fontweight='bold', fontname='Times New Roman')
plt.ylabel('Percentage of Total Traffic Data', fontsize=12, fontweight='bold', fontname='Times New Roman')

# 去除白边
plt.tight_layout()

# 保存图片
plt.savefig('traffic_distribution.png', dpi=600, bbox_inches='tight')

# 显示图表
plt.show()
