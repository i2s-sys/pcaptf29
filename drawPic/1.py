import matplotlib.pyplot as plt

# 数据
years = [2020, 2021, 2022, 2023, 2024, 2025]
graduates = [874, 909, 1076, 1158, 1179, 1222]

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(years, graduates, marker='o', color='red')

# 添加标题和标签
plt.title('高校毕业生人数（2020-2025）', color='red')
plt.xlabel('年份', color='red')
plt.ylabel('人数（万）', color='red')

# 修改刻度标签的颜色
plt.xticks(color='red')
plt.yticks(color='red')

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
