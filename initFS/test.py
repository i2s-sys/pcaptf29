import numpy as np
from InfFS_S import inf_fs_s

data = np.loadtxt("../../datasets/UNSW_iot/train_data.csv", delimiter=",")
X = data[:, :-1]
Y = data[:, -1]

# alpha 系数（权重和为 1）
alpha = [0.33, 0.33, 0.34]

# 运行 Inf-FS
RANKED, WEIGHT, SUBSET = inf_fs_s(X, Y, alpha)

print("特征重要性排序 (前 K):", RANKED[:16])
print("对应权重 (前 K):", WEIGHT[RANKED[:16]])
print("选择的特征子集:", SUBSET)
# [50 23 18 13 45 40 35 26 16 21 31 12 19 14 20 24 15 27 17 22 25 10  9 68 66 67 38 43 41 36 42 47]  32
# [50 23 18 13 45 40 35 26 16 21 31 12 19 14 20 24 15 27 17 22 25 10  9 68] 24
# [50 23 18 13 45 40 35 26 16 21 31 12 19 14 20 24] 16


# 法2 实例
# import numpy as np
# from InfFS_U import inf_fs_u
#
# data = np.loadtxt("../../datasets/UNSW_iot/train_data.csv", delimiter=",")
#
# X = data[:, :72]
# # 运行无监督 Inf-FS
# RANKED, WEIGHT, SUBSET = inf_fs_u(X, alpha=0.5, verbose=1)
#
# print("特征重要性排序 (前 K):", RANKED[:16])
# print("对应权重 (前 K):", WEIGHT[RANKED[:16]])
# print("选择的特征子集:", SUBSET)

# [50 60 61 25 56  8 48 12 54 42 15 37  9 55 53 35 20 49 36 38 51 33 45 32 43 40 41  5 31 52 44 22]  32
# [50 60 61 25 56  8 48 12 54 42 15 37  9 55 53 35 20 49 36 38 51 33 45 32]  24
# [50 60 61 25 56  8 48 12 54 42 15 37  9 55 53 35] 16
# [50 60 61 25 56  8 48 12]  8

