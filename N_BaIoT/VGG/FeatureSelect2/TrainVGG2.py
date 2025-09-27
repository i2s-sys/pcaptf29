import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapVGG2 import VGG2 # 导入DNN类
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25
K = 32 # topk 特征
WIDTHLITMIT = 256 # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 3

feature_widths = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 1, 32, 32, 32, 32, 1]
scaling_factor_value =[0.7447, 1.0233, 0.9861, 0.8669, 0.9567, 0.9841, 0.9846, 1.0279, 0.9290, 1.0344, 0.9683, 1.0186, 0.8808, 0.9699, 1.0498, 0.9065, 0.9164, 0.8329, 0.8468, 0.9555, 0.9980, 1.0388, 0.9941, 0.9978, 1.0094, 1.0151, 1.0314, 0.9703, 1.0369, 0.9842, 0.8988, 1.0236, 0.0000, 0.9655, -0.0000, 0.2991, 0.9227, 0.7514, 0.8145, 0.0000, 0.8729, -0.0000, 0.9873, 1.0174, 0.5974, 0.8277, 0.0000, 0.8581, -0.0000, 0.7543, 0.9793, 0.9880, 0.9524, 0.3231, 0.9546, -0.0000, 0.3809, 0.8821, 0.7938, 0.8449, 0.0000, 0.9036, 0.0000, 0.0995, 0.6316, 0.0000, 0.9180, 0.2482, -0.0000, 0.9699, 0.4735, 0.0001, 0.9298, 0.7183, 0.7230, 0.9296, 0.3862, 0.7575, 0.8514, 0.7408, 0.0000, 0.8033, 0.0000, 0.8230, -0.0000, -0.0000, 0.1521, 0.0000, 0.4759, 0.0000, 0.7052, -0.0000, -0.0000, -0.0000, -0.0000, 0.6618, -0.0000, 0.8872, 0.0000, 0.1011, 0.1881, 0.7461, 0.8885, 0.0000, 0.8395, 0.0000, -0.0000, 0.0000, -0.0000, 0.3246, -0.0000, 0.4183, 0.0000, 0.0000, 0.0000]
selected_features = [1]  # 无早停

# selected_features = [39] 早停

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

def knapsack(values, weights, max_weight):
    n = len(values) - 1 # 真实物品的个数
    dp = np.zeros((n + 1, max_weight + 1))
    keep = np.empty((n + 1, max_weight + 1), dtype=object)
    for i in range(n + 1):
        for j in range(max_weight + 1):
            keep[i][j] = []

    for i in range(1, n + 1):
        for j in range(0, max_weight + 1):
            dp[i][j] = dp[i - 1][j]
            keep[i][j] = keep[i - 1][j].copy()
            if(j >= weights[i]):
                if(dp[i - 1][j - weights[i]] + values[i] > dp[i][j]):
                    dp[i][j] = dp[i - 1][j - weights[i]] + values[i]
                    keep[i][j] = keep[i - 1][j - weights[i]].copy()
                    keep[i][j].append(i)
    total_weight = sum(weights[i] for i in keep[n][max_weight])
    print("total_weight",total_weight,"keep[n][max_weight]",keep[n][max_weight])
    return keep[n][max_weight]

model_dir = "./model"
curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
if __name__ == '__main__':
    # 加入背包部分
    top_k_values = np.array(scaling_factor_value)[selected_features]  # 获取对应因子值
    top_k_weights = np.array(feature_widths)[selected_features]  # 获取对应特征的位宽
    top_k_values = np.insert(top_k_values, 0, -1)
    top_k_weights = np.insert(top_k_weights, 0, -1)
    print("top_k_values", top_k_values)
    print("top_k_weights", top_k_weights)
    selected_indices = knapsack(top_k_values, top_k_weights, WIDTHLITMIT) # 背包中的下标是从1开始的 所以下面得用i-1
    selected_features = [selected_features[i - 1] for i in selected_indices]
    # 无背包算法如下：
    print("nums:",len(selected_features),"selected_features",selected_features)
    model2 = VGG2("cb_focal_loss",dim=len(selected_features), selected_features=selected_features,seed=SEED)
    print('start retraining...')
    start_time = time.time()
    # 验证集参数初始化
    best_accuracy = 0.0  # 初始化最高accuracy
    best_model_path = None
    new_folder = "model_" + curr_time
    new_folder2 = new_folder + "best_model"
    os.mkdir(os.path.join(model_dir, new_folder2))
    os.mkdir(os.path.join(model_dir, new_folder))
    saver = tf.compat.v1.train.Saver()
    for _ in range(TRAIN_EPOCH):
        accuracy = model2.train()
        model2.epoch_count += 1
        # 保存验证集最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(model_dir, new_folder2, f"model_best_epoch_{model2.epoch_count}.ckpt")
            saver.save(model2.sess, best_model_path)
    end_time = time.time()
    total_training_time = end_time - start_time
    print("dnn2_loss_history", model2.loss_history)
    # 获取验证集最佳模型
    saver.restore(sess, best_model_path)
    accuracy2 = model2.test()