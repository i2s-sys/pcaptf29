# 测试没有早停策略所选特征的 模型精度
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os

from pcapResnet2 import Resnet2
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt

file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25
WIDTHLITMIT = 256 # 位宽限制加大
TRAIN_EPOCH = 30
# selected_features = [1] # 无早停特征
selected_features = [1, 24]
# selected_features = [1, 24, 35, 40]
# selected_features = [1, 24, 35, 40, 0, 27, 30, 33]
# selected_features = [1, 24, 35, 40, 0, 27, 30, 33, 23, 2, 26, 8, 32, 38, 39, 20]
# selected_features = [1, 24, 35, 40, 0, 27, 30, 33, 23, 2, 26, 8, 32, 38, 39, 20, 17, 25, 10, 9, 19, 34, 3, 4]
# selected_features = [1, 24, 35, 40, 0, 27, 30, 33, 23, 2, 26, 8, 32, 38, 39, 20, 17, 25, 10, 9, 19, 34, 3, 4, 15, 11, 36, 31, 37, 21, 16, 22]

# selected_features = [9] # 早停策略下的特征


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

feature_widths = [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 1, 32, 32, 32, 32, 1]
scaling_factor_value = [1.1003, 1.2429, 1.0352, 0.9458, 0.9355, 0.8145, -0.0000, 0.4196, 1.0261, 0.9696, 0.9778, 0.9273, 0.6965, 0.5279, 0.6262, 0.9291, 0.8593, 0.9931, 0.7850, 0.9671, 1.0083, 0.8682, 0.8565, 1.0461, 1.1745, 0.9787, 1.0313, 1.0808, 0.7615, 0.6339, 1.0587, 0.8804, 1.0260, 1.0531, 0.9667, 1.1717, 0.9172, 0.8776, 1.0257, 1.0139, 1.1097, 0.5797]

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
    # top_k_values = np.array(scaling_factor_value)[selected_features] # 获取对应因子值
    # top_k_weights = np.array(feature_widths)[selected_features]  # 获取对应特征的位宽
    # top_k_values = np.insert(top_k_values, 0, -1)
    # top_k_weights = np.insert(top_k_weights, 0, -1)
    # print("top_k_values",top_k_values)
    # print("top_k_weights",top_k_weights)
    # selected_indices = knapsack(top_k_values, top_k_weights, WIDTHLITMIT)
    # selected_features = [selected_features[i - 1] for i in selected_indices]
    # 无背包算法 如下
    print("len",len(selected_features),"f WIDTHLITMIT: {WIDTHLITMIT}  selected_features: ", selected_features)
    model2 = Resnet2(dim=len(selected_features), selected_features=selected_features, seed=SEED)
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
    print("dnn2_macro_F1List", model2.macro_F1List)
    print("dnn2_micro_F1List", model2.micro_F1List)
    print('start testing...')
    # 获取验证集最佳模型
    saver.restore(sess, best_model_path)
    accuracy2 = model2.test()