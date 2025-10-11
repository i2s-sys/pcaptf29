# TensorFlow 2.9.0 compatible second training script for UNSW-NB VGG
import argparse
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapVGG2 import VGG2
import matplotlib.pyplot as plt

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 减少TensorFlow日志输出

# 配置GPU内存增长
def configure_gpu():
    """配置GPU设置"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU配置成功: {len(gpus)} 个GPU")
            return True
        else:
            print("未检测到GPU，将使用CPU")
            return False
    except Exception as e:
        print(f"GPU配置失败: {e}")
        return False

# 执行GPU配置
gpu_available = configure_gpu()

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25
K = 16  # topk 特征
TRAIN_EPOCH = 30

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

# 使用预定义的特征选择结果
sorted_indices = [20,21,11,12,6,8,7,29,17,15,18,16,27,26,4,19,22,10,5,9,13,14,1,35,40,30,39,32,33,0,34,2]
top_k_indices = sorted_indices[:K]
print("K=",K,"top_k_indices",top_k_indices)
selected_features = top_k_indices

vgg2 = VGG2("ce",dim=len(selected_features), selected_features=selected_features, seed=SEED)
print('start retraining...')

model_dir = "./model"
new_folder = "model_" + curr_time
best_accuracy = 0.0  # 初始化最高accuracy
best_model_path = None  # 初始化最佳模型路径
new_folder2 = new_folder + "best_model"
os.makedirs(os.path.join(model_dir, new_folder2), exist_ok=True)

start_time = time.time()
for _ in range(TRAIN_EPOCH):
    accuracy = vgg2.train()
    vgg2.epoch_count += 1
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = os.path.join(model_dir, new_folder2, f"model_best_epoch_{vgg2.epoch_count}.h5")
        vgg2.model.save_weights(best_model_path)
end_time = time.time()
total_training_time = end_time - start_time
print("vgg2_loss_history", vgg2.loss_history)
print('start testing...')
vgg2.test()
if best_model_path:
    vgg2.model.load_weights(best_model_path)
    print('Best Model testing...')
    vgg2.test()  # 现在测试的是最佳模型