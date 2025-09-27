# 使用早停策略
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from AEAddRF2 import AE2 # 导入DNN类
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")
SEED = 25
K = 32 # topk 特征
WIDTHLITMIT = 1024 # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 3

# selected_features = [11]
# selected_features = [11, 9]
# selected_features = [11, 9, 12,  1]
# selected_features = [11, 9, 12,  1,  7, 21, 20, 61]
# selected_features = [11, 9, 12,  1,  7, 21, 20, 61, 17, 15, 74, 72, 71, 65, 18, 77]
# selected_features = [11, 9, 12,  1,  7, 21, 20, 61, 17, 15, 74, 72, 71, 65, 18, 77, 53, 88, 37, 101,  83,  46,  57,  64]
selected_features = [11, 9, 12,  1,  7, 21, 20, 61, 17, 15, 74, 72, 71, 65, 18, 77, 53, 88, 37, 101,  83,  46,  57,  64,  32,  94,  79,  60, 108,  67, 114,  80]


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)
curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

if __name__ == '__main__':
    print("selected_features",selected_features)
    model2 = AE2(dim=len(selected_features), selected_features=selected_features,seed=SEED)
    start_time = time.time()
    for _ in range(TRAIN_EPOCH):
        model2.train()
        model2.epoch_count += 1
    model2.train_classifier()
    end_time = time.time()  # 记录训练结束时间
    total_training_time = end_time - start_time  # 计算训练总时长