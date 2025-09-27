# TensorFlow 2.9.0 compatible training script for FeatureSelect2
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os, random
from pcapResnetSeed_factor import Resnet
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt

# 设置随机种子确保结果可复现
def set_deterministic_seed(seed):
    """设置所有随机种子确保结果可复现"""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25
# 设置随机种子
set_deterministic_seed(SEED)

K = 32 # topk 特征
WIDTHLITMIT = 1024 # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

feature_widths = [
    32, 32, 32, 32,  # fiat_mean, fiat_min, fiat_max, fiat_std
    32, 32, 32, 32,  # biat_mean, biat_min, biat_max, biat_std
    32, 32, 32, 32,  # diat_mean, diat_min, diat_max, diat_std
    32,              # duration 13
    64, 32, 32, 32, 32,  # fwin_total, fwin_mean, fwin_min, fwin_max, fwin_std
    64, 32, 32, 32, 32,  # bwin_total, bwin_mean, bwin_min, bwin_max, bwin_std
    64, 32, 32, 32, 32,  # dwin_total, dwin_mean, dwin_min, dwin_max, dwin_std
    16, 16, 16,         # fpnum, bpnum, dpnum
    32, 32, 32, 32,         # bfpnum_rate, fpnum_s, bpnum_s, dpnum_s 22
    64, 32, 32, 32, 32,  # fpl_total, fpl_mean, fpl_min, fpl_max, fpl_std
    64, 32, 32, 32, 32,  # bpl_total, bpl_mean, bpl_min, bpl_max, bpl_std
    64, 32, 32, 32, 32,  # dpl_total, dpl_mean, dpl_min, dpl_max, dpl_std
    32, 32, 32, 32,         # bfpl_rate, fpl_s, bpl_s, dpl_s  19
    16, 16, 16, 16, 16, 16, 16, 16,  # fin_cnt, syn_cnt, rst_cnt, pst_cnt, ack_cnt, urg_cnt, cwe_cnt, ece_cnt
    16, 16, 16, 16,     # fwd_pst_cnt, fwd_urg_cnt, bwd_pst_cnt, bwd_urg_cnt
    16, 16, 16,         # fp_hdr_len, bp_hdr_len, dp_hdr_len
    32, 32, 32          # f_ht_len, b_ht_len, d_ht_len 18
]

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model = Resnet(K, ES_THRESHOLD, SEED)
start_time = time.time()

for _ in range(TRAIN_EPOCH):
    delta_loss, count = model.train()
    model.epoch_count += 1

end_time = time.time()
total_training_time = end_time - start_time  # 计算训练总时长
model.test()
print("TSMRecord—100", model.TSMRecord)
print("loss_history—100", model.loss_history)
print(f"Total training time: {total_training_time:.2f} seconds")

model_dir = "./model"
new_folder = "model_" + curr_time
os.mkdir(os.path.join(model_dir, new_folder))

model_path = os.path.join(model_dir, new_folder, "model.h5")

# TensorFlow 2.9.0 compatible model saving
model.model.save(model_path)

scaling_factor_value = model.model.scaling_factor.numpy()
print('scaling_factor_value：', scaling_factor_value)
print('start testing...')

# 扁平化矩阵，并返回排序后的索引
sorted_indices = np.argsort(scaling_factor_value.flatten())[::-1]

'''choose top K feature '''
K_values = [1, 2, 4, 8, 16, 24, 32]
for K in K_values:
    top_k_indices = sorted_indices[:K]
    print(f"K = {K}, selected_features = {top_k_indices}")
