# TensorFlow 2.9.0 compatible training script with configurable loss for UNSW-NB VGG
import argparse
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapVGGSeed_loss import VGG
import matplotlib.pyplot as plt

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
K = 32  # topk 特征
WIDTHLITMIT = 1024  # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 3

parser = argparse.ArgumentParser(description='Specify number of features.')
parser.add_argument('--Q', type=int, default=3, help='Number of ES_THRESHOLD')
parser.add_argument('--lossType', type=str, default='cb', help='')
parser.add_argument('--beta', type=float, default=0.999, help='')
parser.add_argument('--gamma', type=float, default=2, help='')
args = parser.parse_args()

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
model = VGG(K, args.lossType, args.Q, SEED, args.beta, args.gamma)
start_time = time.time()

model_dir = "./model"
new_folder = "model_" + curr_time
os.makedirs(os.path.join(model_dir, new_folder), exist_ok=True)

best_accuracy = 0.0  # 初始化最高accuracy
best_model_path = None  # 初始化最佳模型路径
new_folder2 = new_folder + "best_model"
os.makedirs(os.path.join(model_dir, new_folder2), exist_ok=True)

for epoch in range(TRAIN_EPOCH):
    accuracy = model.train()
    model.epoch_count += 1
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = os.path.join(model_dir, new_folder2, f"model_best_epoch_{model.epoch_count}.ckpt")
        model.model.save_weights(best_model_path)

end_time = time.time()
total_training_time = end_time - start_time
print("TSMRecord—100", model.TSMRecord)
print("loss_history—100", model.loss_history)
print(f"Total training time: {total_training_time:.2f} seconds")

model_path = os.path.join(model_dir, new_folder, "model.ckpt")
scaling_factor_value = model.model.scaling_factor.numpy()
print('scaling_factor_value：', scaling_factor_value)

if best_model_path:
    model.model.load_weights(best_model_path)
    print('best Model testing...')
    model.test()
