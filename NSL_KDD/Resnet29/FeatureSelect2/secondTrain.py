# TensorFlow 2.9.0 compatible training script with early stopping
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapResnetPacketSeed import Resnet2
import matplotlib.pyplot as plt

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

WIDTHLITMIT = 1024 # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 3

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

# sorted_indices = [4,5,0,22,32,31,23,15,12,2,3,9,11,16,33,25,24,38,37,28,1,27,26,40,35,39,30,7,34,29,36,18] # infs 32
sorted_indices = [35, 27, 31, 25, 23, 22, 15, 16, 8, 6, 1, 9, 12, 14, 5, 38, 39, 3, 21, 36, 32, 13, 34, 10, 2, 20, 4, 18, 11, 26, 0, 17]   # pso
# sorted_indices = [31, 22, 24, 26, 10, 35, 8, 21, 25, 2, 1, 12, 4, 32, 3, 16, 9, 28, 17, 34, 15, 14, 20, 27, 13, 23, 5, 0, 37, 38, 39, 40] # sca
# sorted_indices = [39, 38, 36, 32, 28, 26, 31, 18, 10, 15, 16, 2, 22, 11, 6, 12, 19, 33, 8, 29, 14, 20, 17, 9, 35, 21, 34, 5, 37, 4, 24, 1] # fpa
# sorted_indices = [36,39,9,23,22,40,34,0,32,38,2,10,29,37,16,31,1,7,30,18,33,21,17,27,24,3,26,13,25,35,6,14] # factor
K = 32 # topk 特征

top_k_indices = sorted_indices[:K]
selected_features = top_k_indices
dnn2 = Resnet2(dim=len(selected_features), selected_features=selected_features, seed=SEED)
print('start retraining...')

start_time = time.time()
for _ in range(TRAIN_EPOCH):
    delta_loss, count = dnn2.train()
    dnn2.epoch_count += 1
end_time = time.time()
total_training_time = end_time - start_time
print("dnn2_loss_history", dnn2.loss_history)
print("dnn2_macro_F1List", dnn2.macro_F1List)
print("dnn2_micro_F1List", dnn2.micro_F1List)
print('start testing...')
accuracy2 = dnn2.test()
