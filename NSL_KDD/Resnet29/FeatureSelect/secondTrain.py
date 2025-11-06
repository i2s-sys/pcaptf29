# TensorFlow 2.9.0 compatible training script with early stopping
import time
import tensorflow as tf
import numpy as np
import csv, os, random
from pcapResnetPacketSeed import Resnet2

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 在import tensorflow之后添加
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # 限制GPU显存占用为70%
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10 * 1024 * 1024 * 1024)]  # 假设GPU显存为16GB，这里限制为70%
        )
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
# 设置随机种子确保结果可复现
def set_deterministic_seed(seed):
    """设置所有随机种子确保结果可复现"""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
SEED = 25
# 设置随机种子
set_deterministic_seed(SEED)


WIDTHLITMIT = 1024 # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 3
# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

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

'''choose top K feature '''
K = 24 # topk 特征
k = K
# 提取前 k 个特征的下标和因子值
# features = [4,5,0,22,32,31,23,15,12,2,3,9,11,16,33,25,24,38,37,28,1,27,26,40,35,39,30,7,34,29,36,18] # infs 32
# features = [35, 27, 31, 25, 23, 22, 15, 16, 8, 6, 1, 9, 12, 14, 5, 38, 39, 3, 21, 36, 32, 13, 34, 10, 2, 20, 4, 18, 11, 26, 0, 17]   # pso
features = [31, 22, 24, 26, 10, 35, 8, 21, 25, 2, 1, 12, 4, 32, 3, 16, 9, 28, 17, 34, 15, 14, 20, 27, 13, 23, 5, 0, 37, 38, 39, 40] # sca
# features = [39, 38, 36, 32, 28, 26, 31, 18, 10, 15, 16, 2, 22, 11, 6, 12, 19, 33, 8, 29, 14, 20, 17, 9, 35, 21, 34, 5, 37, 4, 24, 1] # fpa


top_k_indices = features[:K]
print(f"K={K}, top_k_indices={top_k_indices}")
selected_features = top_k_indices
model2 = Resnet2(dim=len(selected_features), selected_features=selected_features, seed=SEED)
print('start retraining...')

start_time = time.time()
for _ in range(TRAIN_EPOCH):
    delta_loss, count = model2.train()
    model2.epoch_count += 1
end_time = time.time()
total_training_time = end_time - start_time
print("dnn2_loss_history", model2.loss_history)
print("dnn2_macro_F1List", model2.macro_F1List)
print("dnn2_micro_F1List", model2.micro_F1List)
print('start testing...')
accuracy2 = model2.test()
