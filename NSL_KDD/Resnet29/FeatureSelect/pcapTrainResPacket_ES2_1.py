# TensorFlow 2.9.0 compatible training script with early stopping
import time
import tensorflow as tf
import numpy as np
import csv, os, random
from pcapResnetPacketSeed import Resnet, Resnet2
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt

# 在import tensorflow之后添加
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 限制GPU显存占用为70%
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0.5 * 1024 * 1024 * 1024)]  # 假设GPU显存为16GB，这里限制为70%
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

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25
# 设置随机种子
set_deterministic_seed(SEED)

K = 1 # topk 特征
WIDTHLITMIT = 1024 # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
model.train_start = 0

for _ in range(TRAIN_EPOCH):
    delta_loss, count = model.train()
    model.epoch_count += 1
    if model.earlyStop == True:
        print("model.earlyStop == True")
        break

end_time = time.time()
total_training_time = end_time - start_time  # 计算训练总时长
print("TSMRecord—100", model.TSMRecord)
print("loss_history—100", model.loss_history)
print(f"Total training time: {total_training_time:.2f} seconds")

model_dir = "./model"
new_folder = "model_" + curr_time
os.mkdir(os.path.join(model_dir, new_folder))

model_path = os.path.join(model_dir, new_folder, "model.h5")

# TensorFlow 2.9.0 compatible model saving
model.model.save(model_path)

reTrainAccuracy_history = {} # 存储对应的 删掉特征数量和精度
scaling_factor_value = model.model.scaling_factor.numpy()
print('scaling_factor_value：', scaling_factor_value)
print('start testing...')

# 扁平化矩阵，并返回排序后的索引
sorted_indices = np.argsort(scaling_factor_value.flatten())[::-1]
print("sorted_indices：", sorted_indices)
model.test()
print("loss_history", model.loss_history)
print("macro_F1List", model.macro_F1List)
print("micro_F1List", model.micro_F1List)
print('starting retraining')  # 重新训练网络，只用 top_k_indices 对应的矩阵

'''choose top K feature '''
k = K
# 提取前 k 个特征的下标和因子值
top_k_indices = sorted_indices[:k]
print("K=", k, "top_k_indices", top_k_indices)
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
