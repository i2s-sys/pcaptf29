# TensorFlow 2.9.0 compatible VGG training script with early stopping
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os, random
from pcapVGGSeed import VGG, VGG2
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

K = 1  # topk 特征
WIDTHLITMIT = 1024  # 位宽限制加大
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

if __name__ == '__main__':
    # GPU configuration for TensorFlow 2.x
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    model = VGG(K, "cb_focal_loss", ES_THRESHOLD, SEED)
    start_time = time.time()
    
    for epoch in range(TRAIN_EPOCH):
        model.train()
        model.epoch_count += 1
        # Early stopping logic can be implemented here if needed
    
    end_time = time.time()
    total_training_time = end_time - start_time
    print("TSMRecord—100", model.TSMRecord)
    print("loss_history—100", model.loss_history)
    print(f"Total training time: {total_training_time:.2f} seconds")
    
    # Save model
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    new_folder = "model_" + curr_time
    model_save_dir = os.path.join(model_dir, new_folder)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    # Save model weights
    model_path = os.path.join(model_save_dir, "model.h5")
    model.save_weights(model_path)
    
    reTrainAccuracy_history = {}
    scaling_factor_value = model.scaling_factor.numpy()
    print('scaling_factor_value：', scaling_factor_value)
    print('start testing...')
    
    # Get sorted indices
    sorted_indices = np.argsort(scaling_factor_value.flatten())[::-1]
    print("sorted_indices：", sorted_indices)
    
    print("loss_history", model.loss_history)
    print('starting retraining')
    k = K
    
    # Extract top k features
    top_k_indices = sorted_indices[:k]
    print("K=", k, "top_k_indices", top_k_indices)
    selected_features = top_k_indices
    
    # Second phase training with selected features
    vgg2 = VGG2("cb_focal_loss", dim=len(selected_features), selected_features=selected_features, seed=SEED)
    print('start retraining...')
    start_time = time.time()
    
    for epoch in range(TRAIN_EPOCH):
        vgg2.train()
        vgg2.epoch_count += 1
    
    end_time = time.time()
    total_training_time = end_time - start_time
    print("vgg2_loss_history", vgg2.loss_history)
    print("vgg2_macro_F1List", vgg2.macro_F1List)
    print("vgg2_micro_F1List", vgg2.micro_F1List)
    accuracy2 = vgg2.test()
    
    print(f"Final accuracy: {accuracy2}")
    print("Training completed successfully!")
