# TensorFlow 2.9.0 compatible training script with early stopping - K=4
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os, random
from pcapResnetPacketSeed import Resnet, Resnet2
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

SEED = 25
# 设置随机种子
set_deterministic_seed(SEED)
K = 4 # topk 特征
TRAIN_EPOCH = 30
ES_THRESHOLD = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model = Resnet(K, ES_THRESHOLD, SEED)
start_time = time.time()

for _ in range(TRAIN_EPOCH):
    delta_loss, count = model.train()
    model.epoch_count += 1
    if model.earlyStop == True:
        print("model.earlyStop == True")
        break

end_time = time.time()
total_training_time = end_time - start_time
print("TSMRecord—100", model.TSMRecord)
print("loss_history—100", model.loss_history)
print(f"Total training time: {total_training_time:.2f} seconds")

model_dir = "./model"
new_folder = "model_" + curr_time
os.mkdir(os.path.join(model_dir, new_folder))

model_path = os.path.join(model_dir, new_folder, "model.h5")
model.model.save(model_path)

scaling_factor_value = model.model.scaling_factor.numpy()
print('scaling_factor_value：', scaling_factor_value)
print('start testing...')

sorted_indices = np.argsort(scaling_factor_value.flatten())[::-1]
print("sorted_indices：", sorted_indices)
model.test()
print("loss_history", model.loss_history)
print("macro_F1List", model.macro_F1List)
print("micro_F1List", model.micro_F1List)
print('starting retraining')

k = K
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
