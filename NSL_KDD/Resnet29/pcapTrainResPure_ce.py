# TensorFlow 2.9.0 compatible training script for pure ResNet with CE loss
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os, random
from pcapResnetPureSeed import Resnet
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 25
# 设置随机种子
set_deterministic_seed(SEED)
EPOCH_NUM = 30

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model = Resnet("ce", SEED, 0.9999, 1)
start_time = time.time()

for _ in range(EPOCH_NUM):
    delta_loss, count, micro_f1 = model.train()
    model.epoch_count += 1

end_time = time.time()
total_training_time = end_time - start_time

model_dir = "./model"
new_folder = "model_" + curr_time
os.mkdir(os.path.join(model_dir, new_folder))

model_path = os.path.join(model_dir, new_folder, "model.h5")

# TensorFlow 2.9.0 compatible model saving
model.model.save(model_path)

print("loss_history", model.loss_history)
print("macro_F1List", model.macro_F1List)
print("micro_F1List", model.micro_F1List)

print('start testing...')
accuracy = model.test()
