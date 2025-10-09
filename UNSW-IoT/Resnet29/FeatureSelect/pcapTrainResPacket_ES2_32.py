import sys
import time
import tensorflow as tf
import numpy as np
import csv, os, random
from pcapResnetPacketSeed import Resnet
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
WIDTHLITMIT = 1024  # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 3

if __name__ == '__main__':
    # GPU configuration for TensorFlow 2.x
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    K = 32  # 特征选择数量
    print(f"K = {K}, ES_THRESHOLD = {ES_THRESHOLD}")

    # 创建ResNet模型进行特征选择
    resnet = Resnet(K, ES_THRESHOLD, SEED)
    print('start training...')
    start_time = time.time()

    for epoch in range(TRAIN_EPOCH):
        print(f"\n=== Epoch {epoch + 1}/{TRAIN_EPOCH} ===")
        delta_loss, count = resnet.train()
        resnet.epoch_count += 1
        
        # 显示训练进度
        if epoch > 0:
            print(f"Loss change: {delta_loss:.6f}, Stable count: {count}")
        
        # 早停检查
        if resnet.earlyStop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    end_time = time.time()
    total_training_time = end_time - start_time
    
    print(f"\n=== 训练完成 ===")
    print(f"总训练时间: {total_training_time:.2f} 秒")
    print(f"训练轮数: {resnet.epoch_count}")
    print("Loss历史:", resnet.loss_history)
    print("Micro-F1历史:", resnet.micro_F1List)
    print("Macro-F1历史:", resnet.macro_F1List)
    print("TSM记录:", resnet.TSMRecord)
    
    # 最终测试
    print("\n=== 最终测试 ===")
    macro_f1, micro_f1 = resnet.test2()
    
    print(f"\n=== 训练总结 ===")
    print(f"最终Macro-F1: {macro_f1:.4f}")
    print(f"最终Micro-F1: {micro_f1:.4f}")
    if resnet.micro_F1List:
        print(f"最佳Micro-F1: {max(resnet.micro_F1List):.4f}")
    if resnet.macro_F1List:
        print(f"最佳Macro-F1: {max(resnet.macro_F1List):.4f}")
    print("训练完成！")