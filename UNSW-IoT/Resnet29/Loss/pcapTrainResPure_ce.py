# TensorFlow 2.9.0 compatible ResNet training script for cross-entropy loss
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os, random
from pcapResnetPureSeed import ResNet
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

TRAIN_EPOCH = 30

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    # GPU configuration for TensorFlow 2.x
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 限制GPU显存占用为50%
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0.5 * 1024 * 1024 * 1024)]  # 假设GPU显存为16GB，这里限制为50%
            )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    
    curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    model = ResNet("ce", SEED, 0.9999, 1)
    start_time = time.time()
    
    print("开始训练ResNet模型（Cross-Entropy Loss）")
    
    for epoch in range(TRAIN_EPOCH):
        print(f"\n=== Epoch {epoch + 1}/{TRAIN_EPOCH} ===")
        micro_f1 = model.train()
        model.epoch_count += 1
        print(f"当前Micro-F1: {micro_f1:.4f}")
    
    end_time = time.time()
    total_training_time = end_time - start_time
    
    print(f"\n=== ResNet训练完成 ===")
    print(f"总训练时间: {total_training_time:.2f} 秒")
    print("Loss历史:", model.loss_history)
    print("Micro-F1历史:", model.micro_F1List)
    print("Macro-F1历史:", model.macro_F1List)
    
    # 保存模型
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    new_folder = "model_resnet_ce_" + curr_time
    model_save_dir = os.path.join(model_dir, new_folder)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    # 保存模型权重
    model_path = os.path.join(model_save_dir, "resnet_ce_model.h5")
    model.save_weights(model_path)
    print(f"ResNet模型已保存到: {model_path}")
    
    # 最终测试
    print("\n=== ResNet最终测试 ===")
    accuracy = model.test()
    print(f"ResNet最终准确率: {accuracy:.4f}")
    
    # 保存训练结果
    results = {
        'loss_history': model.loss_history,
        'micro_f1_history': model.micro_F1List,
        'macro_f1_history': model.macro_F1List,
        'final_accuracy': accuracy,
        'training_time': total_training_time,
        'seed': SEED,
        'epochs': TRAIN_EPOCH,
        'loss_type': 'ce'
    }
    
    import pickle
    result_path = os.path.join(model_save_dir, "training_results.pkl")
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"训练结果已保存到: {result_path}")
    
    print("\n=== 训练完成总结 ===")
    print(f"ResNet准确率: {accuracy:.4f}")
    print(f"最佳Micro-F1: {max(model.micro_F1List):.4f}")
    print(f"最佳Macro-F1: {max(model.macro_F1List):.4f}")
    print("训练完成！")
