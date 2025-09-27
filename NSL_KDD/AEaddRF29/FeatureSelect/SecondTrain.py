# TensorFlow 2.9.0 compatible second phase training script for AE + Random Forest
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os, random
from AEAddRF2 import AE2
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
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # 示例：使用前16个最重要的特征
    # 在实际使用中，这些特征应该从第一阶段训练的结果中获取
    selected_features = list(range(16))  # 选择前16个特征作为示例
    print(f"选择的特征: {selected_features}")
    
    curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    model = AE2(dim=len(selected_features), selected_features=selected_features, seed=SEED)
    start_time = time.time()
    
    print(f"开始第二阶段训练AutoEncoder + Random Forest模型（使用{len(selected_features)}个选定特征）")
    
    for epoch in range(TRAIN_EPOCH):
        print(f"\n=== Epoch {epoch + 1}/{TRAIN_EPOCH} ===")
        micro_f1 = model.train()
        model.epoch_count += 1
        print(f"当前Micro-F1: {micro_f1:.4f}")
    
    end_time = time.time()
    total_training_time = end_time - start_time
    
    print(f"\n=== 第二阶段AutoEncoder训练完成 ===")
    print(f"总训练时间: {total_training_time:.2f} 秒")
    print("Loss历史:", model.loss_history)
    print("Micro-F1历史:", model.micro_F1List)
    print("Macro-F1历史:", model.macro_F1List)
    
    # 保存模型
    model_dir = "./model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    new_folder = "model_ae2_" + curr_time
    model_save_dir = os.path.join(model_dir, new_folder)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    # 保存模型权重
    model_path = os.path.join(model_save_dir, "ae2_model.h5")
    model.save_weights(model_path)
    print(f"第二阶段AutoEncoder模型已保存到: {model_path}")
    
    # 最终测试AutoEncoder
    print("\n=== 第二阶段AutoEncoder最终测试 ===")
    ae_accuracy = model.test()
    print(f"第二阶段AutoEncoder最终准确率: {ae_accuracy:.4f}")
    
    # 训练Random Forest
    print("\n=== 开始训练Random Forest ===")
    rf_accuracy, rf_macro_f1, rf_micro_f1 = model.train_random_forest()
    
    # 保存训练结果
    results = {
        'loss_history': model.loss_history,
        'micro_f1_history': model.micro_F1List,
        'macro_f1_history': model.macro_F1List,
        'ae_final_accuracy': ae_accuracy,
        'rf_accuracy': rf_accuracy,
        'rf_macro_f1': rf_macro_f1,
        'rf_micro_f1': rf_micro_f1,
        'selected_features': selected_features,
        'feature_count': len(selected_features),
        'training_time': total_training_time,
        'seed': SEED,
        'epochs': TRAIN_EPOCH
    }
    
    import pickle
    result_path = os.path.join(model_save_dir, "second_phase_results.pkl")
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"第二阶段训练结果已保存到: {result_path}")
    
    print("\n=== 第二阶段训练完成总结 ===")
    print(f"使用特征数量: {len(selected_features)}")
    print(f"AutoEncoder准确率: {ae_accuracy:.4f}")
    print(f"Random Forest准确率: {rf_accuracy:.4f}")
    print(f"Random Forest Macro-F1: {rf_macro_f1:.4f}")
    print(f"Random Forest Micro-F1: {rf_micro_f1:.4f}")
    print("第二阶段训练完成！")
