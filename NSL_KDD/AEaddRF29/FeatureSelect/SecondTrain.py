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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# InfFS_S 方法选择的特征
# features = [4,5,0,22,32,31,23,15,12,2,3,9,11,16,33,25,24,38,37,28,1,27,26,40,35,39,30,7,34,29,36,18] # infs 32
# features = [35, 27, 31, 25, 23, 22, 15, 16, 8, 6, 1, 9, 12, 14, 5, 38, 39, 3, 21, 36, 32, 13, 34, 10, 2, 20, 4, 18, 11, 26, 0, 17]   # pso
# features = [31, 22, 24, 26, 10, 35, 8, 21, 25, 2, 1, 12, 4, 32, 3, 16, 9, 28, 17, 34, 15, 14, 20, 27, 13, 23, 5, 0, 37, 38, 39, 40] # sca
# features = [39, 38, 36, 32, 28, 26, 31, 18, 10, 15, 16, 2, 22, 11, 6, 12, 19, 33, 8, 29, 14, 20, 17, 9, 35, 21, 34, 5, 37, 4, 24, 1] # fpa
features = [35,1,7,39,11,9,34,32,28,38,22,37,24,33,23,2,3,27,26,31,40,25,36,21,30,29,0,6,10,17,8,18,16,4,5,12,19,15,14,20,13] # factor
k=16
if __name__ == '__main__':
    # GPU configuration for TensorFlow 2.x
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # 在实际使用中，这些特征应该从第一阶段训练的结果中获取
    selected_features =features[:k]   # inf_fs_s方法
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
