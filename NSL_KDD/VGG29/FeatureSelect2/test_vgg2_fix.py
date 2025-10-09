# 测试修复后的VGG2训练逻辑
import sys
import os
sys.path.append('NSL_KDD/VGG29/FeatureSelect2')

import tensorflow as tf
import numpy as np
import random

# 设置随机种子
def set_deterministic_seed(seed):
    """设置所有随机种子确保结果可复现"""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# 创建测试数据
def create_test_data():
    """创建测试数据"""
    train_data = []
    test_data = []

    # 生成1000个训练样本
    for i in range(1000):
        # 41个特征 + 1个标签
        features = np.random.rand(41).astype(np.float32)
        label = np.random.randint(0, 2)
        train_data.append(np.concatenate([features, [label]]))
    
    # 生成200个测试样本
    for i in range(200):
        features = np.random.rand(41).astype(np.float32)
        label = np.random.randint(0, 2)
        test_data.append(np.concatenate([features, [label]]))
    
    return train_data, test_data

def test_vgg2_training():
    """测试VGG2训练逻辑"""
    print("=== 测试VGG2训练逻辑 ===")
    
    # 设置随机种子
    SEED = 25
    set_deterministic_seed(SEED)
    
    # 创建临时数据文件
    train_data, test_data = create_test_data()
    
    # 保存临时数据文件
    import csv
    with open('NSL_KDD/train_data_test.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in train_data:
            writer.writerow(row)
    
    with open('NSL_KDD/test_data_test.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in test_data:
            writer.writerow(row)
    
    try:
        # 导入VGG2模型
        from pcapVGGSeed import VGG2
        
        # 选择前16个特征
        selected_features = list(range(16))
        
        # 创建VGG2模型
        vgg2 = VGG2("ce", dim=len(selected_features), selected_features=selected_features, seed=SEED)
        
        print(f"模型创建成功，使用{len(selected_features)}个特征")
        print(f"训练数据大小: {len(vgg2.train_data)}")
        print(f"测试数据大小: {len(vgg2.test_data)}")
        
        # 训练几个epoch
        print("\n开始训练测试...")
        for epoch in range(5):
            print(f"\n--- Epoch {epoch + 1} ---")
            delta_loss, count = vgg2.train()
            vgg2.epoch_count += 1
            
            print(f"Loss变化: {delta_loss:.6f}")
            print(f"稳定计数: {count}")
            
            # 检查F1分数是否在变化
            if vgg2.micro_F1List:
                print(f"当前Micro-F1: {vgg2.micro_F1List[-1]:.4f}")
            if vgg2.macro_F1List:
                print(f"当前Macro-F1: {vgg2.macro_F1List[-1]:.4f}")
        
        print("\n=== 训练结果分析 ===")
        print(f"Loss历史: {vgg2.loss_history}")
        print(f"Micro-F1历史: {vgg2.micro_F1List}")
        print(f"Macro-F1历史: {vgg2.macro_F1List}")
        
        # 检查是否有变化
        if len(vgg2.loss_history) > 1:
            loss_changes = [abs(vgg2.loss_history[i] - vgg2.loss_history[i-1]) for i in range(1, len(vgg2.loss_history))]
            print(f"Loss变化幅度: {loss_changes}")
        
        if len(vgg2.micro_F1List) > 1:
            micro_f1_changes = [abs(vgg2.micro_F1List[i] - vgg2.micro_F1List[i-1]) for i in range(1, len(vgg2.micro_F1List))]
            print(f"Micro-F1变化幅度: {micro_f1_changes}")
        
        if len(vgg2.macro_F1List) > 1:
            macro_f1_changes = [abs(vgg2.macro_F1List[i] - vgg2.macro_F1List[i-1]) for i in range(1, len(vgg2.macro_F1List))]
            print(f"Macro-F1变化幅度: {macro_f1_changes}")
        
        print("\n✅ 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理临时文件
        try:
            os.remove('NSL_KDD/train_data_test.csv')
            os.remove('NSL_KDD/test_data_test.csv')
        except:
            pass

if __name__ == "__main__":
    test_vgg2_training()

