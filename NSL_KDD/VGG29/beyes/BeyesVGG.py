# TensorFlow 2.9.0 compatible Bayesian optimization for VGG
import numpy as np
import random
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from VGG2 import VGG2
import tensorflow as tf

# 设置随机种子确保结果可复现
def set_deterministic_seed(seed):
    """设置所有随机种子确保结果可复现"""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

SEED = 20

# 设置随机种子
set_deterministic_seed(SEED)

# ========== Step 1: 加载数据 ==========
train_data = []
test_data = []
label_data = {i: [] for i in range(29)}

# 读取训练数据
with open('../../train_data.csv', 'r') as file:
    for line in file:
        row = line.strip().split(',')
        data = [float(x) if x != 'None' else 0.0 for x in row[:-1]]
        label = int(row[-1])
        train_data.append(data + [label])
        if label < 29:
            label_data[label].append(data + [label])

# 读取测试数据
with open('../../test_data.csv', 'r') as file:
    for line in file:
        row = line.strip().split(',')
        data = [float(x) if x != 'None' else 0.0 for x in row[:-1]]
        label = int(row[-1])
        test_data.append(data + [label])

print(f"训练数据: {len(train_data)} 条")
print(f"测试数据: {len(test_data)} 条")

# ========== Step 2: 定义目标函数 ==========
def objective(params):
    """
    目标函数：使用随机选择的特征子集训练VGG2模型
    返回负的Micro-F1分数（因为hyperopt最小化目标函数）
    """
    try:
        # 解析参数
        num_features = int(params['num_features'])
        
        # 随机选择特征
        all_features = list(range(41))  # 假设有41个特征
        selected_features = random.sample(all_features, num_features)
        selected_features.sort()
        
        print(f"尝试特征数量: {num_features}, 选择的特征: {selected_features}")
        
        # 创建VGG2模型
        model = VGG2(
            lossType="cb_focal_loss",
            dim=num_features,
            selected_features=selected_features,
            seed=SEED
        )
        
        # 训练模型
        epochs = 10  # 减少epoch数以加快搜索速度
        for epoch in range(epochs):
            model.train()
            model.epoch_count += 1
        
        # 评估模型
        micro_f1, macro_f1 = model.test()
        
        print(f"特征数量: {num_features}, Micro-F1: {micro_f1:.4f}, Macro-F1: {macro_f1:.4f}")
        
        # 清理内存
        del model
        tf.keras.backend.clear_session()
        
        # 返回负的Micro-F1（因为hyperopt最小化）
        return {'loss': -micro_f1, 'status': STATUS_OK}
        
    except Exception as e:
        print(f"错误: {e}")
        return {'loss': 0, 'status': STATUS_OK}

# ========== Step 3: 定义搜索空间 ==========
space = {
    'num_features': hp.choice('num_features', [1, 2, 4, 8, 16, 32])
}

# ========== Step 4: 执行贝叶斯优化 ==========
def run_bayesian_optimization():
    """运行贝叶斯优化"""
    print("开始贝叶斯优化...")
    
    trials = Trials()
    
    # 执行优化
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=20,  # 最大评估次数
        trials=trials,
        verbose=True
    )
    
    print(f"最佳参数: {best}")
    
    # 获取最佳结果
    best_trial = trials.best_trial
    best_loss = best_trial['result']['loss']
    best_micro_f1 = -best_loss
    
    print(f"最佳Micro-F1: {best_micro_f1:.4f}")
    
    return best, best_micro_f1, trials

# ========== Step 5: 主函数 ==========
if __name__ == "__main__":
    # GPU配置
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # 运行贝叶斯优化
    best_params, best_score, trials = run_bayesian_optimization()
    
    print("\n========== 优化结果 ==========")
    print(f"最佳参数: {best_params}")
    print(f"最佳Micro-F1分数: {best_score:.4f}")
    
    # 保存结果
    import pickle
    with open('bayesian_optimization_results.pkl', 'wb') as f:
        pickle.dump({
            'best_params': best_params,
            'best_score': best_score,
            'trials': trials
        }, f)
    
    print("结果已保存到 bayesian_optimization_results.pkl")
    print("贝叶斯优化完成！")
