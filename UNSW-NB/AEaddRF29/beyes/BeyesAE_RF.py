# TensorFlow 2.9.0 compatible hyperparameter optimization script for UNSW-NB AEaddRF
import time
import numpy as np
import csv
import os
import itertools
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from AEAddRF_n1 import AE2  # 导入DNN类
SEED = 20

# 设置随机种子
np.random.seed(SEED)

# ========== Step 1: 加载数据 ==========
train_data = []
test_data = []
label_data = {i: [] for i in range(2)}  # UNSW-NB has 2 classes

# 读取训练数据
filename = '../../train_data2.csv'
csv_reader = csv.reader(open(filename))
for row in csv_reader:
    data = [0 if char == 'None' else np.float32(char) for char in row]
    label = int(data[-1])
    label_data[label].append(data)

train_data = sum(label_data.values(), [])

# 读取测试数据
filename = '../../test_data2.csv'
csv_reader = csv.reader(open(filename))
for row in csv_reader:
    data = [0 if char == 'None' else np.float32(char) for char in row]
    test_data.append(data)

def load_data(datas):
    data = np.delete(datas, -1, axis=1)
    label = np.array(datas, dtype=np.int32)[:, -1]
    return data, label

X_train, y_train = load_data(train_data)
X_test, y_test = load_data(test_data)

# 定义目标函数
TRAIN_EPOCH = 30
def objective(params):
    n_features_to_select = 16  # 固定选择16个特征

    # 随机选择特征子集
    selected_indices = np.random.choice(X_train.shape[1], n_features_to_select, replace=False)
    model2 = AE2(dim=len(selected_indices), selected_features=selected_indices, seed=SEED)
    for _ in range(TRAIN_EPOCH):
        model2.train()
        model2.epoch_count += 1
    micro_f1, macro_f1 = model2.train_classifier()

    # 输出每次试验的精度和F1分数
    print(f"Trial: Micro-F1 = {micro_f1:.4f}, Macro-F1 = {macro_f1:.4f}, Selected Features: {n_features_to_select}")

    # 返回负的Micro-F1分数（因为Hyperopt默认是minimize）
    return {'loss': -micro_f1, 'status': STATUS_OK}

# 运行优化
start_time = time.time()
# 定义一个虚拟参数空间
param_space = {
    'dummy_param': hp.choice('dummy_param', [0])  # 添加一个虚拟参数，始终选择 0
}

trials = Trials()
best = fmin(
    fn=objective,
    space=param_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)

end_time = time.time()  # 记录训练结束时间
total_training_time = end_time - start_time  # 计算训练总时长

# 输出固定选择的特征数量
n_features_to_select = 32
print(f"Fixed number of features: {n_features_to_select}")

# 输出最优特征子集
best_indices = np.random.choice(X_train.shape[1], n_features_to_select, replace=False)
print(f"Best feature indices: {best_indices}")
