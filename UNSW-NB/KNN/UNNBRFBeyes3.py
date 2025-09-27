import numpy as np
import csv
import os
import itertools
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

SEED = 20

# 设置随机种子
np.random.seed(SEED)

# ========== Step 1: 加载数据 ==========
train_data = []
test_data = []
label_data = {i: [] for i in range(29)}

# 读取训练数据
filename = '../train_data2.csv'
csv_reader = csv.reader(open(filename))
for row in csv_reader:
    data = [0 if char == 'None' else np.float32(char) for char in row]
    label = int(data[-1])
    label_data[label].append(data)

train_data = sum(label_data.values(), [])

# 读取测试数据
filename = '../test_data2.csv'
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
def objective(params):
    n_features_to_select = 32  # 固定选择32个特征

    # 随机选择特征子集
    selected_indices = np.random.choice(X_train.shape[1], n_features_to_select, replace=False)
    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]

    # 训练模型
    clf = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=42
    )
    clf.fit(X_train_sel, y_train)

    # 评估模型
    y_pred = clf.predict(X_test_sel)
    accuracy = accuracy_score(y_test, y_pred)
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    # 输出每次试验的精度和F1分数
    print(f"Trial: Accuracy = {accuracy:.4f}, Micro-F1 = {micro_f1:.4f}, Macro-F1 = {macro_f1:.4f}, Selected Features: {n_features_to_select}, n_estimators: {params['n_estimators']}, max_depth: {params['max_depth']}")

    # 返回负的Micro-F1分数（因为Hyperopt默认是minimize）
    return {'loss': -micro_f1, 'status': STATUS_OK}

# 定义参数空间
param_space = {
    'n_estimators': hp.choice('n_estimators', [50, 100, 200]),  # 随机森林的树的数量
    'max_depth': hp.choice('max_depth', [None, 10, 20, 30])  # 随机森林的最大深度
}

# 运行优化
trials = Trials()
best = fmin(
    fn=objective,
    space=param_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)

# 输出最优参数
best_n_estimators = best['n_estimators']
best_max_depth = best['max_depth']
print(f"Best number of estimators: {best_n_estimators}")
print(f"Best max depth: {best_max_depth}")

# 输出固定选择的特征数量
n_features_to_select = 32
print(f"Fixed number of features: {n_features_to_select}")

# 输出最优特征子集
best_indices = np.random.choice(X_train.shape[1], n_features_to_select, replace=False)
print(f"Best feature indices: {best_indices}")