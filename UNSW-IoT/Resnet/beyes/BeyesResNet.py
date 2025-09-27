import time
import numpy as np
import csv
import os
import itertools
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from Resnet2 import Resnet2
import tensorflow as tf
SEED = 20

# 设置随机种子
np.random.seed(SEED)

# ========== Step 1: 加载数据 ==========
train_data = []
test_data = []
label_data = {i: [] for i in range(29)}

# 读取训练数据
filename = '../../train_data.csv'
csv_reader = csv.reader(open(filename))
for row in csv_reader:
    data = [0 if char == 'None' else np.float32(char) for char in row]
    label = int(data[-1])
    label_data[label].append(data)

train_data = sum(label_data.values(), [])
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 读取测试数据
filename = '../../test_data.csv'
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)
# 定义目标函数
TRAIN_EPOCH = 30
def objective(params):
    n_features_to_select = 32  # 固定选择32个特征

    # 随机选择特征子集
    selected_indices = np.random.choice(X_train.shape[1], n_features_to_select, replace=False)
    curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    model2 = Resnet2(dim=len(selected_indices), selected_features=selected_indices, seed=SEED)
    model_dir = "./model"
    best_accuracy = 0.0  # 初始化最高accuracy
    best_model_path = None
    new_folder = "model_" + curr_time
    new_folder2 = new_folder + "best_model"
    os.mkdir(os.path.join(model_dir, new_folder2))
    os.mkdir(os.path.join(model_dir, new_folder))
    saver = tf.compat.v1.train.Saver()
    for _ in range(TRAIN_EPOCH):
        accuracy = model2.train()
        model2.epoch_count += 1
        # 保存验证集最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(model_dir, new_folder2, f"model_best_epoch_{model2.epoch_count}.ckpt")
            saver.save(model2.sess, best_model_path)
    saver.restore(sess, best_model_path)
    micro_f1, macro_f1  = model2.test()

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