# 纯resnet 就检查全部特征的时候 resnet的准确率
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from LSTMPure import LSTM # 导入DNN类
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

DATA_DIM = 72 # 特征数
# 对原始的训练和测试数据进行处理，如有必要的话进行数值化 从original_train_data -> train_data
def handle_data():
    source_file = 'origin_train_data.csv'
    handled_file = 'train_data.csv'  # write to csv file
    data_file = open(handled_file, 'w', newline='')
    csv_writer = csv.writer(data_file)
    with open(source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        next(csv_reader)
        for row in csv_reader:
            csv_writer.writerow(row)
        data_file.close()
    test_source_file = 'origin_test_data.csv'
    test_handled_file = 'test_data.csv'  # write to csv file
    test_data_file = open(test_handled_file, 'w', newline='')
    test_csv_writer = csv.writer(test_data_file)
    with open(test_source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        next(csv_reader)
        for row in csv_reader:
            test_csv_writer.writerow(row)
        test_data_file.close()
    print('pre process completed!')

# 创建一个DNN对象
# handle_data()
# 获取当前时间，并格式化为字符串
curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model = LSTM()
start_time = time.time()
# 训练模型
for _ in range(50):  # 调用train函数，并获取损失的变化
    delta_loss, count = model.train()  # 判断损失的变化是否小于阈值
    model.epoch_count += 1
    # if count >= 8:
    #     print("count >= 8: 连续5个epoch满足条件 变化不大 相差小于0.03")
    #     break
end_time = time.time()  # 记录训练结束时间
total_training_time = end_time - start_time  # 计算训练总时长
model_dir = "./model"
new_folder = "model_" + curr_time
os.mkdir(os.path.join(model_dir, new_folder))
model_path = os.path.join(model_dir, new_folder, "model.ckpt")
saver = tf.compat.v1.train.Saver()
with model.sess as sess:
    saver.save(sess, model_path)
    print('start testing...')
    accuracy = model.test()
