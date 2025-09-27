# from FeatureExtractor import *
import csv
import time
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import f1_score

from KitNET.KitNET import KitNET

TRAIN_FILE = '../train_data2.csv'
TEST_FILE = '../test_data2.csv'
OUTPUT_DIM = 2
BATCH_SIZE = 1
DATA_DIM = 42

# maxAE = 10 #maximum size for any autoencoder in the ensemble layer
# FMgrace = 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
# ADgrace = 50000 #the number of instances used to train the anomaly detector (ensemble itself)
# 将原本的FE修改成 数据集特征
# 已调整参数：maxAE、learning_rate、FM_grace_period、AD_grace_period、最后一个尝试：增加epoch

class Kitsune:
    def __init__(self,learning_rate=0.0001,hidden_ratio=0.75):
        #init packet feature extractor (AfterImage)
        # self.FE = FE(file_path,limit)
        max_autoencoder_size = 10
        self.batch_size = BATCH_SIZE
        self.init_data()
        FM_grace_period = self.train_length // 3
        AD_grace_period = self.train_length - FM_grace_period - 1
        self.AnomDetector = KitNET(DATA_DIM,max_autoencoder_size,FM_grace_period,AD_grace_period,learning_rate,hidden_ratio)

    def trainSAE(self):
        RMSE = []
        total_loss = 0
        epoch_start_time = time.time()  # 记录epoch开始时间
        for i in range(self.train_length):
            data = np.delete(self.train_data[i], -1)
            self.AnomDetector.process(data)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f' duration: {epoch_duration:.2f} seconds')
        # 测试阶段
        for i in range(self.test_length):
            data = np.delete(self.test_data[i], -1)
            rmse = self.AnomDetector.process(data)
            RMSE.append(rmse)
            total_loss += rmse
            average_loss = total_loss / self.train_length
        print(f'testData average loss: {average_loss}')
        return RMSE

    def train_classifier(self):
        self.train_start = 0
        epoch_start_time = time.time()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        all_encoded_features = []
        all_labels = []
        for i in range(self.train_length):
            data = np.delete(self.train_data[i], -1)
            label = self.train_data[i][-1]
            encoded_feature = data
            # encoded_feature = self.AnomDetector.getEncode(data)
            # encoded_feature = self.AnomDetector.getReconstruct(data)
            encoded_feature = np.squeeze(encoded_feature)
            all_encoded_features.append(encoded_feature)
            all_labels.append(label)
        # 在所有批次数据累积后训练分类器
        all_encoded_features = np.vstack(all_encoded_features)
        all_labels = np.hstack(all_labels)
        self.classifier.fit(all_encoded_features, all_labels)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'train classifier duration: {epoch_duration:.2f} seconds')
        self.test_classifier()

    def test_classifier(self):
        self.train_start = 0
        all_encoded_features = []
        all_labels = []
        label_count = {}
        label_correct = {}
        for i in range(self.test_length):
            data = np.delete(self.test_data[i], -1)
            label = self.train_data[i][-1]
            encoded_feature = data
            # encoded_feature = self.AnomDetector.getEncode(data)
            # encoded_feature = self.AnomDetector.getReconstruct(data)
            encoded_feature = np.squeeze(encoded_feature)
            all_encoded_features.append(encoded_feature)
            all_labels.append(label)
        all_encoded_features = np.vstack(all_encoded_features)
        all_labels = np.hstack(all_labels)
        predictions = self.classifier.predict(all_encoded_features)
        # 统计每个类的正确预测次数和总预测次数
        for true_label, pred_label in zip(all_labels, predictions):
            true_label_str = str(int(true_label))
            if true_label_str not in label_count:
                label_count[true_label_str] = 0
                label_correct[true_label_str] = 0
            label_count[true_label_str] += 1
            if true_label == pred_label:
                label_correct[true_label_str] += 1

        # 计算并打印每个类的准确率
        for label in sorted(label_count):
            accuracy = label_correct[label] / label_count[label]
            print(f'Label {label}: Accuracy {accuracy:.2f} ({label_correct[label]}/{label_count[label]})')

        macro_f1 = f1_score(all_labels, predictions, average='macro')
        micro_f1 = f1_score(all_labels, predictions, average='micro')
        print(f'Macro-F1: {macro_f1}')
        print(f'Micro-F1: {micro_f1}')

    def get_a_train_batch(self, step):
        '''从训练数据集中获取一个批次的数据 '''
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.train_length:
            max_index = self.train_length
        return self.train_data[min_index:max_index]  # 返回min到max之间的数据样本

    def init_data(self):
        self.train_data = []
        self.test_data = []  # init train and test data
        self.label_status = {}
        filename = TRAIN_FILE
        csv_reader = csv.reader(open(filename))
        label_data = [[] for _ in range(OUTPUT_DIM)]
        for row in csv_reader:
            data = []
            for char in row:
                if char == 'None':
                    data.append(0)
                else:
                    data.append(np.float32(char))  # 将数据从字符串格式转换为float32格式
            label = int(data[-1])
            if label < OUTPUT_DIM:
                label_data[label].append(data)
            if self.label_status.get(str(label), 0) > 0:
                self.label_status[str(label)] += 1
            else:
                self.label_status[str(label)] = 1
        self.train_data = sum(label_data, [])
        filename = TEST_FILE
        csv_reader = csv.reader(open(filename))
        for row in csv_reader:
            data = []
            for char in row:
                if char == 'None':
                    data.append(0)
                else:
                    data.append(np.float32(char))  # 将数据从字符串格式转换为float32格式
            self.test_data.append(data)
        # self.train_data = self.normalization(self.train_data)
        # self.test_data = self.normalization(self.test_data)
        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)

        print('init data completed!')

    def normalization(self, minibatch):
        data = np.delete(minibatch, -1, axis=1)
        labels = np.array(minibatch, dtype=np.int64)[:, -1]
        mmax = np.max(data, axis=0)
        mmin = np.min(data, axis=0)
        for i in range(len(mmax)):
            if mmax[i] == mmin[i]:
                mmax[i] += 0.000001  # avoid getting devided by 0
        res = (data - mmin) / (mmax - mmin)
        res = np.c_[res, labels]
        return res