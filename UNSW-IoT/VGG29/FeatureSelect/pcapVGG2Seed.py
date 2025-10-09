# TensorFlow 2.9.0 compatible VGG2 implementation for UNSW-IoT FeatureSelect
import gc
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from sklearn.metrics import f1_score
import numpy as np
import csv, random

# Enable eager execution (default in TF 2.x)
tf.config.run_functions_eagerly(True)

# 设置随机种子确保结果可复现
def set_deterministic_seed(seed):
    """设置所有随机种子确保结果可复现"""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

DATA_DIM = 72 # 特征
OUTPUT_DIM = 29
LEARNING_RATE = 0.0001
BETA = 0.999
GAMMA = 1
BATCH_SIZE = 128
TRAIN_FILE = '../../train_data.csv'
TEST_FILE = '../../test_data.csv'

MODEL_SAVE_PATH = './model/'
MODEL_SAVE_PATH2 = './model2/'
MODEL_NAME = 'model'
MODEL_NAME2 = 'model2'
top_k_values=[]
top_k_indice=[]

class VGGModel2(Model):
    def __init__(self, dim, selected_features=[], seed=25):
        super(VGGModel2, self).__init__()
        self.dim = dim
        self.selected_features = selected_features
        self.seed = seed
        
        # Build VGG architecture
        self.conv1_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                    name='conv_1',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.conv1_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                    name='conv_2',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_1')
        
        self.conv2_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                    name='conv_3',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.conv2_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                    name='conv_4',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_2')
        
        self.conv3_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                    name='conv_5',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.conv3_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                    name='conv_6',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.conv3_3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                    name='conv_7',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_3')
        
        self.conv4_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                                    name='conv_8',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024, activation='relu', name='fc_1',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.dropout1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(1024, activation='relu', name='fc_2',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.dropout2 = layers.Dropout(0.5)
        self.fc3 = layers.Dense(OUTPUT_DIM, name='fc_3',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
    
    def call(self, inputs, training=None):
        scaled_input = tf.reshape(inputs, [tf.shape(inputs)[0], self.dim, 1, 1])
        
        # VGG forward pass
        x = self.conv1_1(scaled_input)
        x = self.conv1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        
        x = self.conv4_1(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        return self.fc3(x)

class VGG2(): # 第二次训练
    def __init__(self,lossType,dim,selected_features=[],seed=25):  # 需输入维度 即当前特征数
        self.lossType = lossType
        set_deterministic_seed(seed)
        self.gamma = GAMMA
        self.dim = dim
        self.top_k_indices = selected_features
        self.seed = seed
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epoch_count = 0
        self.loss_history = []
        self.micro_F1List = []
        self.macro_F1List = []
        self.init_data()
        self.total_iterations_per_epoch = self.train_length // BATCH_SIZE

        # Create the model
        self.model = VGGModel2(dim=dim, selected_features=selected_features, seed=self.seed)
        
        # Setup optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Calculate class weights for CB loss
        beta = BETA
        ClassNum = len(self.label_status)
        effective_num = {}
        for key, value in self.label_status.items():
            new_value = (1.0 - beta) / (1.0 - np.power(beta, value))
            effective_num[key] = new_value
        
        total_effective_num = sum(effective_num.values())
        self.weights = {}
        for key, value in effective_num.items():
            new_value = effective_num[key] / total_effective_num * ClassNum
            self.weights[key] = new_value

    def compute_loss(self, y_true, y_pred):
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        class_weights = tf.constant([self.weights[str(i)] for i in range(len(self.weights))], dtype=tf.float32)
        sample_weights = tf.gather(class_weights, y_true)
        
        # 计算Focal Loss的modulator
        softmax_probs = tf.nn.softmax(y_pred)
        labels_one_hot = tf.one_hot(y_true, depth=len(self.weights))
        pt = tf.reduce_sum(labels_one_hot * softmax_probs, axis=1)
        modulator = tf.pow(1.0 - pt, self.gamma)
        focal_loss = modulator * ce
        cb_focal_loss = tf.multiply(focal_loss, sample_weights)
        
        if self.lossType == "ce":
            return tf.reduce_sum(ce)
        elif self.lossType == "cb":
            return tf.reduce_sum(tf.multiply(ce, sample_weights))
        elif self.lossType == "cb_focal_loss":
            return tf.reduce_sum(cb_focal_loss)

    @tf.function
    def train_step(self, data, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(data, training=True)
            loss = self.compute_loss(labels, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss

    def train(self):
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()
        
        # 随机打乱训练数据
        np.random.shuffle(self.train_data)
        
        for step in range(self.total_iterations_per_epoch):
            batch = self.get_a_train_batch(step)
            data, label = self.get_data_label(batch)
            
            loss = self.train_step(data, label)
            total_loss += loss.numpy()
            num_batches += 1
        
        average_loss = total_loss / num_batches
        self.loss_history.append(average_loss)
        
        micro_F1, macro_F1 = self.test2()
        self.micro_F1List.append(micro_F1)
        self.macro_F1List.append(macro_F1)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss:.6f}, duration: {epoch_duration:.2f} seconds')

    def get_a_train_batch(self, step):
        '''从训练数据集中获取一个批次的数据 '''
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.train_length:
            max_index = self.train_length
        return self.train_data[min_index:max_index]

    def get_data_label(self, batch):
        data = np.delete(batch, -1, axis=1)
        label = np.array(batch, dtype=np.int32)[:, -1]
        return data, label

    def init_data(self):
        self.train_data = []
        self.test_data = []
        self.label_status = {}

        # 初始化标签数据字典
        label_data = {i: [] for i in range(29)}

        # 读取训练数据
        filename = TRAIN_FILE
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data = []
                for i, char in enumerate(row):
                    if i in self.top_k_indices or i == len(row) - 1:
                        if char == 'None':
                            data.append(0)
                        else:
                            data.append(np.float32(char))

                label = int(data[-1])
                if label in label_data:
                    label_data[label].append(data)

                if str(label) not in self.label_status:
                    self.label_status[str(label)] = 0
                self.label_status[str(label)] += 1

        self.train_data = [item for sublist in label_data.values() for item in sublist]

        # 读取测试数据
        filename = TEST_FILE
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data = []
                for i, char in enumerate(row):
                    if i in self.top_k_indices or i == len(row) - 1:
                        if char == 'None':
                            data.append(0)
                        else:
                            data.append(np.float32(char))
                self.test_data.append(data)

        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)
        self.train_data = self.normalization(self.train_data)
        self.test_data = self.normalization(self.test_data)
         
        print('init data completed!')

    def normalization(self, minibatch):
        data = np.delete(minibatch, -1, axis=1)
        labels = np.array(minibatch, dtype=np.int64)[:, -1]
        mmax = np.max(data, axis=0)
        mmin = np.min(data, axis=0)
        for i in range(len(mmax)):
            if mmax[i] == mmin[i]:
                mmax[i] += 0.000001
        res = (data - mmin) / (mmax - mmin)
        res = np.c_[res, labels]
        return res

    def predict_top_k(self, x_feature, k=5):
        x_tensor = tf.constant([x_feature], dtype=tf.float32)
        predict = self.model(x_tensor, training=False)[0]
        top_k_indices = np.argsort(predict.numpy())[-k:][::-1]
        return top_k_indices

    def test(self):
        label_count = {}
        label_correct = {}
        length = len(self.test_data)
        count = 0
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        y_true = []
        y_pred = []
        
        for row in self.test_data:
            feature = row[0:-1]
            label = row[-1]
            count += 1
            x_feature = np.array(feature)
            top_k_labels = self.predict_top_k(x_feature, k=5)
            y_true.append(label)
            y_pred.append(top_k_labels[0])

            if str(int(label)) not in label_count:
                label_count[str(int(label))] = 0
                label_correct[str(int(label))] = 0
            if label == top_k_labels[0]:
                label_correct[str(int(label))] += 1
                top1_correct += 1
            if label in top_k_labels[:3]:
                top3_correct += 1
            if label in top_k_labels[:5]:
                top5_correct += 1
            label_count[str(int(label))] += 1
            if count % 10000 == 0:
                print(f"Processed {count} rows")

        accuracy1 = {}
        for label in sorted(label_count):
            accuracy1[label] = label_correct[label] / label_count[label]
            print(label, accuracy1[label], label_correct[label], label_count[label])
        
        top1_accuracy = top1_correct / length
        top3_accuracy = top3_correct / length
        top5_accuracy = top5_correct / length
        print(f"Top-1 accuracy: {top1_accuracy}")
        print(f"Top-3 accuracy: {top3_accuracy}")
        print(f"Top-5 accuracy: {top5_accuracy}")

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
        return top1_accuracy

    def test2(self):
        y_true = []
        y_pred = []
        for row in self.test_data:
            feature = row[0:-1]
            label = row[-1]
            x_feature = np.array(feature)
            top_k_labels = self.predict_top_k(x_feature, k=5)
            y_true.append(label)
            y_pred.append(top_k_labels[0])
        
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
        del y_true, y_pred
        gc.collect()
        return macro_f1, micro_f1
