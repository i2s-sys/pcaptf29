# TensorFlow 2.9.0 compatible AutoEncoder + RandomForest for selected features (UNSW-NB)
import time
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
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

DATA_DIM = 42
OUTPUT_DIM = 2
BETA = 0.999
GAMMA = 1
TRAIN_FILE = '../../train_data2.csv'
TEST_FILE = '../../test_data2.csv'
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
MODEL_SAVE_PATH = '../model/'
MODEL_SAVE_PATH2 = './model2/'
MODEL_NAME = 'model'
MODEL_NAME2 = 'model2'
KEEP_PROB = 0.5
top_k_values=[]
top_k_indice=[]

class AEModel2(Model):
    def __init__(self, dim, seed=None):
        super(AEModel2, self).__init__()
        self.dim = dim
        self.seed = seed
        
        # Encoder layers
        self.encoder1 = layers.Dense(128, activation='relu', 
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.encoder2 = layers.Dense(64, activation='relu',
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        
        # Decoder layers
        self.decoder1 = layers.Dense(128, activation='relu',
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.decoder2 = layers.Dense(dim,
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
    
    def call(self, inputs, training=None):
        # Encoder
        encoded = self.encoder1(inputs)
        encoded = self.encoder2(encoded)
        
        # Decoder
        decoded = self.decoder1(encoded)
        decoded = self.decoder2(decoded)
        
        return decoded, encoded

class AE2():
    def __init__(self, dim, selected_features, seed=25):
        set_deterministic_seed(seed)
        self.dim = dim
        self.top_k_indices = selected_features
        self.seed = seed
        self.loss_history = []
        self.micro_F1List = []
        self.macro_F1List = []
        self.intersection_sets = []
        self.TSMRecord = []
        self.batch_size = BATCH_SIZE
        self.earlyStop = False
        self.learning_rate = LEARNING_RATE
        self.epoch_count = 0
        
        self.init_data()
        self.total_iterations_per_epoch = self.train_length // BATCH_SIZE
        
        # Create the model
        self.model = AEModel2(dim=dim, seed=self.seed)
        
        # Setup optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compute_loss(self, inputs, outputs):
        return tf.sqrt(tf.reduce_mean(tf.square(inputs - outputs)))

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            decoded, encoded = self.model(data, training=True)
            loss = self.compute_loss(data, decoded)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, encoded

    def train_classifier(self):
        num_batches = 0
        epoch_start_time = time.time()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        all_encoded_features = []
        all_labels = []
        
        for step in range(self.total_iterations_per_epoch):
            batch = self.get_a_train_batch(step)
            data, label = self.getBatch_data_label(batch)
            
            _, encoded_feature = self.train_step(data)
            encoded_feature = encoded_feature.numpy()
            encoded_feature = np.squeeze(encoded_feature)
            all_encoded_features.append(encoded_feature)
            all_labels.append(label)
            num_batches += 1
        
        # 在所有批次数据累积后训练分类器
        all_encoded_features = np.vstack(all_encoded_features)
        all_labels = np.hstack(all_labels)
        self.classifier.fit(all_encoded_features, all_labels)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'duration: {epoch_duration:.2f} seconds')
        self.test_classifier()

    def test_classifier(self):
        all_encoded_features = []
        all_labels = []
        label_count = {}
        label_correct = {}
        self.test_iterations = self.test_length // BATCH_SIZE
        
        for step in range(self.test_iterations):
            batch = self.get_a_test_batch(step)
            data, label = self.getBatch_data_label(batch)
            
            _, encoded_feature = self.model(data, training=False)
            encoded_feature = encoded_feature.numpy()
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
        
        for label in sorted(label_count):
            accuracy = label_correct[label] / label_count[label]
            print(f'Label {label}: Accuracy {accuracy:.2f} ({label_correct[label]}/{label_count[label]})')
        
        macro_f1 = f1_score(all_labels, predictions, average='macro')
        micro_f1 = f1_score(all_labels, predictions, average='micro')
        print(f'Macro-F1: {macro_f1}')
        print(f'Micro-F1: {micro_f1}')

    def train(self):
        self.epoch_rmse = []
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()
        
        # 随机打乱训练数据
        np.random.shuffle(self.train_data)
        
        for step in range(self.total_iterations_per_epoch):
            batch = self.get_a_train_batch(step)
            data, label = self.getBatch_data_label(batch)
            
            loss, _ = self.train_step(data)
            self.epoch_rmse.append(loss.numpy())
            total_loss += loss.numpy()
            num_batches += 1
        
        average_loss = total_loss / self.train_length
        self.loss_history.append(average_loss)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss:.6f}, duration: {epoch_duration:.2f} seconds')
        return average_loss

    def getBatch_data_label(self, batch):
        data = np.delete(batch, -1, axis=1)
        label = np.array(batch, dtype=np.int32)[:, -1]
        return data, label

    def get_a_train_batch(self, step):
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.train_length:
            max_index = self.train_length
        return self.train_data[min_index:max_index]

    def get_a_test_batch(self, step):
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.test_length:
            max_index = self.test_length
        return self.test_data[min_index:max_index]

    def init_data(self):
        self.train_data = []
        self.test_data = []
        self.label_status = {}
        num_labels = OUTPUT_DIM
        label_data = [[] for _ in range(num_labels)]
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
                label_data[label].append(data)
                if str(label) not in self.label_status:
                    self.label_status[str(label)] = 0
                self.label_status[str(label)] += 1
        self.train_data = [item for sublist in label_data for item in sublist]
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
