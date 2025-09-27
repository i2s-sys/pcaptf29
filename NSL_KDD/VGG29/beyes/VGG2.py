# TensorFlow 2.9.0 compatible VGG2 implementation
import time
import tensorflow as tf
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

DATA_DIM = 41
OUTPUT_DIM = 2
LEARNING_RATE = 0.0001
BETA = 0.9999
GAMMA = 2
BATCH_SIZE = 128
TRAIN_FILE = '../../train_data.csv'
TEST_FILE = '../../test_data.csv'

MODEL_SAVE_PATH = './model/'
MODEL_SAVE_PATH2 = './model2/'
MODEL_NAME = 'model'
MODEL_NAME2 = 'model2'
top_k_values=[]
top_k_indice=[]

def set_seed(seed):
    """设置随机种子（保持向后兼容）"""
    set_deterministic_seed(seed)

class VGGBlock(layers.Layer):
    def __init__(self, filters, kernel_size, activation='relu', padding='same', name_prefix='conv', seed=25):
        super(VGGBlock, self).__init__()
        self.conv = layers.Conv2D(
            filters, kernel_size, 
            activation=activation, 
            padding=padding,
            name=name_prefix,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        )
    
    def call(self, inputs, training=None):
        return self.conv(inputs)

class VGG2(Model):
    def __init__(self, lossType, dim, selected_features=[], seed=25):
        super(VGG2, self).__init__()
        self.lossType = lossType
        set_seed(seed)
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
        
        # Build VGG architecture
        self.build_vgg(dim)
        
        # Class balancing setup
        beta = BETA
        ClassNum = len(self.label_status)
        effective_num = {}
        for key, value in self.label_status.items():
            new_value = (1.0 - beta) / (1.0 - np.power(beta, value))
            effective_num[key] = new_value
        
        total_effective_num = sum(effective_num.values())
        self.class_weights_dict = {}
        for key, value in effective_num.items():
            new_value = effective_num[key] / total_effective_num * ClassNum
            self.class_weights_dict[key] = new_value
        
        # Setup optimizer and loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.build_loss_fn()
    
    def build_vgg(self, NEW_DIM):
        """Build VGG architecture"""
        # VGG blocks
        self.conv1_1 = VGGBlock(64, (3, 3), name_prefix='conv_1', seed=self.seed)
        self.conv1_2 = VGGBlock(64, (3, 3), name_prefix='conv_2', seed=self.seed)
        self.pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_1')
        
        self.conv2_1 = VGGBlock(128, (3, 3), name_prefix='conv_3', seed=self.seed)
        self.conv2_2 = VGGBlock(128, (3, 3), name_prefix='conv_4', seed=self.seed)
        self.pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_2')
        
        self.conv3_1 = VGGBlock(256, (3, 3), name_prefix='conv_5', seed=self.seed)
        self.conv3_2 = VGGBlock(256, (3, 3), name_prefix='conv_6', seed=self.seed)
        self.conv3_3 = VGGBlock(256, (3, 3), name_prefix='conv_7', seed=self.seed)
        self.pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_3')
        
        self.conv4_1 = VGGBlock(512, (3, 3), name_prefix='conv_8', seed=self.seed)
        
        # Fully connected layers
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024, activation='relu', name='fc_1',
                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        self.dropout1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(1024, activation='relu', name='fc_2',
                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        self.dropout2 = layers.Dropout(0.5)
        self.fc3 = layers.Dense(OUTPUT_DIM, name='fc_3',
                               kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
    
    def call(self, inputs, training=None):
        # Reshape input for VGG
        x = tf.reshape(inputs, [-1, self.dim, 1, 1])
        
        # VGG forward pass
        x = self.conv1_1(x, training=training)
        x = self.conv1_2(x, training=training)
        x = self.pool1(x)
        
        x = self.conv2_1(x, training=training)
        x = self.conv2_2(x, training=training)
        x = self.pool2(x)
        
        x = self.conv3_1(x, training=training)
        x = self.conv3_2(x, training=training)
        x = self.conv3_3(x, training=training)
        x = self.pool3(x)
        
        x = self.conv4_1(x, training=training)
        
        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        x = self.fc3(x)
        
        return x
    
    def build_loss_fn(self):
        """Build loss function based on loss type"""
        self.class_weights = tf.constant([self.class_weights_dict[str(i)] for i in range(len(self.class_weights_dict))], dtype=tf.float32)
    
    def compute_loss(self, y_true, y_pred):
        """Compute loss based on loss type"""
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        
        if self.lossType == "ce":
            return tf.reduce_mean(ce)
        elif self.lossType == "cb":
            sample_weights = tf.gather(self.class_weights, y_true)
            cbce = tf.multiply(ce, sample_weights)
            return tf.reduce_mean(cbce)
        elif self.lossType == "cb_focal_loss":
            # Class-balanced focal loss
            sample_weights = tf.gather(self.class_weights, y_true)
            softmax_probs = tf.nn.softmax(y_pred)
            labels_one_hot = tf.one_hot(y_true, depth=len(self.class_weights_dict))
            pt = tf.reduce_sum(labels_one_hot * softmax_probs, axis=1)
            modulator = tf.pow(1.0 - pt, self.gamma)
            focal_loss = modulator * ce
            cb_focal_loss = tf.multiply(focal_loss, sample_weights)
            return tf.reduce_mean(cb_focal_loss)
    
    @tf.function
    def train_step(self, x, y):
        """Single training step"""
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compute_loss(y, predictions)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss, predictions
    
    def train(self):
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()
        
        # Shuffle training data
        np.random.shuffle(self.train_data)
        
        for step in range(self.total_iterations_per_epoch):
            batch = self.get_a_train_batch(step)
            data, label = self.get_data_label(batch)
            
            loss, predictions = self.train_step(data, label)
            total_loss += loss
            num_batches += 1
        
        average_loss = total_loss / num_batches
        self.loss_history.append(float(average_loss))
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss:.6f}, duration: {epoch_duration:.2f} seconds')
        
        micro_F1, macro_F1 = self.test2()
        return micro_F1
    
    def get_a_train_batch(self, step):
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
        self.valid_data = []
        self.label_status = {}
        label_data = {i: [] for i in range(OUTPUT_DIM)}
        
        def process_row(row):
            data = [0 if char == 'None' else np.float32(char) for i, char in enumerate(row) if
                    i in self.top_k_indices or i == len(row) - 1]
            label = int(data[-1])
            label_data[label].append(data)
            self.label_status[str(label)] = self.label_status.get(str(label), 0) + 1
        
        with open(TRAIN_FILE, mode="r", encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header if exists
            for row in csv_reader:
                process_row(row)
        self.train_data = [data for label in label_data.values() for data in label]
        
        # Processing test file and splitting data
        test_data_temp = {i: [] for i in range(OUTPUT_DIM)}
        with open(TEST_FILE, mode="r", encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                data = [0 if char == 'None' else np.float32(char) for i, char in enumerate(row) if
                        i in self.top_k_indices or i == len(row) - 1]
                label = int(data[-1])
                test_data_temp[label].append(data)
        
        # Splitting test data into test and validation sets
        for label, data in test_data_temp.items():
            split_idx = int(len(data) * 2 / 3)
            self.test_data.extend(data[:split_idx])
            self.valid_data.extend(data[split_idx:])
        
        self.train_data = self.normalization(self.train_data)
        self.test_data = self.normalization(self.test_data)
        self.valid_data = self.normalization(self.valid_data)
        np.random.shuffle(self.train_data)
        
        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)
        self.valid_length = len(self.valid_data)
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
    
    def get_a_test_batch(self, step):
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.test_length:
            max_index = self.test_length
        return self.test_data[min_index:max_index]
    
    def get_a_valid_batch(self, step):
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.valid_length:
            max_index = self.valid_length
        return self.valid_data[min_index:max_index]
    
    def predict_top_k(self, x_features, k=5):
        predicts = self(x_features, training=False)
        top_k_indices = [np.argsort(predict.numpy())[-k:][::-1] for predict in predicts]
        return top_k_indices
    
    def test(self):
        label_count = {}
        label_correct = {}
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        y_true = []
        y_pred = []
        num_batches = self.test_length // self.batch_size
        
        for step in range(num_batches):
            batch = self.get_a_test_batch(step)
            data, labels = self.get_data_label(batch)
            top_k_labels = self.predict_top_k(data, k=5)
            y_true.extend(labels)
            y_pred.extend([labels[0] for labels in top_k_labels])
            
            for label, top_k in zip(labels, top_k_labels):
                label_str = str(int(label))
                if label_str not in label_count:
                    label_count[label_str] = 0
                    label_correct[label_str] = 0
                
                if label == top_k[0]:
                    label_correct[label_str] += 1
                    top1_correct += 1
                if label in top_k[:3]:
                    top3_correct += 1
                if label in top_k[:5]:
                    top5_correct += 1
                label_count[label_str] += 1
        
        accuracy1 = {}
        for label in sorted(label_count):
            accuracy1[label] = label_correct[label] / label_count[label]
        
        top1_accuracy = top1_correct / self.test_length
        top3_accuracy = top3_correct / self.test_length
        top5_accuracy = top5_correct / self.test_length
        
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        return micro_f1, macro_f1
    
    def test2(self):
        y_true = []
        y_pred = []
        num_batches = self.valid_length // self.batch_size
        
        for step in range(num_batches):
            batch = self.get_a_valid_batch(step)
            data, label = self.get_data_label(batch)
            top_k_labels = self.predict_top_k(data, k=5)
            y_true.extend(label)
            y_pred.extend([labels[0] for labels in top_k_labels])
        
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        return micro_f1, macro_f1
