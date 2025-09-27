# TensorFlow 2.9.0 compatible VGG implementation for loss function experiments
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

class VGG(Model):
    def __init__(self, lossType, seed, beta, gamma):
        super(VGG, self).__init__()
        print(f"BETA = {beta}, GAMMA = {gamma}")
        self.beta = beta
        self.gamma = gamma
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.lossType = lossType
        self.seed = seed
        set_seed(seed)
        self.epoch_count = 0
        self.loss_history = []
        self.micro_F1List = []
        self.macro_F1List = []
        
        self.init_data()
        self.total_iterations_per_epoch = self.train_length // BATCH_SIZE
        
        # Build VGG architecture
        self.build_vgg()
        
        # Class balancing setup
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
    
    def build_vgg(self):
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
        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(inputs, [batch_size, DATA_DIM, 1, 1])
        
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
        self.micro_F1List.append(micro_F1)
        self.macro_F1List.append(macro_F1)
        
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
        self.label_status = {}
        
        with open(TRAIN_FILE, 'r') as file:
            csv_reader = csv.reader(file)
            label0_data = []
            label1_data = []
            for row in csv_reader:
                data = []
                for char in row:
                    if char == 'None':
                        data.append(0)
                    else:
                        data.append(np.float32(char))
                if data[-1] == 0:
                    label0_data.append(data)
                if data[-1] == 1:
                    label1_data.append(data)
                if self.label_status.get(str(int(data[-1])), 0) > 0:
                    self.label_status[str(int(data[-1]))] += 1
                else:
                    self.label_status[str(int(data[-1]))] = 1
            self.train_data = label0_data + label1_data
        
        with open(TEST_FILE, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data = []
                for char in row:
                    if char == 'None':
                        data.append(0)
                    else:
                        data.append(np.float32(char))
                self.test_data.append(data)
        
        self.train_data = self.normalization(self.train_data)
        self.test_data = self.normalization(self.test_data)
        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)
        np.random.shuffle(self.train_data)
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
        x_feature = tf.expand_dims(x_feature, 0)
        predict = self(x_feature, training=False)[0]
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
            x_feature = np.array(feature, dtype=np.float32)
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
            x_feature = np.array(feature, dtype=np.float32)
            top_k_labels = self.predict_top_k(x_feature, k=5)
            y_true.append(label)
            y_pred.append(top_k_labels[0])
        
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
        return micro_f1, macro_f1
