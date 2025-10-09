# TensorFlow 2.9.0 compatible ResNet implementation for second phase training
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
OUTPUT_DIM = 29
LOSSTYPE = "ce"
BETA = 0.9999  # 类平衡损失的β gamma 使用cb 因为cb效果最好
GAMMA = 1

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

class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride, seed):
        super(BasicBlock, self).__init__()
        self.seed = seed
        set_seed(seed)
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same',
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same',
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride,
                                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)))
        else:
            self.downsample = None

    def call(self, inputs, training=None):
        residual = inputs
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        
        if self.downsample is not None:
            residual = self.downsample(inputs)
        
        out += residual
        out = self.relu(out)
        return out

class ResNet(Model):
    def __init__(self, lossType, seed, beta, gamma):
        super(ResNet, self).__init__()
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
        
        # Build ResNet architecture
        self.build_resnet()
        
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
    
    def build_resnet(self):
        """Build ResNet architecture"""
        # Input layer
        self.input_layer = layers.Input(shape=(DATA_DIM,), name='input')
        
        # Reshape input to 2D for Conv2D
        self.reshape = layers.Reshape((DATA_DIM, 1, 1))
        
        # Initial convolution
        self.conv1 = layers.Conv2D(64, (7, 7), strides=2, padding='same',
                                   kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')
        self.maxpool = layers.MaxPool2D((3, 3), strides=2, padding='same')
        
        # ResNet blocks
        self.layer1 = Sequential([
            BasicBlock(64, 1, self.seed),
            BasicBlock(64, 1, self.seed)
        ])
        
        self.layer2 = Sequential([
            BasicBlock(128, 2, self.seed),
            BasicBlock(128, 1, self.seed)
        ])
        
        self.layer3 = Sequential([
            BasicBlock(256, 2, self.seed),
            BasicBlock(256, 1, self.seed)
        ])
        
        self.layer4 = Sequential([
            BasicBlock(512, 2, self.seed),
            BasicBlock(512, 1, self.seed)
        ])
        
        # Global average pooling and classifier
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(OUTPUT_DIM, activation='softmax',
                              kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
    
    def call(self, inputs, training=None):
        """Forward pass through ResNet"""
        x = self.reshape(inputs)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        
        x = self.global_avg_pool(x)
        x = self.fc(x)
        
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
            # Class-balanced cross-entropy
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
        else:
            return tf.reduce_mean(ce)
    
    @tf.function
    def train_step(self, x, y_true):
        """Single training step"""
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y_true, y_pred)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss, y_pred
    
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
            
            loss, y_pred = self.train_step(data, label)
            
            total_loss += loss
            num_batches += 1
        
        average_loss = total_loss / num_batches
        self.loss_history.append(float(average_loss))
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, loss: {average_loss:.6f}, duration: {epoch_duration:.2f} seconds')
        
        # Test performance
        micro_F1, macro_F1 = self.test2()
        self.micro_F1List.append(micro_F1)
        self.macro_F1List.append(macro_F1)
        return micro_F1
    
    def get_a_train_batch(self, step):
        """Get training batch"""
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.train_length:
            max_index = self.train_length
        return self.train_data[min_index:max_index]
    
    def get_data_label(self, batch):
        """Extract data and labels from batch"""
        data = np.delete(batch, -1, axis=1)
        label = np.array(batch, dtype=np.int32)[:, -1]
        return data, label
    
    def init_data(self):
        """Initialize training and test data"""
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
        """Normalize data"""
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
        """Predict top-k classes"""
        x_feature = tf.expand_dims(x_feature, 0)
        y_pred = self(x_feature, training=False)
        top_k_indices = np.argsort(y_pred.numpy()[0])[-k:][::-1]
        return top_k_indices
    
    def test(self):
        """Test on test set"""
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
        """Quick test for validation"""
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
