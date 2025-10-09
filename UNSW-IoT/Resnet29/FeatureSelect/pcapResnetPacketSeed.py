# TensorFlow 2.9.0 compatible ResNet implementation with feature selection
import gc
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

DATA_DIM = 72
OUTPUT_DIM = 29  # 0-28类
LEARNING_RATE = 0.0001
BETA = 0.999
GAMMA = 1
BATCH_SIZE = 128
TRAIN_FILE = '../train_data.csv'
TEST_FILE = '../test_data.csv'

MODEL_SAVE_PATH = './model/'
MODEL_SAVE_PATH2 = './model2/'
MODEL_NAME = 'model'
MODEL_NAME2 = 'model2'
KEEP_PROB = 0.5
top_k_values = []
top_k_indice = []
NUM_ATTENTION_CHANNELS = 1

feature_widths = [
    32, 32, 32, 32,  # fiat_mean, fiat_min, fiat_max, fiat_std
    32, 32, 32, 32,  # biat_mean, biat_min, biat_max, biat_std
    32, 32, 32, 32,  # diat_mean, diat_min, diat_max, diat_std
    32,              # duration 13
    64, 32, 32, 32, 32,  # fwin_total, fwin_mean, fwin_min, fwin_max, fwin_std
    64, 32, 32, 32, 32,  # bwin_total, bwin_mean, bwin_min, bwin_max, bwin_std
    64, 32, 32, 32, 32,  # dwin_total, dwin_mean, dwin_min, dwin_max, dwin_std
    16, 16, 16,         # fpnum, bpnum, dpnum
    32, 32, 32, 32,         # bfpnum_rate, fpnum_s, bpnum_s, dpnum_s 22
    64, 32, 32, 32, 32,  # fpl_total, fpl_mean, fpl_min, fpl_max, fpl_std
    64, 32, 32, 32, 32,  # bpl_total, bpl_mean, bpl_min, bpl_max, bpl_std
    64, 32, 32, 32, 32,  # dpl_total, dpl_mean, dpl_min, dpl_max, dpl_std
    32, 32, 32, 32,         # bfpl_rate, fpl_s, bpl_s, dpl_s  19
    16, 16, 16, 16, 16, 16, 16, 16,  # fin_cnt, syn_cnt, rst_cnt, pst_cnt, ack_cnt, urg_cnt, cwe_cnt, ece_cnt
    16, 16, 16, 16,     # fwd_pst_cnt, fwd_urg_cnt, bwd_pst_cnt, bwd_urg_cnt
    16, 16, 16,         # fp_hdr_len, bp_hdr_len, dp_hdr_len
    32, 32, 32          # f_ht_len, b_ht_len, d_ht_len 18
]

def set_seed(seed):
    """设置随机种子（保持向后兼容）"""
    set_deterministic_seed(seed)

class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1, seed=None):
        super(BasicBlock, self).__init__()
        set_deterministic_seed(seed)
        self.seed = seed
        self.conv1 = layers.Conv2D(
            filter_num, (3, 3), strides=stride, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        )
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(
            filter_num, (3, 3), strides=1, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
        )
        self.bn2 = layers.BatchNormalization()
        
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(
                filter_num, (1, 1), strides=stride,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
            ))
        else:
            self.downsample = lambda x: x
    
    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output

class Resnet(Model):
    def __init__(self, K, ES_THRESHOLD, seed):
        super(Resnet, self).__init__()
        self.K = K
        self.ES_THRESHOLD = ES_THRESHOLD
        set_seed(seed)
        self.gamma = GAMMA
        self.seed = seed
        self.maintainCnt = 0
        self.loss_history = []
        self.micro_F1List = []
        self.macro_F1List = []
        self.intersection_sets = []
        self.TSMRecord = []
        self.earlyStop = False
        print(f"BETA = {BETA}, GAMMA = {GAMMA}")
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epoch_count = 0
        
        self.init_data()
        self.total_iterations_per_epoch = self.train_length // BATCH_SIZE
        
        # Build ResNet architecture
        self.build_resnet()
        
        # Setup class weights
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
    
    def build_resnet(self):
        """Build ResNet architecture"""
        # Initial convolution layer
        self.conv_layer = layers.Conv2D(
            64, kernel_size=(3, 3), strides=1, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
        )
        self.stm = Sequential([
            self.conv_layer,
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same')
        ])
        
        # ResNet blocks
        layer_dims = [2, 2, 2, 2]
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        
        # Global average pooling and final dense layer
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(OUTPUT_DIM)
        
        # Scaling factor for feature selection
        self.scaling_factor = tf.Variable(
            tf.constant(1, dtype=tf.float32, shape=[1, DATA_DIM])
        )
    
    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride, seed=self.seed))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1, seed=self.seed))
        return res_blocks
    
    def call(self, inputs, training=None):
        """Forward pass"""
        # Apply scaling factor for feature selection
        scaling_factor_extended = tf.tile(self.scaling_factor, [tf.shape(inputs)[0], 1])
        scaled_input = tf.multiply(inputs, scaling_factor_extended)
        scaled_input = tf.reshape(scaled_input, [tf.shape(inputs)[0], DATA_DIM, 1, 1])
        
        x = self.stm(scaled_input, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        y = self.fc(x)
        return y
    
    def build_loss_fn(self):
        """Build loss function"""
        self.class_weights = tf.constant(
            [self.class_weights_dict[str(i)] for i in range(len(self.class_weights_dict))], 
            dtype=tf.float32
        )
    
    def compute_loss(self, y_true, y_pred):
        """Compute loss with L1 regularization on scaling factor"""
        l1_regularizer = tf.keras.regularizers.l1(0.001)
        regularization_penalty = l1_regularizer(self.scaling_factor)
        
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        
        # Class-balanced focal loss
        sample_weights = tf.gather(self.class_weights, y_true)
        softmax_probs = tf.nn.softmax(y_pred)
        labels_one_hot = tf.one_hot(y_true, depth=len(self.class_weights_dict))
        pt = tf.reduce_sum(labels_one_hot * softmax_probs, axis=1)
        modulator = tf.pow(1.0 - pt, self.gamma)
        focal_loss = modulator * ce
        cb_focal_loss = tf.multiply(focal_loss, sample_weights)
        
        return tf.reduce_sum(cb_focal_loss) + regularization_penalty
    
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
        
        # 随机打乱训练数据
        np.random.shuffle(self.train_data)
        
        for step in range(self.total_iterations_per_epoch):
            batch = self.get_a_train_batch(step)
            data, label = self.get_data_label(batch)
            
            loss, predictions = self.train_step(data, label)
            total_loss += loss
            num_batches += 1
        
        average_loss = total_loss / num_batches
        self.loss_history.append(float(average_loss))
        
        # 修复返回值顺序：test2返回(macro_f1, micro_f1)
        macro_F1, micro_F1 = self.test2()
        self.micro_F1List.append(micro_F1)
        self.macro_F1List.append(macro_F1)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss:.6f}, '
              f'micro-F1: {micro_F1:.4f}, macro-F1: {macro_F1:.4f}, duration: {epoch_duration:.2f} seconds')
        
        # 早停策略
        keyFeatureNums = self.K
        values, indices = tf.math.top_k(self.scaling_factor, k=keyFeatureNums, sorted=True)
        max_indices = indices.numpy()[0]
        max_set = set(max_indices)
        self.intersection_sets.append(max_set)
        
        if len(self.intersection_sets) > self.ES_THRESHOLD:
            self.intersection_sets.pop(0)
        
        if len(self.intersection_sets) >= self.ES_THRESHOLD:
            intersection = set.intersection(*self.intersection_sets)
            print(f"Epoch {self.epoch_count + 1}, Intersection size: {len(intersection)}")
            self.TSMRecord.append(len(intersection))
            if len(intersection) == keyFeatureNums:
                print("Early stopping condition met.")
                self.earlyStop = True
        
        if self.epoch_count == 0:
            delta_loss = 0
            self.count = 0
            self.curr_loss = average_loss
        else:
            self.prev_loss = self.curr_loss
            self.curr_loss = average_loss
            delta_loss = abs(self.curr_loss - self.prev_loss)
            if delta_loss <= 0.03:
                self.count += 1
            else:
                self.count = 0
        
        return delta_loss, self.count
    
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
        label_data = {i: [] for i in range(29)}
        
        def process_row(row):
            data = [0 if char == 'None' else np.float32(char) for char in row]
            label = int(data[-1])
            label_data[label].append(data)
            self.label_status[str(label)] = self.label_status.get(str(label), 0) + 1
        
        with open(TRAIN_FILE, mode="r", encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header if exists
            for row in csv_reader:
                process_row(row)
        
        self.train_data = [data for label in label_data.values() for data in label]
        
        with open(TEST_FILE, mode="r", encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            self.test_data = [
                [0 if char == 'None' else np.float32(char) for char in row]
                for row in csv_reader
            ]
        
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
    
    def test2(self):
        """仅返回top1测试结果"""
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
        return macro_f1, micro_f1

class Resnet2(Model):  # 第二次训练
    def __init__(self, dim, selected_features=[], seed=25):
        super(Resnet2, self).__init__()
        self.dim = dim
        self.top_k_indices = selected_features
        set_seed(seed)
        self.gamma = GAMMA
        self.seed = seed
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epoch_count = 0
        self.loss_history = []
        self.micro_F1List = []
        self.macro_F1List = []
        
        self.init_data()
        self.total_iterations_per_epoch = self.train_length // BATCH_SIZE
        
        # Build ResNet architecture
        self.build_resnet()
        
        # Setup class weights
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
    
    def build_resnet(self):
        """Build ResNet architecture"""
        # Initial convolution layer
        self.conv_layer = layers.Conv2D(
            64, kernel_size=(3, 3), strides=1, padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
        )
        self.stm = Sequential([
            self.conv_layer,
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same')
        ])
        
        # ResNet blocks
        layer_dims = [2, 2, 2, 2]
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        
        # Global average pooling and final dense layer
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(OUTPUT_DIM)
    
    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride, seed=self.seed))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1, seed=self.seed))
        return res_blocks
    
    def call(self, inputs, training=None):
        """Forward pass"""
        scaled_input = tf.reshape(inputs, [tf.shape(inputs)[0], self.dim, 1, 1])
        x = self.stm(scaled_input, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        y = self.fc(x)
        return y
    
    def build_loss_fn(self):
        """Build loss function"""
        self.class_weights = tf.constant(
            [self.class_weights_dict[str(i)] for i in range(len(self.class_weights_dict))], 
            dtype=tf.float32
        )
    
    def compute_loss(self, y_true, y_pred):
        """Compute loss"""
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        
        # Class-balanced focal loss
        sample_weights = tf.gather(self.class_weights, y_true)
        softmax_probs = tf.nn.softmax(y_pred)
        labels_one_hot = tf.one_hot(y_true, depth=len(self.class_weights_dict))
        pt = tf.reduce_sum(labels_one_hot * softmax_probs, axis=1)
        modulator = tf.pow(1.0 - pt, self.gamma)
        focal_loss = modulator * ce
        cb_focal_loss = tf.multiply(focal_loss, sample_weights)
        
        return tf.reduce_sum(cb_focal_loss)
    
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
        
        # 随机打乱训练数据
        np.random.shuffle(self.train_data)
        
        for step in range(self.total_iterations_per_epoch):
            batch = self.get_a_train_batch(step)
            data, label = self.get_data_label(batch)
            
            loss, predictions = self.train_step(data, label)
            total_loss += loss
            num_batches += 1
        
        average_loss = total_loss / num_batches
        self.loss_history.append(float(average_loss))
        
        # 修复返回值顺序：test2返回(macro_f1, micro_f1)
        macro_F1, micro_F1 = self.test2()
        self.micro_F1List.append(micro_F1)
        self.macro_F1List.append(macro_F1)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss:.6f}, '
              f'micro-F1: {micro_F1:.4f}, macro-F1: {macro_F1:.4f}, duration: {epoch_duration:.2f} seconds')
        
        if self.epoch_count == 0:
            delta_loss = 0
            self.count = 0
            self.curr_loss = average_loss
        else:
            self.prev_loss = self.curr_loss
            self.curr_loss = average_loss
            delta_loss = abs(self.curr_loss - self.prev_loss)
            if delta_loss <= 0.03:
                self.count += 1
            else:
                self.count = 0
        
        return delta_loss, self.count
    
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
        label_data = {i: [] for i in range(29)}
        
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
        
        with open(TEST_FILE, mode="r", encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            self.test_data = [
                [0 if char == 'None' else np.float32(char) for i, char in enumerate(row) if
                 i in self.top_k_indices or i == len(row) - 1]
                for row in csv_reader
            ]
        
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
        """仅返回top1测试结果"""
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
        return macro_f1, micro_f1