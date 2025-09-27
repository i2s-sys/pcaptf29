# TensorFlow 2.9.0 compatible AutoEncoder + Random Forest with feature selection
import time
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
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
K = 32
OUTPUT_DIM = 2
BETA = 0.999
GAMMA = 1
TRAIN_FILE = '../train_data.csv'
TEST_FILE = '../test_data.csv'
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
MODEL_SAVE_PATH = '../model/'
MODEL_SAVE_PATH2 = './model2/'
MODEL_NAME = 'model'
MODEL_NAME2 = 'model2'
KEEP_PROB = 0.5
top_k_values=[]
top_k_indice=[]
NUM_ATTENTION_CHANNELS=1

def set_seed(seed):
    """设置随机种子（保持向后兼容）"""
    set_deterministic_seed(seed)

class AE(Model):
    def __init__(self, seed=25):
        super(AE, self).__init__()
        set_seed(seed)
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
        
        # Initialize scaling factor for feature selection
        self.scaling_factor = tf.Variable(
            tf.constant(1.0, dtype=tf.float32, shape=[1, DATA_DIM]),
            trainable=True,
            name='scaling_factor'
        )
        
        # Build AutoEncoder architecture
        self.build_encoder()
        self.build_decoder()
        self.build_classifier()
        
        # Setup optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Build loss functions
        self.build_loss_fn()
    
    def build_encoder(self):
        """Build encoder network"""
        self.encoder = Sequential([
            layers.Dense(32, activation='relu', name='encoder_1',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),
            layers.Dropout(KEEP_PROB),
            layers.Dense(16, activation='relu', name='encoder_2',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),
            layers.Dropout(KEEP_PROB),
            layers.Dense(K, activation='relu', name='encoder_3',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        ])
    
    def build_decoder(self):
        """Build decoder network"""
        self.decoder = Sequential([
            layers.Dense(16, activation='relu', name='decoder_1',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),
            layers.Dropout(KEEP_PROB),
            layers.Dense(32, activation='relu', name='decoder_2',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),
            layers.Dropout(KEEP_PROB),
            layers.Dense(DATA_DIM, activation='sigmoid', name='decoder_3',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        ])
    
    def build_classifier(self):
        """Build classifier network"""
        self.classifier = Sequential([
            layers.Dense(64, activation='relu', name='classifier_1',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),
            layers.Dropout(KEEP_PROB),
            layers.Dense(32, activation='relu', name='classifier_2',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),
            layers.Dropout(KEEP_PROB),
            layers.Dense(OUTPUT_DIM, activation='softmax', name='classifier_3',
                        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed))
        ])
    
    def encode(self, x, training=None):
        """Encode input to latent representation"""
        return self.encoder(x, training=training)
    
    def decode(self, z, training=None):
        """Decode latent representation to reconstruction"""
        return self.decoder(z, training=training)
    
    def call(self, x, training=None):
        """Forward pass through autoencoder with feature selection"""
        # Apply scaling factor for feature selection
        batch_size = tf.shape(x)[0]
        scaling_factor_extended = tf.tile(self.scaling_factor, [batch_size, 1])
        scaled_input = tf.multiply(x, scaling_factor_extended)
        
        encoded = self.encode(scaled_input, training=training)
        decoded = self.decode(encoded, training=training)
        classified = self.classifier(encoded, training=training)
        return decoded, classified, encoded
    
    def build_loss_fn(self):
        """Build loss functions"""
        self.reconstruction_loss_fn = tf.keras.losses.MeanSquaredError()
        self.classification_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.l1_regularizer = tf.keras.regularizers.l1(0.001)
    
    def compute_loss(self, x, y_true_class, training=None):
        """Compute combined loss with L1 regularization"""
        decoded, classified, encoded = self(x, training=training)
        
        # Reconstruction loss
        reconstruction_loss = self.reconstruction_loss_fn(x, decoded)
        
        # Classification loss
        classification_loss = self.classification_loss_fn(y_true_class, classified)
        
        # L1 regularization on scaling factor
        regularization_penalty = self.l1_regularizer(self.scaling_factor)
        
        # Combined loss
        total_loss = reconstruction_loss + classification_loss + regularization_penalty
        
        return total_loss, reconstruction_loss, classification_loss, decoded, classified, encoded
    
    @tf.function
    def train_step(self, x, y_true_class):
        """Single training step"""
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, classification_loss, decoded, classified, encoded = self.compute_loss(x, y_true_class, training=True)
        
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return total_loss, reconstruction_loss, classification_loss, decoded, classified, encoded
    
    def train(self):
        """Train for one epoch"""
        total_loss = 0
        total_reconstruction_loss = 0
        total_classification_loss = 0
        num_batches = 0
        epoch_start_time = time.time()
        
        # Shuffle training data
        np.random.shuffle(self.train_data)
        
        for step in range(self.total_iterations_per_epoch):
            batch = self.get_a_train_batch(step)
            data, label = self.get_data_label(batch)
            
            loss, recon_loss, class_loss, decoded, classified, encoded = self.train_step(data, label)
            
            total_loss += loss
            total_reconstruction_loss += recon_loss
            total_classification_loss += class_loss
            num_batches += 1
        
        average_loss = total_loss / num_batches
        average_recon_loss = total_reconstruction_loss / num_batches
        average_class_loss = total_classification_loss / num_batches
        
        self.loss_history.append(float(average_loss))
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, total loss: {average_loss:.6f}, '
              f'reconstruction loss: {average_recon_loss:.6f}, classification loss: {average_class_loss:.6f}, '
              f'duration: {epoch_duration:.2f} seconds')
        
        # Test performance
        micro_F1, macro_F1 = self.test2()
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
        _, classified, _ = self(x_feature, training=False)
        top_k_indices = np.argsort(classified.numpy()[0])[-k:][::-1]
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
    
    def get_encoded_features(self, data):
        """Get encoded features for Random Forest"""
        encoded_features = []
        for row in data:
            feature = row[0:-1]
            x_feature = np.array(feature, dtype=np.float32)
            x_feature = tf.expand_dims(x_feature, 0)
            _, _, encoded = self(x_feature, training=False)
            encoded_features.append(encoded.numpy()[0])
        return np.array(encoded_features)
    
    def train_random_forest(self):
        """Train Random Forest on encoded features"""
        print("Training Random Forest on encoded features...")
        
        # Get encoded features for training data
        train_features = self.get_encoded_features(self.train_data)
        train_labels = np.array([row[-1] for row in self.train_data])
        
        # Train Random Forest
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=self.seed)
        self.rf_classifier.fit(train_features, train_labels)
        
        # Test Random Forest
        test_features = self.get_encoded_features(self.test_data)
        test_labels = np.array([row[-1] for row in self.test_data])
        
        rf_predictions = self.rf_classifier.predict(test_features)
        rf_accuracy = np.mean(rf_predictions == test_labels)
        
        print(f"Random Forest accuracy: {rf_accuracy:.4f}")
        
        # Calculate F1 scores
        rf_macro_f1 = f1_score(test_labels, rf_predictions, average='macro')
        rf_micro_f1 = f1_score(test_labels, rf_predictions, average='micro')
        
        print(f"Random Forest Macro-F1: {rf_macro_f1:.4f}")
        print(f"Random Forest Micro-F1: {rf_micro_f1:.4f}")
        
        return rf_accuracy, rf_macro_f1, rf_micro_f1
