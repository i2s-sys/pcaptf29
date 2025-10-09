# UNSW-IoT ResNet29 - TensorFlow 2.9.0 Migration

This directory contains the migrated ResNet implementation for UNSW-IoT dataset, upgraded from TensorFlow 2.4.0 to TensorFlow 2.9.0.

## 🚀 Features

- **TensorFlow 2.9.0 Compatible**: Fully migrated from TF 2.4.0 to TF 2.9.0
- **Model Reproducibility**: Fixed random seeds for consistent results
- **Feature Selection**: L1 regularization-based feature selection with early stopping
- **Class-Balanced Focal Loss**: Handles imbalanced dataset with 29 classes
- **ResNet Architecture**: Deep residual network with 4 blocks (2,2,2,2 layers each)

## 📁 Directory Structure

```
UNSW-IoT/Resnet29/
├── FeatureSelect/
│   ├── pcapResnetPacketSeed.py      # Main ResNet model with feature selection
│   └── pcapTrainResPacket_ES2_32.py # Training script for K=32 features
├── requirements.txt                  # Python dependencies
└── README.md                        # This file
```

## 🔧 Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Ensure data files exist**:
   - `../train_data.csv` - Training data (72 features + 1 label)
   - `../test_data.csv` - Test data (72 features + 1 label)

## 🎯 Usage

### Feature Selection Training

Train the ResNet model with feature selection (K=32 features):

```bash
cd UNSW-IoT/Resnet29/FeatureSelect
python pcapTrainResPacket_ES2_32.py
```

### Parameters

- **K**: Number of features to select (default: 32)
- **ES_THRESHOLD**: Early stopping threshold (default: 3)
- **SEED**: Random seed for reproducibility (default: 25)
- **TRAIN_EPOCH**: Maximum training epochs (default: 30)

## 🏗️ Model Architecture

### ResNet Structure
- **Input**: 72 features → Reshape to (72, 1, 1)
- **Initial Conv**: 64 filters, 3x3 kernel
- **ResNet Blocks**: 4 blocks with [2,2,2,2] layers each
  - Block 1: 64 filters
  - Block 2: 128 filters, stride=2
  - Block 3: 256 filters, stride=2
  - Block 4: 512 filters, stride=2
- **Global Average Pooling**: Reduce spatial dimensions
- **Dense Layer**: 29 classes output

### Feature Selection
- **Scaling Factor**: Learnable parameter for each feature
- **L1 Regularization**: Encourages sparsity in scaling factors
- **Early Stopping**: Based on intersection of top-K features across epochs

## 📊 Dataset Information

- **Dataset**: UNSW-IoT
- **Features**: 72 network traffic features
- **Classes**: 29 IoT device types (0-28)
- **Training**: Class-balanced with effective number weighting
- **Loss Function**: Class-Balanced Focal Loss (β=0.999, γ=1)

## 🔄 Migration Changes

### From TensorFlow 2.4.0 to 2.9.0

1. **Removed tf.compat.v1**:
   - `tf.compat.v1.Session()` → `tf.keras.Model` subclassing
   - `tf.compat.v1.placeholder()` → Direct tensor operations
   - `tf.compat.v1.disable_eager_execution()` → Removed (eager execution default)

2. **Updated Training Loop**:
   - `sess.run()` → `tf.GradientTape()` with `@tf.function`
   - Manual gradient computation → `optimizer.apply_gradients()`

3. **Model Reproducibility**:
   - Added `set_deterministic_seed()` function
   - Uses `tf.keras.utils.set_random_seed()` and `tf.config.experimental.enable_op_determinism()`

4. **Fixed Training Issues**:
   - Added data shuffling: `np.random.shuffle(self.train_data)`
   - Fixed F1 score return order: `(macro_f1, micro_f1)`
   - Added F1 score history tracking

## 🎛️ Key Classes

### Resnet
- Main model for feature selection
- Includes scaling factor for L1 regularization
- Early stopping based on feature intersection

### Resnet2
- Second phase training with selected features
- Simplified architecture without scaling factor
- Used for final model training

### BasicBlock
- ResNet basic block with skip connections
- Batch normalization and ReLU activation
- Configurable stride for downsampling

## 📈 Performance Metrics

- **Top-1 Accuracy**: Primary classification accuracy
- **Top-3 Accuracy**: Top-3 prediction accuracy
- **Top-5 Accuracy**: Top-5 prediction accuracy
- **Macro-F1**: Average F1 score across all classes
- **Micro-F1**: Global F1 score considering all samples

## 🔧 Configuration

### Hyperparameters
```python
DATA_DIM = 72          # Input feature dimension
OUTPUT_DIM = 29        # Number of classes
LEARNING_RATE = 0.0001 # Adam optimizer learning rate
BATCH_SIZE = 128       # Training batch size
BETA = 0.999          # Class balancing parameter
GAMMA = 1             # Focal loss parameter
```

### Feature Selection
```python
K = 32                # Number of features to select
ES_THRESHOLD = 3      # Early stopping threshold
L1_REGULARIZATION = 0.001  # L1 penalty for scaling factors
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `BATCH_SIZE` from 128 to 64 or 32
   - Set `CUDA_VISIBLE_DEVICES` to use specific GPU

2. **Data File Not Found**:
   - Ensure `train_data.csv` and `test_data.csv` exist in parent directory
   - Check file paths in `TRAIN_FILE` and `TEST_FILE` constants

3. **Reproducibility Issues**:
   - Verify `set_deterministic_seed()` is called
   - Check that `tf.config.experimental.enable_op_determinism()` is enabled

## 📝 Example Output

```
=== Epoch 1/30 ===
Epoch 1 completed, average loss: 2.456789, micro-F1: 0.1234, macro-F1: 0.0987, duration: 45.67 seconds
Epoch 1, Intersection size: 15

=== Epoch 2/30 ===
Epoch 2 completed, average loss: 2.123456, micro-F1: 0.1456, macro-F1: 0.1123, duration: 43.21 seconds
Loss change: 0.333333, Stable count: 0
Epoch 2, Intersection size: 18
...
```

## 🤝 Contributing

When modifying the code:
1. Maintain the ResNet architecture structure
2. Ensure reproducibility with fixed seeds
3. Test with both feature selection and final training phases
4. Update documentation for any parameter changes

## 📄 License

This code is part of the UNSW-IoT ResNet implementation for network traffic classification.