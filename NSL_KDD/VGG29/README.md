# TensorFlow 2.9.0 Compatible VGG Implementation

This directory contains VGG implementations migrated from TensorFlow 2.4.0 to TensorFlow 2.9.0 with full reproducibility support.

## 🔧 Random Seed Configuration (Ensuring Reproducibility)

All code uses TensorFlow 2.9.0 recommended random seed settings to ensure reproducible results:

```python
def set_deterministic_seed(seed):
    """Set all random seeds to ensure reproducible results"""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
```

### Test Random Seed Configuration

Run the test script to verify random seed settings:

```bash
cd NSL_KDD/VGG29
python ../resnet29/test_reproducibility.py
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow==2.9.0
pip install numpy scikit-learn matplotlib hyperopt pandas
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### 2. Run Training Scripts

#### Method 1: Using Shell Scripts (Recommended)

**Linux/Mac:**
```bash
# Basic usage with default parameters
./run_vgg_training.sh

# Specify parameters: K=2, cb_focal_loss, ES_THRESHOLD=3, SEED=25, feature selection
./run_vgg_training.sh 2 cb_focal_loss 3 25 feature

# Run loss function experiments
./run_vgg_training.sh 1 ce 3 25 loss

# Run Bayesian optimization
./run_vgg_training.sh 1 cb_focal_loss 3 25 beyes
```

**Windows:**
```cmd
# Basic usage with default parameters
run_vgg_training.bat

# Specify parameters
run_vgg_training.bat 2 cb_focal_loss 3 25 feature

# Run loss function experiments  
run_vgg_training.bat 1 ce 3 25 loss

# Run Bayesian optimization
run_vgg_training.bat 1 cb_focal_loss 3 25 beyes
```

#### Method 2: Direct Python Execution

```bash
# Feature selection training
cd FeatureSelect
python pcapTrainVGG_ES3_1.py  # K=1
python pcapTrainVGG_ES3_2.py  # K=2

# Loss function experiments
cd Loss
python pcapTrainVGG_ce.py

# Bayesian optimization
cd beyes
python BeyesVGG.py
```

## 📁 Directory Structure

```
VGG29/
├── beyes/                    # Bayesian optimization
│   ├── BeyesVGG.py          # Bayesian optimization script
│   └── VGG2.py              # VGG model for second phase training
├── FeatureSelect/           # Feature selection experiments
│   ├── pcapVGGSeed.py       # Main VGG model with feature selection
│   ├── pcapTrainVGG_ES3_1.py # Training script for K=1
│   └── pcapTrainVGG_ES3_2.py # Training script for K=2
├── FeatureSelect2/          # Advanced feature selection (to be implemented)
├── Loss/                    # Loss function experiments
│   ├── pcapVGGSeed.py       # VGG model for loss experiments
│   └── pcapTrainVGG_ce.py   # Cross-entropy loss training
├── run_vgg_training.sh      # Linux/Mac training script
├── run_vgg_training.bat     # Windows training script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🎯 Configurable Parameters

The shell scripts support the following parameters:

### VGG Model Parameters

```python
model = VGG(K, loss_type, ES_THRESHOLD, SEED)
```

- **K**: Number of top features to select (default: 1)
  - Supported values: 1, 2, 4, 8, 16, 32
  - Controls feature selection granularity

- **loss_type**: Loss function type (default: "cb_focal_loss")
  - `"ce"`: Cross-Entropy Loss
  - `"cb"`: Class-Balanced Cross-Entropy Loss  
  - `"cb_focal_loss"`: Class-Balanced Focal Loss

- **ES_THRESHOLD**: Early stopping threshold (default: 3)
  - Number of epochs to wait for feature intersection stability
  - Higher values = more patient early stopping

- **SEED**: Random seed for reproducibility (default: 25)
  - Ensures consistent results across runs
  - Any integer value is supported

### Script Types

- **feature**: Feature selection experiments
- **loss**: Loss function comparison experiments
- **beyes**: Bayesian optimization for hyperparameter tuning

## 🏗️ VGG Architecture

The VGG model maintains the original architecture:

```
Input (41 features) → Reshape → VGG Blocks → Fully Connected → Output (2 classes)

VGG Blocks:
- Conv2D(64) → Conv2D(64) → MaxPool2D
- Conv2D(128) → Conv2D(128) → MaxPool2D  
- Conv2D(256) → Conv2D(256) → Conv2D(256) → MaxPool2D
- Conv2D(512)
- Flatten → Dense(1024) → Dropout → Dense(1024) → Dropout → Dense(2)
```

## 🔬 Key Features

### 1. Feature Selection
- L1 regularization on scaling factors
- Top-K feature selection based on learned importance
- Two-phase training: feature selection → model retraining

### 2. Class Balancing
- Effective number calculation for imbalanced datasets
- Class-balanced loss functions (CB, CB-Focal)
- Automatic weight computation based on class frequencies

### 3. Loss Functions
- **Cross-Entropy (CE)**: Standard classification loss
- **Class-Balanced (CB)**: Weighted CE for imbalanced data
- **Class-Balanced Focal**: CB + Focal loss for hard examples

### 4. Reproducibility
- Deterministic random seed configuration
- TensorFlow 2.9.0 operation determinism
- Consistent results across multiple runs

## 📊 Usage Examples

### Example 1: Feature Selection with Different K Values

```bash
# Test different feature selection sizes
./run_vgg_training.sh 1 cb_focal_loss 3 25 feature
./run_vgg_training.sh 2 cb_focal_loss 3 25 feature
./run_vgg_training.sh 4 cb_focal_loss 3 25 feature
```

### Example 2: Loss Function Comparison

```bash
# Compare different loss functions
./run_vgg_training.sh 1 ce 3 25 loss
./run_vgg_training.sh 1 cb 3 25 loss  
./run_vgg_training.sh 1 cb_focal_loss 3 25 loss
```

### Example 3: Reproducibility Testing

```bash
# Run same configuration multiple times - should get identical results
./run_vgg_training.sh 2 cb_focal_loss 3 42 feature
./run_vgg_training.sh 2 cb_focal_loss 3 42 feature
./run_vgg_training.sh 2 cb_focal_loss 3 42 feature
```

## 🚨 Important Notes

1. **Data Files**: Ensure `train_data.csv` and `test_data.csv` are in the parent directory (`../../`)
2. **GPU Memory**: The code includes GPU memory growth configuration for TensorFlow 2.x
3. **Model Saving**: Models are automatically saved with timestamps in `./model/` directories
4. **Reproducibility**: All results should be identical when using the same parameters and seed

## 🐛 Troubleshooting

### Common Issues

1. **TensorFlow Version**: Ensure you're using TensorFlow 2.9.0
2. **CUDA Compatibility**: Check CUDA version compatibility with TensorFlow 2.9.0
3. **Memory Issues**: Reduce batch size if encountering OOM errors
4. **File Paths**: Verify data files are in the correct relative path

### Performance Tips

1. Use GPU acceleration when available
2. Adjust batch size based on available memory
3. Monitor GPU utilization during training
4. Use early stopping to prevent overfitting

## 📈 Expected Results

- **Feature Selection**: Identifies most important features for classification
- **Loss Comparison**: CB-Focal typically performs best on imbalanced data
- **Reproducibility**: Identical results across runs with same parameters
- **Training Time**: Varies based on hardware and configuration

## 🤝 Contributing

When modifying the code:
1. Maintain the random seed configuration
2. Preserve the VGG architecture
3. Keep parameter configurability
4. Test reproducibility after changes

## 🔧 Bug Fixes

### AttributeError: Can't set the attribute "weights"

**Problem**: The original code used `self.weights` which conflicts with Keras Model's built-in `weights` property.

**Solution**: Renamed all instances of `self.weights` to `self.class_weights_dict` throughout the codebase.

**Files affected**:
- `beyes/VGG2.py`
- `FeatureSelect/pcapVGGSeed.py` 
- `Loss/pcapVGGSeed.py`

## 📄 License

This code is part of the TensorFlow 2.9.0 migration project.
