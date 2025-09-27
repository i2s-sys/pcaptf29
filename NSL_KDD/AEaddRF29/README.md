# TensorFlow 2.9.0 Compatible AutoEncoder + Random Forest Implementation

This directory contains AutoEncoder + Random Forest implementations migrated from TensorFlow 2.4.0 to TensorFlow 2.9.0 with full reproducibility support.

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
cd NSL_KDD/AEaddRF29
python ../VGG29/test_basic.py
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow==2.9.0
pip install numpy scikit-learn matplotlib pandas
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### 2. Run Training Scripts

#### Method 1: Using Shell Scripts (Recommended)

**Linux/Mac:**
```bash
# Basic AutoEncoder + Random Forest training
./run_aeaddrf_training.sh

# Specify parameters: ae_factor, SEED=25, TRAIN_EPOCH=30
./run_aeaddrf_training.sh ae_factor 25 30

# Second phase training with selected features
./run_aeaddrf_training.sh second_train 25 30 16
```

**Windows:**
```cmd
# Basic AutoEncoder + Random Forest training
run_aeaddrf_training.bat

# Specify parameters
run_aeaddrf_training.bat ae_factor 25 30

# Second phase training with selected features
run_aeaddrf_training.bat second_train 25 30 16
```

#### Method 2: Direct Python Execution

```bash
# Basic AutoEncoder + Random Forest training
python TrainpcapAEAddRF.py

# AutoEncoder + Random Forest with feature selection
python TrainpcapAEAddRF_factor.py

# Second phase training
cd FeatureSelect
python SecondTrain.py
```

## 📁 Directory Structure

```
AEaddRF29/
├── FeatureSelect/              # Second phase training
│   ├── AEAddRF2.py             # AutoEncoder model for selected features
│   └── SecondTrain.py          # Second phase training script
├── model/                      # Model storage directory
├── pcapAEAddRF.py              # Basic AutoEncoder + Random Forest model
├── pcapAEAddRF_factor.py       # AutoEncoder + Random Forest with feature selection
├── TrainpcapAEAddRF.py         # Basic training script
├── TrainpcapAEAddRF_factor.py  # Feature selection training script
├── run_aeaddrf_training.sh     # Linux/Mac training script
├── run_aeaddrf_training.bat    # Windows training script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🎯 Configurable Parameters

The shell scripts support the following parameters:

### AutoEncoder + Random Forest Parameters

- **script_type**: Training script type (default: "ae")
  - `"ae"`: Basic AutoEncoder + Random Forest
  - `"ae_factor"`: AutoEncoder + Random Forest with feature selection
  - `"second_train"`: Second phase training with selected features

- **SEED**: Random seed for reproducibility (default: 25)
  - Ensures consistent results across runs
  - Any integer value is supported

- **TRAIN_EPOCH**: Number of training epochs (default: 30)
  - Controls training duration
  - Higher values = longer training

- **selected_features**: Number of selected features for second phase (default: 16)
  - Only used for second_train script type
  - Should match the number of features selected in first phase

## 🏗️ AutoEncoder Architecture

The AutoEncoder model maintains the original architecture:

```
Input (41 features) → Encoder → Latent (K=32) → Decoder → Reconstruction (41 features)
                                    ↓
                              Classifier → Classification (2 classes)
```

**Encoder Network:**
- Dense(32) → Dropout → Dense(16) → Dropout → Dense(K=32)

**Decoder Network:**
- Dense(16) → Dropout → Dense(32) → Dropout → Dense(41)

**Classifier Network:**
- Dense(64) → Dropout → Dense(32) → Dropout → Dense(2)

## 🔬 Key Features

### 1. AutoEncoder + Random Forest Hybrid
- **AutoEncoder**: Learns compressed representations of input data
- **Random Forest**: Trains on encoded features for classification
- **Hybrid Approach**: Combines deep learning feature extraction with ensemble learning

### 2. Feature Selection
- L1 regularization on scaling factors
- Top-K feature selection based on learned importance
- Two-phase training: feature selection → model retraining

### 3. Loss Functions
- **Reconstruction Loss**: Mean Squared Error for autoencoder
- **Classification Loss**: Sparse Categorical Crossentropy for classifier
- **Combined Loss**: Reconstruction + Classification + L1 Regularization

### 4. Reproducibility
- Deterministic random seed configuration
- TensorFlow 2.9.0 operation determinism
- Consistent results across multiple runs

## 📊 Usage Examples

### Example 1: Basic Training

```bash
# Train basic AutoEncoder + Random Forest
./run_aeaddrf_training.sh ae 25 30
```

### Example 2: Feature Selection Training

```bash
# Train with feature selection
./run_aeaddrf_training.sh ae_factor 25 30
```

### Example 3: Second Phase Training

```bash
# Train second phase with 16 selected features
./run_aeaddrf_training.sh second_train 25 30 16
```

### Example 4: Reproducibility Testing

```bash
# Run same configuration multiple times - should get identical results
./run_aeaddrf_training.sh ae 42 20
./run_aeaddrf_training.sh ae 42 20
./run_aeaddrf_training.sh ae 42 20
```

## 🚨 Important Notes

1. **Data Files**: Ensure `train_data.csv` and `test_data.csv` are in the parent directory (`../`)
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
4. Use appropriate number of epochs for your dataset

## 📈 Expected Results

- **AutoEncoder**: Learns meaningful compressed representations
- **Random Forest**: Provides robust classification on encoded features
- **Feature Selection**: Identifies most important features for classification
- **Reproducibility**: Identical results across runs with same parameters
- **Training Time**: Varies based on hardware and configuration

## 🔄 Migration Improvements

Compared to original TensorFlow 2.4.0 code, main improvements include:

1. **Removed tf.compat.v1 dependencies**: Uses native TensorFlow 2.x API
2. **Eager Execution**: Default enabled, improved debugging experience
3. **Model subclassing**: Uses tf.keras.Model subclassing, clearer architecture
4. **@tf.function decorator**: Improved training performance
5. **Deterministic random seeds**: Ensures fully reproducible results
6. **GPU memory management**: Improved GPU memory configuration
7. **Parameterized scripts**: Flexible parameter configuration system

## 🤝 Contributing

When modifying the code:
1. Maintain the random seed configuration
2. Preserve the AutoEncoder architecture
3. Keep parameter configurability
4. Test reproducibility after changes

## 📄 License

This code is part of the TensorFlow 2.9.0 migration project.
