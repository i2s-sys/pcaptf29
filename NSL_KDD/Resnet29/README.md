# TensorFlow 2.9.0 Compatible ResNet Implementation

This directory contains the migrated ResNet implementation from TensorFlow 2.4.0 to TensorFlow 2.9.0.

## Key Changes Made

### 1. TensorFlow 2.x Migration
- **Removed tf.compat.v1 usage**: Replaced all `tf.compat.v1` calls with native TensorFlow 2.x APIs
- **Eliminated Session-based execution**: Replaced `tf.compat.v1.Session()` with eager execution
- **Removed placeholders**: Replaced `tf.compat.v1.placeholder()` with direct tensor operations
- **Updated model architecture**: Converted to `tf.keras.Model` subclassing for better compatibility

### 2. Model Architecture Updates
- **ResNetModel class**: New Model subclass that handles the ResNet architecture
- **Eager execution**: All operations now run in eager mode by default
- **GradientTape**: Used for automatic differentiation instead of manual gradient computation
- **Model saving**: Updated to use `.h5` format instead of checkpoint files

### 3. Training Loop Changes
- **@tf.function decorator**: Added for better performance in training loops
- **Tensor operations**: Direct tensor operations instead of session.run()
- **Optimizer updates**: Using tf.keras.optimizers.Adam instead of tf.compat.v1.train.AdamOptimizer

## Directory Structure

```
resnet29/
├── beyes/                    # Bayesian optimization experiments
│   ├── BeyesResNet.py       # Main Bayesian ResNet implementation
│   ├── Resnet.py            # Core ResNet model
│   └── Resnet2.py           # Secondary ResNet for retraining
├── FeatureSelect/            # Feature selection experiments
│   ├── pcapResnetPacketSeed.py
│   ├── pcapTrainResPacket_ES2_1.py
│   ├── pcapTrainResPacket_ES2_2.py
│   └── pcapTrainResPacket_ES2_4.py
├── FeatureSelect2/           # Advanced feature selection
│   ├── pcapResnetSeed_factor.py
│   └── TrainResnet_factor.py
├── Loss/                     # Loss function experiments
│   ├── pcapResnetPureSeed.py
│   └── pcapTrainResPure_ce.py
├── requirements.txt         # Package dependencies
└── README.md               # This file
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install tensorflow==2.9.0
pip install numpy>=1.21.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install hyperopt>=0.2.7
pip install pandas>=1.3.0
```

## Usage

### Running Feature Selection Experiments

```bash
cd FeatureSelect
python pcapTrainResPacket_ES2_1.py  # K=1 features
python pcapTrainResPacket_ES2_2.py  # K=2 features
python pcapTrainResPacket_ES2_4.py  # K=4 features
```

### Running Loss Function Experiments

```bash
cd Loss
python pcapTrainResPure_ce.py  # Cross-entropy loss
```

### Running Bayesian Optimization

```bash
cd beyes
python BeyesResNet.py
```

## Key Features

1. **Feature Selection**: Automatic feature selection using scaling factors
2. **Early Stopping**: Implemented early stopping based on feature intersection
3. **Multiple Loss Functions**: Support for CE, CB, and CB-Focal loss
4. **Class Balancing**: Automatic class weight calculation for imbalanced datasets
5. **Model Persistence**: Save and load models in HDF5 format

## Performance Notes

- The migrated code maintains the same functionality as the original TensorFlow 2.4.0 version
- Eager execution provides better debugging capabilities
- Model saving/loading is more straightforward with HDF5 format
- Training performance should be comparable or better than the original implementation

## Compatibility

- **Python**: 3.7+
- **TensorFlow**: 2.9.0
- **CUDA**: Compatible with CUDA 11.2+ (if using GPU)
- **Operating System**: Windows, Linux, macOS
