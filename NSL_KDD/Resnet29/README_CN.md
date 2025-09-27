# TensorFlow 2.9.0 兼容的ResNet实现

本目录包含了从TensorFlow 2.4.0迁移到TensorFlow 2.9.0的ResNet实现代码。

## 🔧 随机种子设置（确保结果可复现）

为了确保实验结果的可复现性，所有代码都使用了TensorFlow 2.9.0推荐的随机种子设置方法：

```python
def set_deterministic_seed(seed):
    """设置所有随机种子确保结果可复现"""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
```

### 测试随机种子设置

运行测试脚本验证随机种子设置是否正确：

```bash
cd NSL_KDD/resnet29
python test_reproducibility.py
```

如果看到 "✅ 所有随机种子设置正确，结果可以复现！" 说明设置成功。

## 🚀 快速开始

### 1. 安装依赖包

首先安装所有必需的Python包：

```bash
pip install tensorflow==2.9.0
pip install numpy>=1.21.0
pip install scikit-learn>=1.0.0
pip install matplotlib>=3.5.0
pip install hyperopt>=0.2.7
pip install pandas>=1.3.0
```

或者直接使用requirements.txt文件：

```bash
pip install -r requirements.txt
```

### 2. 准备数据文件

确保以下数据文件存在于项目根目录：
- `train_data.csv` - 训练数据
- `test_data.csv` - 测试数据
- `train_data_S.csv` - 标准化训练数据
- `test_data_S.csv` - 标准化测试数据

## 📁 目录结构说明

```
resnet29/
├── beyes/                           # 贝叶斯优化实验
│   ├── BeyesResNet.py              # 贝叶斯ResNet主程序
│   ├── Resnet.py                   # 核心ResNet模型
│   └── Resnet2.py                  # 二次训练ResNet模型
├── FeatureSelect/                   # 特征选择实验
│   ├── pcapResnetPacketSeed.py     # 特征选择核心模型
│   ├── pcapTrainResPacket_ES2_1.py # K=1特征训练脚本
│   ├── pcapTrainResPacket_ES2_2.py # K=2特征训练脚本
│   └── pcapTrainResPacket_ES2_4.py # K=4特征训练脚本
├── FeatureSelect2/                  # 高级特征选择
│   ├── pcapResnetSeed_factor.py    # 高级特征选择模型
│   └── TrainResnet_factor.py       # 多K值训练脚本
├── Loss/                           # 损失函数实验
│   ├── pcapResnetPureSeed.py       # 纯ResNet实现
│   └── pcapTrainResPure_ce.py      # 交叉熵损失训练脚本
├── requirements.txt                # 依赖包列表
└── README_CN.md                    # 中文说明文档
```

## 🎯 功能详解

### 1. 特征选择功能
- **自动特征选择**: 通过缩放因子自动识别最重要的特征
- **早停策略**: 当特征选择稳定时自动停止训练
- **多K值测试**: 可以测试选择不同数量特征的效果

### 2. 损失函数支持
- **交叉熵损失 (CE)**: 标准分类损失
- **类别平衡损失 (CB)**: 处理不平衡数据集的损失函数
- **CB-Focal损失**: 结合类别平衡和Focal损失的复合损失函数

### 3. 贝叶斯优化
- **自动超参数调优**: 使用贝叶斯方法自动寻找最佳参数组合
- **特征子集优化**: 自动选择最优的特征子集

## 🚀 代码执行方法

### 方法1: 特征选择实验

#### 选择1个最重要特征：
```bash
cd NSL_KDD/resnet29/FeatureSelect
python pcapTrainResPacket_ES2_1.py
```

#### 选择2个最重要特征：
```bash
cd NSL_KDD/resnet29/FeatureSelect
python pcapTrainResPacket_ES2_2.py
```

#### 选择4个最重要特征：
```bash
cd NSL_KDD/resnet29/FeatureSelect
python pcapTrainResPacket_ES2_4.py
```

### 方法2: 高级特征选择

测试多个K值（1,2,4,8,16,24,32）的特征选择效果：
```bash
cd NSL_KDD/resnet29/FeatureSelect2
python TrainResnet_factor.py
```

### 方法3: 损失函数实验

使用交叉熵损失训练纯ResNet：
```bash
cd NSL_KDD/resnet29/Loss
python pcapTrainResPure_ce.py
```

### 方法4: 贝叶斯优化

运行贝叶斯优化实验：
```bash
cd NSL_KDD/resnet29/beyes
python BeyesResNet.py
```

## 📊 输出结果说明

### 训练过程输出
- **Epoch进度**: 显示每个epoch的训练损失和训练时间
- **特征选择进度**: 显示当前选择的重要特征索引
- **早停信息**: 当特征选择稳定时显示早停信息

### 测试结果输出
- **Top-K准确率**: Top-1, Top-3, Top-5的预测准确率
- **F1分数**: Macro-F1和Micro-F1分数
- **各类别准确率**: 每个类别的详细准确率统计

### 模型保存
- **模型文件**: 自动保存为`.h5`格式的TensorFlow模型
- **保存位置**: `./model/model_时间戳/model.h5`

## ⚙️ 参数配置

### 主要参数
- **SEED**: 随机种子，默认25
- **K**: 选择的特征数量
- **TRAIN_EPOCH**: 训练轮数，默认30
- **BATCH_SIZE**: 批次大小，默认128
- **LEARNING_RATE**: 学习率，默认0.0001

### 损失函数参数
- **BETA**: 类别平衡参数，默认0.9999
- **GAMMA**: Focal损失参数，默认1

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 设置使用CPU
   os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
   ```

2. **数据文件找不到**
   - 确保CSV文件在正确的相对路径位置
   - 检查文件名是否正确

3. **包版本冲突**
   ```bash
   # 创建虚拟环境
   python -m venv tf29_env
   tf29_env\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

## 📈 性能优化建议

1. **GPU加速**: 确保安装了CUDA和cuDNN以使用GPU加速
2. **内存管理**: 如果内存不足，可以减少BATCH_SIZE
3. **并行训练**: 可以同时运行多个不同K值的实验

## 📝 注意事项

- 确保数据文件格式正确（CSV格式，最后一列为标签）
- 训练过程中会自动创建model目录保存模型
- 建议先运行小规模实验验证环境配置正确
- 所有代码都支持Windows、Linux和macOS系统
