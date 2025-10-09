# UNSW-IoT ResNet29 - TensorFlow 2.9.0 迁移版本

本目录包含从TensorFlow 2.4.0升级到TensorFlow 2.9.0的UNSW-IoT数据集ResNet实现。

## 🚀 功能特性

- **TensorFlow 2.9.0兼容**: 完全从TF 2.4.0迁移到TF 2.9.0
- **模型可复现**: 固定随机种子确保结果一致
- **特征选择**: 基于L1正则化的特征选择，带早停机制
- **类别平衡焦点损失**: 处理29类不平衡数据集
- **ResNet架构**: 深度残差网络，4个块，每个块[2,2,2,2]层

## 📁 目录结构

```
UNSW-IoT/Resnet29/
├── FeatureSelect/
│   ├── pcapResnetPacketSeed.py      # 主要ResNet模型，带特征选择
│   └── pcapTrainResPacket_ES2_32.py # K=32特征训练脚本
├── requirements.txt                  # Python依赖包
└── README_CN.md                     # 中文说明文档
```

## 🔧 安装

1. **安装Python依赖包**:
```bash
pip install -r requirements.txt
```

2. **确保数据文件存在**:
   - `../train_data.csv` - 训练数据（72个特征 + 1个标签）
   - `../test_data.csv` - 测试数据（72个特征 + 1个标签）

## 🎯 使用方法

### 特征选择训练

训练ResNet模型进行特征选择（K=32个特征）：

```bash
cd UNSW-IoT/Resnet29/FeatureSelect
python pcapTrainResPacket_ES2_32.py
```

### 参数说明

- **K**: 选择的特征数量（默认：32）
- **ES_THRESHOLD**: 早停阈值（默认：3）
- **SEED**: 随机种子，确保可复现（默认：25）
- **TRAIN_EPOCH**: 最大训练轮数（默认：30）

## 🏗️ 模型架构

### ResNet结构
- **输入**: 72个特征 → 重塑为(72, 1, 1)
- **初始卷积**: 64个滤波器，3x3核
- **ResNet块**: 4个块，每个块[2,2,2,2]层
  - 块1: 64个滤波器
  - 块2: 128个滤波器，步长=2
  - 块3: 256个滤波器，步长=2
  - 块4: 512个滤波器，步长=2
- **全局平均池化**: 减少空间维度
- **全连接层**: 29类输出

### 特征选择
- **缩放因子**: 每个特征的可学习参数
- **L1正则化**: 鼓励缩放因子的稀疏性
- **早停机制**: 基于连续epochs中top-K特征的交集

## 📊 数据集信息

- **数据集**: UNSW-IoT
- **特征**: 72个网络流量特征
- **类别**: 29种IoT设备类型（0-28）
- **训练**: 类别平衡，使用有效数量加权
- **损失函数**: 类别平衡焦点损失（β=0.999, γ=1）

## 🔄 迁移改进

### 从TensorFlow 2.4.0到2.9.0

1. **移除tf.compat.v1**:
   - `tf.compat.v1.Session()` → `tf.keras.Model`子类化
   - `tf.compat.v1.placeholder()` → 直接张量操作
   - `tf.compat.v1.disable_eager_execution()` → 移除（默认启用eager execution）

2. **更新训练循环**:
   - `sess.run()` → `tf.GradientTape()`配合`@tf.function`
   - 手动梯度计算 → `optimizer.apply_gradients()`

3. **模型可复现性**:
   - 添加`set_deterministic_seed()`函数
   - 使用`tf.keras.utils.set_random_seed()`和`tf.config.experimental.enable_op_determinism()`

4. **修复训练问题**:
   - 添加数据打乱: `np.random.shuffle(self.train_data)`
   - 修复F1分数返回顺序: `(macro_f1, micro_f1)`
   - 添加F1分数历史记录

## 🎛️ 主要类

### Resnet
- 特征选择的主要模型
- 包含L1正则化的缩放因子
- 基于特征交集的早停机制

### Resnet2
- 使用选定特征的第二阶段训练
- 简化架构，无缩放因子
- 用于最终模型训练

### BasicBlock
- 带跳跃连接的ResNet基本块
- 批归一化和ReLU激活
- 可配置步长用于下采样

## 📈 性能指标

- **Top-1准确率**: 主要分类准确率
- **Top-3准确率**: Top-3预测准确率
- **Top-5准确率**: Top-5预测准确率
- **Macro-F1**: 所有类别的平均F1分数
- **Micro-F1**: 考虑所有样本的全局F1分数

## 🔧 配置

### 超参数
```python
DATA_DIM = 72          # 输入特征维度
OUTPUT_DIM = 29        # 类别数量
LEARNING_RATE = 0.0001 # Adam优化器学习率
BATCH_SIZE = 128       # 训练批次大小
BETA = 0.999          # 类别平衡参数
GAMMA = 1             # 焦点损失参数
```

### 特征选择
```python
K = 32                # 选择的特征数量
ES_THRESHOLD = 3      # 早停阈值
L1_REGULARIZATION = 0.001  # 缩放因子的L1惩罚
```

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**:
   - 将`BATCH_SIZE`从128减少到64或32
   - 设置`CUDA_VISIBLE_DEVICES`使用特定GPU

2. **数据文件未找到**:
   - 确保`train_data.csv`和`test_data.csv`存在于父目录
   - 检查`TRAIN_FILE`和`TEST_FILE`常量中的文件路径

3. **可复现性问题**:
   - 验证`set_deterministic_seed()`被调用
   - 检查`tf.config.experimental.enable_op_determinism()`已启用

## 📝 示例输出

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

## 🧪 测试验证

创建了测试脚本来验证迁移效果：

```bash
cd UNSW-IoT/Resnet29
python test_unsw_resnet29_simple.py
```

## 🤝 贡献

修改代码时请注意：
1. 保持ResNet架构结构
2. 确保固定种子的可复现性
3. 测试特征选择和最终训练两个阶段
4. 更新任何参数更改的文档

## 📄 许可证

此代码是UNSW-IoT ResNet实现的一部分，用于网络流量分类。