# TensorFlow 2.9.0 兼容的VGG实现

本目录包含了从TensorFlow 2.4.0迁移到TensorFlow 2.9.0的VGG实现代码，支持完全可复现的结果。

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
cd NSL_KDD/VGG29
python ../resnet29/test_reproducibility.py
```

## 🚀 快速开始

### 1. 安装依赖包

```bash
pip install tensorflow==2.9.0
pip install numpy scikit-learn matplotlib hyperopt pandas
```

或者一次性安装所有依赖：

```bash
pip install -r requirements.txt
```

### 2. 运行训练脚本

#### 方法1：使用Shell脚本（推荐）

**Linux/Mac:**
```bash
# 使用默认参数
./run_vgg_training.sh

# 指定参数：K=2, cb_focal_loss, ES_THRESHOLD=3, SEED=25, 特征选择
./run_vgg_training.sh 2 cb_focal_loss 3 25 feature

# 运行损失函数实验
./run_vgg_training.sh 1 ce 3 25 loss

# 运行贝叶斯优化
./run_vgg_training.sh 1 cb_focal_loss 3 25 beyes
```

**Windows:**
```cmd
# 使用默认参数
run_vgg_training.bat

# 指定参数
run_vgg_training.bat 2 cb_focal_loss 3 25 feature

# 运行损失函数实验
run_vgg_training.bat 1 ce 3 25 loss

# 运行贝叶斯优化
run_vgg_training.bat 1 cb_focal_loss 3 25 beyes
```

#### 方法2：直接执行Python脚本

```bash
# 特征选择训练
cd FeatureSelect
python pcapTrainVGG_ES3_1.py  # K=1
python pcapTrainVGG_ES3_2.py  # K=2

# 损失函数实验
cd Loss
python pcapTrainVGG_ce.py

# 贝叶斯优化
cd beyes
python BeyesVGG.py
```

## 📁 目录结构

```
VGG29/
├── beyes/                    # 贝叶斯优化
│   ├── BeyesVGG.py          # 贝叶斯优化脚本
│   └── VGG2.py              # 第二阶段训练的VGG模型
├── FeatureSelect/           # 特征选择实验
│   ├── pcapVGGSeed.py       # 带特征选择的主VGG模型
│   ├── pcapTrainVGG_ES3_1.py # K=1的训练脚本
│   └── pcapTrainVGG_ES3_2.py # K=2的训练脚本
├── FeatureSelect2/          # 高级特征选择（待实现）
├── Loss/                    # 损失函数实验
│   ├── pcapVGGSeed.py       # 损失函数实验的VGG模型
│   └── pcapTrainVGG_ce.py   # 交叉熵损失训练
├── run_vgg_training.sh      # Linux/Mac训练脚本
├── run_vgg_training.bat     # Windows训练脚本
├── requirements.txt         # Python依赖包
└── README.md               # 英文说明文档
```

## 🎯 可配置参数

Shell脚本支持以下参数：

### VGG模型参数

```python
model = VGG(K, loss_type, ES_THRESHOLD, SEED)
```

- **K**: 选择的顶级特征数量（默认：1）
  - 支持的值：1, 2, 4, 8, 16, 32
  - 控制特征选择的粒度

- **loss_type**: 损失函数类型（默认："cb_focal_loss"）
  - `"ce"`: 交叉熵损失
  - `"cb"`: 类平衡交叉熵损失
  - `"cb_focal_loss"`: 类平衡焦点损失

- **ES_THRESHOLD**: 早停阈值（默认：3）
  - 等待特征交集稳定的轮数
  - 值越高 = 早停越耐心

- **SEED**: 随机种子（默认：25）
  - 确保多次运行结果一致
  - 支持任何整数值

### 脚本类型

- **feature**: 特征选择实验
- **loss**: 损失函数对比实验
- **beyes**: 贝叶斯优化超参数调优

## 🏗️ VGG架构

VGG模型保持原始架构：

```
输入(41特征) → 重塑 → VGG块 → 全连接层 → 输出(2类)

VGG块:
- Conv2D(64) → Conv2D(64) → MaxPool2D
- Conv2D(128) → Conv2D(128) → MaxPool2D  
- Conv2D(256) → Conv2D(256) → Conv2D(256) → MaxPool2D
- Conv2D(512)
- Flatten → Dense(1024) → Dropout → Dense(1024) → Dropout → Dense(2)
```

## 🔬 核心功能

### 1. 特征选择
- 对缩放因子进行L1正则化
- 基于学习到的重要性进行Top-K特征选择
- 两阶段训练：特征选择 → 模型重训练

### 2. 类平衡
- 针对不平衡数据集的有效数量计算
- 类平衡损失函数（CB, CB-Focal）
- 基于类频率的自动权重计算

### 3. 损失函数
- **交叉熵(CE)**: 标准分类损失
- **类平衡(CB)**: 针对不平衡数据的加权CE
- **类平衡焦点**: CB + 焦点损失，针对困难样本

### 4. 可复现性
- 确定性随机种子配置
- TensorFlow 2.9.0操作确定性
- 多次运行结果一致

## 📊 使用示例

### 示例1：不同K值的特征选择

```bash
# 测试不同的特征选择大小
./run_vgg_training.sh 1 cb_focal_loss 3 25 feature
./run_vgg_training.sh 2 cb_focal_loss 3 25 feature
./run_vgg_training.sh 4 cb_focal_loss 3 25 feature
```

### 示例2：损失函数对比

```bash
# 比较不同损失函数
./run_vgg_training.sh 1 ce 3 25 loss
./run_vgg_training.sh 1 cb 3 25 loss  
./run_vgg_training.sh 1 cb_focal_loss 3 25 loss
```

### 示例3：可复现性测试

```bash
# 多次运行相同配置 - 应该得到相同结果
./run_vgg_training.sh 2 cb_focal_loss 3 42 feature
./run_vgg_training.sh 2 cb_focal_loss 3 42 feature
./run_vgg_training.sh 2 cb_focal_loss 3 42 feature
```

## 📋 文件功能说明

### beyes目录
- **BeyesVGG.py**: 使用Hyperopt库进行贝叶斯优化，自动搜索最佳特征子集和超参数组合
- **VGG2.py**: 第二阶段训练的VGG模型，使用选定的特征进行精细训练

### FeatureSelect目录
- **pcapVGGSeed.py**: 核心VGG模型，包含特征选择机制，使用缩放因子和L1正则化学习特征重要性
- **pcapTrainVGG_ES3_1.py**: K=1的特征选择训练脚本，选择最重要的1个特征
- **pcapTrainVGG_ES3_2.py**: K=2的特征选择训练脚本，选择最重要的2个特征

### Loss目录
- **pcapVGGSeed.py**: 用于损失函数实验的纯VGG模型，不包含特征选择功能
- **pcapTrainVGG_ce.py**: 交叉熵损失函数的训练脚本

### 脚本文件
- **run_vgg_training.sh**: Linux/Mac系统的参数化训练脚本
- **run_vgg_training.bat**: Windows系统的参数化训练脚本
- **requirements.txt**: Python依赖包列表
- **README.md**: 英文说明文档
- **README_CN.md**: 中文说明文档（本文件）

## 🚨 重要注意事项

1. **数据文件**: 确保`train_data.csv`和`test_data.csv`在父目录中（`../../`）
2. **GPU内存**: 代码包含TensorFlow 2.x的GPU内存增长配置
3. **模型保存**: 模型会自动保存到`./model/`目录，带时间戳
4. **可复现性**: 使用相同参数和种子时，所有结果应该完全相同

## 🐛 故障排除

### 常见问题

1. **TensorFlow版本**: 确保使用TensorFlow 2.9.0
2. **CUDA兼容性**: 检查CUDA版本与TensorFlow 2.9.0的兼容性
3. **内存问题**: 如果遇到OOM错误，减少批次大小
4. **文件路径**: 验证数据文件在正确的相对路径中

### 性能优化建议

1. 在可用时使用GPU加速
2. 根据可用内存调整批次大小
3. 训练期间监控GPU利用率
4. 使用早停防止过拟合

## 📈 预期结果

- **特征选择**: 识别分类最重要的特征
- **损失对比**: CB-Focal在不平衡数据上通常表现最佳
- **可复现性**: 相同参数下多次运行结果完全相同
- **训练时间**: 根据硬件和配置而变化

## 🔄 迁移改进

相比原始TensorFlow 2.4.0代码，主要改进包括：

1. **移除tf.compat.v1依赖**: 使用原生TensorFlow 2.x API
2. **Eager Execution**: 默认启用，提高调试体验
3. **模型子类化**: 使用tf.keras.Model子类化，更清晰的架构
4. **@tf.function装饰器**: 提高训练性能
5. **确定性随机种子**: 确保完全可复现的结果
6. **GPU内存管理**: 改进的GPU内存配置
7. **参数化脚本**: 灵活的参数配置系统
8. **属性冲突修复**: 将`self.weights`重命名为`self.class_weights_dict`，解决与Keras Model内置属性的冲突

## 🤝 贡献指南

修改代码时请：
1. 保持随机种子配置
2. 保留VGG架构
3. 保持参数可配置性
4. 修改后测试可复现性

## 📄 许可证

此代码是TensorFlow 2.9.0迁移项目的一部分。
