# TensorFlow 2.9.0 兼容的AutoEncoder + Random Forest实现

本目录包含从TensorFlow 2.4.0迁移到TensorFlow 2.9.0的AutoEncoder + Random Forest实现，支持完全可复现的结果。

## 🔧 随机种子设置（确保结果可复现）

所有代码使用TensorFlow 2.9.0推荐的随机种子设置来确保可复现的结果：

```python
def set_deterministic_seed(seed):
    """设置所有随机种子确保结果可复现"""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
```

### 测试随机种子配置

运行测试脚本验证随机种子设置：

```bash
cd NSL_KDD/AEaddRF29
python ../VGG29/test_basic.py
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install tensorflow==2.9.0
pip install numpy scikit-learn matplotlib pandas
```

或一次性安装所有依赖：

```bash
pip install -r requirements.txt
```

### 2. 运行训练脚本

#### 方法1：使用Shell脚本（推荐）

**Linux/Mac:**
```bash
# 基础AutoEncoder + Random Forest训练
./run_aeaddrf_training.sh

# 指定参数：ae_factor, SEED=25, TRAIN_EPOCH=30
./run_aeaddrf_training.sh ae_factor 25 30

# 使用选定特征的第二阶段训练
./run_aeaddrf_training.sh second_train 25 30 16
```

**Windows:**
```cmd
# 基础AutoEncoder + Random Forest训练
run_aeaddrf_training.bat

# 指定参数
run_aeaddrf_training.bat ae_factor 25 30

# 使用选定特征的第二阶段训练
run_aeaddrf_training.bat second_train 25 30 16
```

#### 方法2：直接Python执行

```bash
# 基础AutoEncoder + Random Forest训练
python TrainpcapAEAddRF.py

# 带特征选择的AutoEncoder + Random Forest
python TrainpcapAEAddRF_factor.py

# 第二阶段训练
cd FeatureSelect
python SecondTrain.py
```

## 📁 目录结构

```
AEaddRF29/
├── FeatureSelect/              # 第二阶段训练
│   ├── AEAddRF2.py             # 用于选定特征的AutoEncoder模型
│   └── SecondTrain.py          # 第二阶段训练脚本
├── model/                      # 模型存储目录
├── pcapAEAddRF.py              # 基础AutoEncoder + Random Forest模型
├── pcapAEAddRF_factor.py       # 带特征选择的AutoEncoder + Random Forest
├── TrainpcapAEAddRF.py         # 基础训练脚本
├── TrainpcapAEAddRF_factor.py  # 特征选择训练脚本
├── run_aeaddrf_training.sh     # Linux/Mac训练脚本
├── run_aeaddrf_training.bat    # Windows训练脚本
├── requirements.txt            # Python依赖
└── README.md                   # 说明文档
```

## 🎯 可配置参数

Shell脚本支持以下参数：

### AutoEncoder + Random Forest参数

- **script_type**: 训练脚本类型（默认："ae"）
  - `"ae"`: 基础AutoEncoder + Random Forest
  - `"ae_factor"`: 带特征选择的AutoEncoder + Random Forest
  - `"second_train"`: 使用选定特征的第二阶段训练

- **SEED**: 随机种子（默认：25）
  - 确保多次运行结果一致
  - 支持任何整数值

- **TRAIN_EPOCH**: 训练轮数（默认：30）
  - 控制训练时长
  - 更高值 = 更长训练时间

- **selected_features**: 第二阶段选择的特征数量（默认：16）
  - 仅用于second_train脚本类型
  - 应与第一阶段选择的特征数量匹配

## 🏗️ AutoEncoder架构

AutoEncoder模型保持原始架构：

```
输入(41特征) → 编码器 → 潜在表示(K=32) → 解码器 → 重构(41特征)
                                    ↓
                              分类器 → 分类(2类)
```

**编码器网络:**
- Dense(32) → Dropout → Dense(16) → Dropout → Dense(K=32)

**解码器网络:**
- Dense(16) → Dropout → Dense(32) → Dropout → Dense(41)

**分类器网络:**
- Dense(64) → Dropout → Dense(32) → Dropout → Dense(2)

## 🔬 主要特性

### 1. AutoEncoder + Random Forest混合方法
- **AutoEncoder**: 学习输入数据的压缩表示
- **Random Forest**: 在编码特征上训练进行分类
- **混合方法**: 结合深度学习特征提取与集成学习

### 2. 特征选择
- 缩放因子的L1正则化
- 基于学习重要性的Top-K特征选择
- 两阶段训练：特征选择 → 模型重训练

### 3. 损失函数
- **重构损失**: AutoEncoder的均方误差
- **分类损失**: 分类器的稀疏分类交叉熵
- **组合损失**: 重构 + 分类 + L1正则化

### 4. 可复现性
- 确定性随机种子配置
- TensorFlow 2.9.0操作确定性
- 多次运行结果一致

## 📊 使用示例

### 示例1：基础训练

```bash
# 训练基础AutoEncoder + Random Forest
./run_aeaddrf_training.sh ae 25 30
```

### 示例2：特征选择训练

```bash
# 带特征选择训练
./run_aeaddrf_training.sh ae_factor 25 30
```

### 示例3：第二阶段训练

```bash
# 使用16个选定特征训练第二阶段
./run_aeaddrf_training.sh second_train 25 30 16
```

### 示例4：可复现性测试

```bash
# 多次运行相同配置 - 应获得相同结果
./run_aeaddrf_training.sh ae 42 20
./run_aeaddrf_training.sh ae 42 20
./run_aeaddrf_training.sh ae 42 20
```

## 🚨 重要说明

1. **数据文件**: 确保`train_data.csv`和`test_data.csv`在父目录（`../`）中
2. **GPU内存**: 代码包含TensorFlow 2.x的GPU内存增长配置
3. **模型保存**: 模型自动保存在`./model/`目录中，带时间戳
4. **可复现性**: 使用相同参数和种子时，所有结果应完全相同

## 🐛 故障排除

### 常见问题

1. **TensorFlow版本**: 确保使用TensorFlow 2.9.0
2. **CUDA兼容性**: 检查CUDA版本与TensorFlow 2.9.0的兼容性
3. **内存问题**: 如遇到OOM错误，减少批次大小
4. **文件路径**: 验证数据文件在正确的相对路径中

### 性能优化建议

1. 可用时使用GPU加速
2. 根据可用内存调整批次大小
3. 训练期间监控GPU利用率
4. 为数据集使用适当的训练轮数

## 📈 预期结果

- **AutoEncoder**: 学习有意义的压缩表示
- **Random Forest**: 在编码特征上提供鲁棒分类
- **特征选择**: 识别分类最重要的特征
- **可复现性**: 相同参数下多次运行结果相同
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

## 🤝 贡献指南

修改代码时请：
1. 保持随机种子配置
2. 保留AutoEncoder架构
3. 保持参数可配置性
4. 修改后测试可复现性

## 📄 许可证

此代码是TensorFlow 2.9.0迁移项目的一部分。
