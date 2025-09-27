#!/bin/bash

# TensorFlow 2.9.0 AutoEncoder + Random Forest训练脚本
# 用法: ./run_aeaddrf_training.sh [script_type] [SEED] [TRAIN_EPOCH] [selected_features]
# 
# 参数说明:
# script_type: 脚本类型 ae/ae_factor/second_train (默认: ae)
# SEED: 随机种子 (默认: 25)
# TRAIN_EPOCH: 训练轮数 (默认: 30)
# selected_features: 选择的特征数量，仅用于second_train (默认: 16)

# 设置默认参数
SCRIPT_TYPE=${1:-"ae"}
SEED=${2:-25}
TRAIN_EPOCH=${3:-30}
SELECTED_FEATURES=${4:-16}

echo "========================================="
echo "TensorFlow 2.9.0 AutoEncoder + Random Forest训练脚本"
echo "========================================="
echo "参数配置:"
echo "Script Type (脚本类型): $SCRIPT_TYPE"
echo "SEED (随机种子): $SEED"
echo "TRAIN_EPOCH (训练轮数): $TRAIN_EPOCH"
echo "Selected Features (选择特征数): $SELECTED_FEATURES"
echo "========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi

# 检查TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow版本: {tf.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: TensorFlow未安装或版本不兼容"
    echo "请运行: pip install tensorflow==2.9.0"
fi

# 根据脚本类型选择执行路径
case $SCRIPT_TYPE in
    "ae")
        echo "执行基础AutoEncoder + Random Forest训练..."
        
        # 创建临时脚本文件，动态设置参数
        TEMP_SCRIPT="temp_ae_${SEED}_${TRAIN_EPOCH}.py"
        cp TrainpcapAEAddRF.py $TEMP_SCRIPT
        
        # 替换脚本中的参数
        sed -i "s/SEED = [0-9]*/SEED = $SEED/g" $TEMP_SCRIPT
        sed -i "s/TRAIN_EPOCH = [0-9]*/TRAIN_EPOCH = $TRAIN_EPOCH/g" $TEMP_SCRIPT
        
        echo "执行脚本: $TEMP_SCRIPT"
        python $TEMP_SCRIPT
        
        # 清理临时文件
        rm $TEMP_SCRIPT
        ;;
        
    "ae_factor")
        echo "执行带特征选择的AutoEncoder + Random Forest训练..."
        
        # 创建临时脚本文件
        TEMP_SCRIPT="temp_ae_factor_${SEED}_${TRAIN_EPOCH}.py"
        cp TrainpcapAEAddRF_factor.py $TEMP_SCRIPT
        
        # 替换脚本中的参数
        sed -i "s/SEED = [0-9]*/SEED = $SEED/g" $TEMP_SCRIPT
        sed -i "s/TRAIN_EPOCH = [0-9]*/TRAIN_EPOCH = $TRAIN_EPOCH/g" $TEMP_SCRIPT
        
        echo "执行脚本: $TEMP_SCRIPT"
        python $TEMP_SCRIPT
        
        # 清理临时文件
        rm $TEMP_SCRIPT
        ;;
        
    "second_train")
        echo "执行第二阶段训练..."
        cd FeatureSelect
        
        # 创建临时脚本文件
        TEMP_SCRIPT="temp_second_train_${SEED}_${TRAIN_EPOCH}_${SELECTED_FEATURES}.py"
        cp SecondTrain.py $TEMP_SCRIPT
        
        # 替换脚本中的参数
        sed -i "s/SEED = [0-9]*/SEED = $SEED/g" $TEMP_SCRIPT
        sed -i "s/TRAIN_EPOCH = [0-9]*/TRAIN_EPOCH = $TRAIN_EPOCH/g" $TEMP_SCRIPT
        sed -i "s/selected_features = list(range([0-9]*))/selected_features = list(range($SELECTED_FEATURES))/g" $TEMP_SCRIPT
        
        echo "执行脚本: $TEMP_SCRIPT"
        python $TEMP_SCRIPT
        
        # 清理临时文件
        rm $TEMP_SCRIPT
        ;;
        
    *)
        echo "错误: 未知的脚本类型 '$SCRIPT_TYPE'"
        echo "支持的类型: ae, ae_factor, second_train"
        exit 1
        ;;
esac

echo "========================================="
echo "训练完成！"
echo "========================================="
