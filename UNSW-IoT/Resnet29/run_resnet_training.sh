#!/bin/bash

# TensorFlow 2.9.0 ResNet训练脚本
# 用法: ./run_resnet_training.sh [script_type] [K] [loss_type] [ES_THRESHOLD] [SEED]
# 
# 参数说明:
# script_type: 脚本类型 feature_select/loss_ce/loss_cb/loss_cb_focal (默认: feature_select)
# K: 特征选择数量 (默认: 32)
# loss_type: 损失函数类型 ce/cb/cb_focal_loss (默认: cb_focal_loss)
# ES_THRESHOLD: 早停阈值 (默认: 3)
# SEED: 随机种子 (默认: 25)

# 设置默认参数
SCRIPT_TYPE=${1:-"feature_select"}
K=${2:-32}
LOSS_TYPE=${3:-"cb_focal_loss"}
ES_THRESHOLD=${4:-3}
SEED=${5:-25}

echo "========================================="
echo "TensorFlow 2.9.0 ResNet训练脚本"
echo "========================================="
echo "参数配置:"
echo "Script Type (脚本类型): $SCRIPT_TYPE"
echo "K (特征选择数量): $K"
echo "Loss Type (损失函数): $LOSS_TYPE"
echo "ES_THRESHOLD (早停阈值): $ES_THRESHOLD"
echo "SEED (随机种子): $SEED"
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
    "feature_select")
        echo "执行特征选择ResNet训练..."
        cd FeatureSelect
        
        # 创建临时脚本文件，动态设置参数
        TEMP_SCRIPT="temp_feature_select_${K}_${ES_THRESHOLD}_${SEED}.py"
        cp pcapTrainResPacket_ES2_32.py $TEMP_SCRIPT
        
        # 替换脚本中的参数
        sed -i "s/K = [0-9]*/K = $K/g" $TEMP_SCRIPT
        sed -i "s/ES_THRESHOLD = [0-9]*/ES_THRESHOLD = $ES_THRESHOLD/g" $TEMP_SCRIPT
        sed -i "s/SEED = [0-9]*/SEED = $SEED/g" $TEMP_SCRIPT
        sed -i "s/model = ResNet(K, \"cb_focal_loss\", ES_THRESHOLD, SEED)/model = ResNet($K, \"$LOSS_TYPE\", $ES_THRESHOLD, $SEED)/g" $TEMP_SCRIPT
        
        echo "执行脚本: $TEMP_SCRIPT"
        python $TEMP_SCRIPT
        
        # 清理临时文件
        rm $TEMP_SCRIPT
        ;;
        
    "loss_ce")
        echo "执行Cross-Entropy损失ResNet训练..."
        cd Loss
        
        # 创建临时脚本文件
        TEMP_SCRIPT="temp_loss_ce_${SEED}.py"
        cp pcapTrainResPure_ce.py $TEMP_SCRIPT
        
        # 替换脚本中的参数
        sed -i "s/SEED = [0-9]*/SEED = $SEED/g" $TEMP_SCRIPT
        sed -i "s/model = ResNet(\"ce\", SEED, 0.9999, 1)/model = ResNet(\"ce\", $SEED, 0.9999, 1)/g" $TEMP_SCRIPT
        
        echo "执行脚本: $TEMP_SCRIPT"
        python $TEMP_SCRIPT
        
        # 清理临时文件
        rm $TEMP_SCRIPT
        ;;
        
    "loss_cb")
        echo "执行Class-Balanced损失ResNet训练..."
        cd Loss
        
        # 创建临时脚本文件
        TEMP_SCRIPT="temp_loss_cb_${SEED}.py"
        cp pcapTrainResPure_ce.py $TEMP_SCRIPT
        
        # 替换脚本中的参数
        sed -i "s/SEED = [0-9]*/SEED = $SEED/g" $TEMP_SCRIPT
        sed -i "s/model = ResNet(\"ce\", SEED, 0.9999, 1)/model = ResNet(\"cb\", $SEED, 0.9999, 1)/g" $TEMP_SCRIPT
        sed -i "s/开始训练ResNet模型（Cross-Entropy Loss）/开始训练ResNet模型（Class-Balanced Loss）/g" $TEMP_SCRIPT
        
        echo "执行脚本: $TEMP_SCRIPT"
        python $TEMP_SCRIPT
        
        # 清理临时文件
        rm $TEMP_SCRIPT
        ;;
        
    "loss_cb_focal")
        echo "执行Class-Balanced Focal损失ResNet训练..."
        cd Loss
        
        # 创建临时脚本文件
        TEMP_SCRIPT="temp_loss_cb_focal_${SEED}.py"
        cp pcapTrainResPure_ce.py $TEMP_SCRIPT
        
        # 替换脚本中的参数
        sed -i "s/SEED = [0-9]*/SEED = $SEED/g" $TEMP_SCRIPT
        sed -i "s/model = ResNet(\"ce\", SEED, 0.9999, 1)/model = ResNet(\"cb_focal_loss\", $SEED, 0.9999, 1)/g" $TEMP_SCRIPT
        sed -i "s/开始训练ResNet模型（Cross-Entropy Loss）/开始训练ResNet模型（Class-Balanced Focal Loss）/g" $TEMP_SCRIPT
        
        echo "执行脚本: $TEMP_SCRIPT"
        python $TEMP_SCRIPT
        
        # 清理临时文件
        rm $TEMP_SCRIPT
        ;;
        
    *)
        echo "错误: 未知的脚本类型 '$SCRIPT_TYPE'"
        echo "支持的类型: feature_select, loss_ce, loss_cb, loss_cb_focal"
        exit 1
        ;;
esac

echo "========================================="
echo "训练完成！"
echo "========================================="
