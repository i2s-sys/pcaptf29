#!/bin/bash

# TensorFlow 2.9.0 VGG训练脚本
# 用法: ./run_vgg_training.sh [K] [loss_type] [ES_THRESHOLD] [SEED] [script_type]
# 
# 参数说明:
# K: 特征选择数量 (默认: 1)
# loss_type: 损失函数类型 ce/cb/cb_focal_loss (默认: cb_focal_loss)  
# ES_THRESHOLD: 早停阈值 (默认: 3)
# SEED: 随机种子 (默认: 25)
# script_type: 脚本类型 feature/loss/beyes (默认: feature)

# 设置默认参数
K=${1:-1}
LOSS_TYPE=${2:-"cb_focal_loss"}
ES_THRESHOLD=${3:-3}
SEED=${4:-25}
SCRIPT_TYPE=${5:-"feature"}

echo "========================================="
echo "TensorFlow 2.9.0 VGG训练脚本"
echo "========================================="
echo "参数配置:"
echo "K (特征数量): $K"
echo "Loss Type (损失函数): $LOSS_TYPE"
echo "ES_THRESHOLD (早停阈值): $ES_THRESHOLD"
echo "SEED (随机种子): $SEED"
echo "Script Type (脚本类型): $SCRIPT_TYPE"
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
    "feature")
        echo "执行特征选择训练..."
        cd FeatureSelect
        
        # 根据K值选择相应的训练脚本
        if [ $K -eq 1 ]; then
            SCRIPT_NAME="pcapTrainVGG_ES3_1.py"
        elif [ $K -eq 2 ]; then
            SCRIPT_NAME="pcapTrainVGG_ES3_2.py"
        else
            echo "警告: K=$K 没有对应的预定义脚本，使用K=1的脚本"
            SCRIPT_NAME="pcapTrainVGG_ES3_1.py"
        fi
        
        # 创建临时脚本文件，动态设置参数
        TEMP_SCRIPT="temp_train_K${K}_${LOSS_TYPE}_${SEED}.py"
        cp $SCRIPT_NAME $TEMP_SCRIPT
        
        # 替换脚本中的参数
        sed -i "s/K = [0-9]*/K = $K/g" $TEMP_SCRIPT
        sed -i "s/ES_THRESHOLD = [0-9]*/ES_THRESHOLD = $ES_THRESHOLD/g" $TEMP_SCRIPT
        sed -i "s/SEED = [0-9]*/SEED = $SEED/g" $TEMP_SCRIPT
        sed -i "s/\"cb_focal_loss\"/\"$LOSS_TYPE\"/g" $TEMP_SCRIPT
        
        echo "执行脚本: $TEMP_SCRIPT"
        python $TEMP_SCRIPT
        
        # 清理临时文件
        rm $TEMP_SCRIPT
        ;;
        
    "loss")
        echo "执行损失函数实验..."
        cd Loss
        
        # 根据损失函数类型选择脚本
        case $LOSS_TYPE in
            "ce")
                SCRIPT_NAME="pcapTrainVGG_ce.py"
                ;;
            "cb")
                SCRIPT_NAME="pcapTrainVGG_cb.py"
                ;;
            "cb_focal_loss")
                SCRIPT_NAME="pcapTrainVGG_cb_focal.py"
                ;;
            *)
                echo "使用默认的CE损失脚本"
                SCRIPT_NAME="pcapTrainVGG_ce.py"
                ;;
        esac
        
        # 创建临时脚本文件
        TEMP_SCRIPT="temp_loss_${LOSS_TYPE}_${SEED}.py"
        if [ -f $SCRIPT_NAME ]; then
            cp $SCRIPT_NAME $TEMP_SCRIPT
        else
            # 如果特定脚本不存在，使用通用模板
            cp pcapTrainVGG_ce.py $TEMP_SCRIPT
        fi
        
        # 替换脚本中的参数
        sed -i "s/SEED = [0-9]*/SEED = $SEED/g" $TEMP_SCRIPT
        sed -i "s/\"ce\"/\"$LOSS_TYPE\"/g" $TEMP_SCRIPT
        
        echo "执行脚本: $TEMP_SCRIPT"
        python $TEMP_SCRIPT
        
        # 清理临时文件
        rm $TEMP_SCRIPT
        ;;
        
    "beyes")
        echo "执行贝叶斯优化..."
        cd beyes
        
        # 创建临时脚本文件
        TEMP_SCRIPT="temp_beyes_${SEED}.py"
        cp BeyesVGG.py $TEMP_SCRIPT
        
        # 替换脚本中的参数
        sed -i "s/SEED = [0-9]*/SEED = $SEED/g" $TEMP_SCRIPT
        
        echo "执行脚本: $TEMP_SCRIPT"
        python $TEMP_SCRIPT
        
        # 清理临时文件
        rm $TEMP_SCRIPT
        ;;
        
    *)
        echo "错误: 未知的脚本类型 '$SCRIPT_TYPE'"
        echo "支持的类型: feature, loss, beyes"
        exit 1
        ;;
esac

echo "========================================="
echo "训练完成！"
echo "========================================="
