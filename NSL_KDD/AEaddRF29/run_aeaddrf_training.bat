@echo off
REM TensorFlow 2.9.0 AutoEncoder + Random Forest训练脚本 (Windows版本)
REM 用法: run_aeaddrf_training.bat [script_type] [SEED] [TRAIN_EPOCH] [selected_features]
REM 
REM 参数说明:
REM script_type: 脚本类型 ae/ae_factor/second_train (默认: ae)
REM SEED: 随机种子 (默认: 25)
REM TRAIN_EPOCH: 训练轮数 (默认: 30)
REM selected_features: 选择的特征数量，仅用于second_train (默认: 16)

setlocal enabledelayedexpansion

REM 设置默认参数
if "%1"=="" (set SCRIPT_TYPE=ae) else (set SCRIPT_TYPE=%1)
if "%2"=="" (set SEED=25) else (set SEED=%2)
if "%3"=="" (set TRAIN_EPOCH=30) else (set TRAIN_EPOCH=%3)
if "%4"=="" (set SELECTED_FEATURES=16) else (set SELECTED_FEATURES=%4)

echo =========================================
echo TensorFlow 2.9.0 AutoEncoder + Random Forest训练脚本
echo =========================================
echo 参数配置:
echo Script Type (脚本类型): %SCRIPT_TYPE%
echo SEED (随机种子): %SEED%
echo TRAIN_EPOCH (训练轮数): %TRAIN_EPOCH%
echo Selected Features (选择特征数): %SELECTED_FEATURES%
echo =========================================

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境
    pause
    exit /b 1
)

REM 检查TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow版本: {tf.__version__}')" 2>nul
if errorlevel 1 (
    echo 警告: TensorFlow未安装或版本不兼容
    echo 请运行: pip install tensorflow==2.9.0
)

REM 根据脚本类型选择执行路径
if "%SCRIPT_TYPE%"=="ae" (
    echo 执行基础AutoEncoder + Random Forest训练...
    
    REM 创建临时脚本文件，动态设置参数
    set TEMP_SCRIPT=temp_ae_%SEED%_%TRAIN_EPOCH%.py
    copy TrainpcapAEAddRF.py !TEMP_SCRIPT! >nul
    
    REM 替换脚本中的参数
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'SEED = [0-9]+', 'SEED = %SEED%' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'TRAIN_EPOCH = [0-9]+', 'TRAIN_EPOCH = %TRAIN_EPOCH%' | Set-Content !TEMP_SCRIPT!"
    
    echo 执行脚本: !TEMP_SCRIPT!
    python !TEMP_SCRIPT!
    
    REM 清理临时文件
    del !TEMP_SCRIPT!
    
) else if "%SCRIPT_TYPE%"=="ae_factor" (
    echo 执行带特征选择的AutoEncoder + Random Forest训练...
    
    REM 创建临时脚本文件
    set TEMP_SCRIPT=temp_ae_factor_%SEED%_%TRAIN_EPOCH%.py
    copy TrainpcapAEAddRF_factor.py !TEMP_SCRIPT! >nul
    
    REM 替换脚本中的参数
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'SEED = [0-9]+', 'SEED = %SEED%' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'TRAIN_EPOCH = [0-9]+', 'TRAIN_EPOCH = %TRAIN_EPOCH%' | Set-Content !TEMP_SCRIPT!"
    
    echo 执行脚本: !TEMP_SCRIPT!
    python !TEMP_SCRIPT!
    
    REM 清理临时文件
    del !TEMP_SCRIPT!
    
) else if "%SCRIPT_TYPE%"=="second_train" (
    echo 执行第二阶段训练...
    cd FeatureSelect
    
    REM 创建临时脚本文件
    set TEMP_SCRIPT=temp_second_train_%SEED%_%TRAIN_EPOCH%_%SELECTED_FEATURES%.py
    copy SecondTrain.py !TEMP_SCRIPT! >nul
    
    REM 替换脚本中的参数
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'SEED = [0-9]+', 'SEED = %SEED%' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'TRAIN_EPOCH = [0-9]+', 'TRAIN_EPOCH = %TRAIN_EPOCH%' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'selected_features = list\(range\([0-9]+\)\)', 'selected_features = list(range(%SELECTED_FEATURES%))' | Set-Content !TEMP_SCRIPT!"
    
    echo 执行脚本: !TEMP_SCRIPT!
    python !TEMP_SCRIPT!
    
    REM 清理临时文件
    del !TEMP_SCRIPT!
    
) else (
    echo 错误: 未知的脚本类型 '%SCRIPT_TYPE%'
    echo 支持的类型: ae, ae_factor, second_train
    pause
    exit /b 1
)

echo =========================================
echo 训练完成！
echo =========================================
pause
