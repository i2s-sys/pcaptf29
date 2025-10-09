@echo off
REM TensorFlow 2.9.0 ResNet训练脚本 (Windows版本)
REM 用法: run_resnet_training.bat [script_type] [K] [loss_type] [ES_THRESHOLD] [SEED]
REM 
REM 参数说明:
REM script_type: 脚本类型 feature_select/loss_ce/loss_cb/loss_cb_focal (默认: feature_select)
REM K: 特征选择数量 (默认: 32)
REM loss_type: 损失函数类型 ce/cb/cb_focal_loss (默认: cb_focal_loss)
REM ES_THRESHOLD: 早停阈值 (默认: 3)
REM SEED: 随机种子 (默认: 25)

setlocal enabledelayedexpansion

REM 设置默认参数
if "%1"=="" (set SCRIPT_TYPE=feature_select) else (set SCRIPT_TYPE=%1)
if "%2"=="" (set K=32) else (set K=%2)
if "%3"=="" (set LOSS_TYPE=cb_focal_loss) else (set LOSS_TYPE=%3)
if "%4"=="" (set ES_THRESHOLD=3) else (set ES_THRESHOLD=%4)
if "%5"=="" (set SEED=25) else (set SEED=%5)

echo =========================================
echo TensorFlow 2.9.0 ResNet训练脚本
echo =========================================
echo 参数配置:
echo Script Type (脚本类型): %SCRIPT_TYPE%
echo K (特征选择数量): %K%
echo Loss Type (损失函数): %LOSS_TYPE%
echo ES_THRESHOLD (早停阈值): %ES_THRESHOLD%
echo SEED (随机种子): %SEED%
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
if "%SCRIPT_TYPE%"=="feature_select" (
    echo 执行特征选择ResNet训练...
    cd FeatureSelect
    
    REM 创建临时脚本文件，动态设置参数
    set TEMP_SCRIPT=temp_feature_select_%K%_%ES_THRESHOLD%_%SEED%.py
    copy pcapTrainResPacket_ES2_32.py !TEMP_SCRIPT! >nul
    
    REM 替换脚本中的参数
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'K = [0-9]+', 'K = %K%' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'ES_THRESHOLD = [0-9]+', 'ES_THRESHOLD = %ES_THRESHOLD%' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'SEED = [0-9]+', 'SEED = %SEED%' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'model = ResNet\(K, \"cb_focal_loss\", ES_THRESHOLD, SEED\)', 'model = ResNet(%K%, \"%LOSS_TYPE%\", %ES_THRESHOLD%, %SEED%)' | Set-Content !TEMP_SCRIPT!"
    
    echo 执行脚本: !TEMP_SCRIPT!
    python !TEMP_SCRIPT!
    
    REM 清理临时文件
    del !TEMP_SCRIPT!
    
) else if "%SCRIPT_TYPE%"=="loss_ce" (
    echo 执行Cross-Entropy损失ResNet训练...
    cd Loss
    
    REM 创建临时脚本文件
    set TEMP_SCRIPT=temp_loss_ce_%SEED%.py
    copy pcapTrainResPure_ce.py !TEMP_SCRIPT! >nul
    
    REM 替换脚本中的参数
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'SEED = [0-9]+', 'SEED = %SEED%' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'model = ResNet\(\"ce\", SEED, 0.9999, 1\)', 'model = ResNet(\"ce\", %SEED%, 0.9999, 1)' | Set-Content !TEMP_SCRIPT!"
    
    echo 执行脚本: !TEMP_SCRIPT!
    python !TEMP_SCRIPT!
    
    REM 清理临时文件
    del !TEMP_SCRIPT!
    
) else if "%SCRIPT_TYPE%"=="loss_cb" (
    echo 执行Class-Balanced损失ResNet训练...
    cd Loss
    
    REM 创建临时脚本文件
    set TEMP_SCRIPT=temp_loss_cb_%SEED%.py
    copy pcapTrainResPure_ce.py !TEMP_SCRIPT! >nul
    
    REM 替换脚本中的参数
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'SEED = [0-9]+', 'SEED = %SEED%' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'model = ResNet\(\"ce\", SEED, 0.9999, 1\)', 'model = ResNet(\"cb\", %SEED%, 0.9999, 1)' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace '开始训练ResNet模型（Cross-Entropy Loss）', '开始训练ResNet模型（Class-Balanced Loss）' | Set-Content !TEMP_SCRIPT!"
    
    echo 执行脚本: !TEMP_SCRIPT!
    python !TEMP_SCRIPT!
    
    REM 清理临时文件
    del !TEMP_SCRIPT!
    
) else if "%SCRIPT_TYPE%"=="loss_cb_focal" (
    echo 执行Class-Balanced Focal损失ResNet训练...
    cd Loss
    
    REM 创建临时脚本文件
    set TEMP_SCRIPT=temp_loss_cb_focal_%SEED%.py
    copy pcapTrainResPure_ce.py !TEMP_SCRIPT! >nul
    
    REM 替换脚本中的参数
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'SEED = [0-9]+', 'SEED = %SEED%' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'model = ResNet\(\"ce\", SEED, 0.9999, 1\)', 'model = ResNet(\"cb_focal_loss\", %SEED%, 0.9999, 1)' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace '开始训练ResNet模型（Cross-Entropy Loss）', '开始训练ResNet模型（Class-Balanced Focal Loss）' | Set-Content !TEMP_SCRIPT!"
    
    echo 执行脚本: !TEMP_SCRIPT!
    python !TEMP_SCRIPT!
    
    REM 清理临时文件
    del !TEMP_SCRIPT!
    
) else (
    echo 错误: 未知的脚本类型 '%SCRIPT_TYPE%'
    echo 支持的类型: feature_select, loss_ce, loss_cb, loss_cb_focal
    pause
    exit /b 1
)

echo =========================================
echo 训练完成！
echo =========================================
pause
