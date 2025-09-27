@echo off
REM TensorFlow 2.9.0 VGG训练脚本 (Windows版本)
REM 用法: run_vgg_training.bat [K] [loss_type] [ES_THRESHOLD] [SEED] [script_type]
REM 
REM 参数说明:
REM K: 特征选择数量 (默认: 1)
REM loss_type: 损失函数类型 ce/cb/cb_focal_loss (默认: cb_focal_loss)  
REM ES_THRESHOLD: 早停阈值 (默认: 3)
REM SEED: 随机种子 (默认: 25)
REM script_type: 脚本类型 feature/loss/beyes (默认: feature)

setlocal enabledelayedexpansion

REM 设置默认参数
if "%1"=="" (set K=1) else (set K=%1)
if "%2"=="" (set LOSS_TYPE=cb_focal_loss) else (set LOSS_TYPE=%2)
if "%3"=="" (set ES_THRESHOLD=3) else (set ES_THRESHOLD=%3)
if "%4"=="" (set SEED=25) else (set SEED=%4)
if "%5"=="" (set SCRIPT_TYPE=feature) else (set SCRIPT_TYPE=%5)

echo =========================================
echo TensorFlow 2.9.0 VGG训练脚本
echo =========================================
echo 参数配置:
echo K (特征数量): %K%
echo Loss Type (损失函数): %LOSS_TYPE%
echo ES_THRESHOLD (早停阈值): %ES_THRESHOLD%
echo SEED (随机种子): %SEED%
echo Script Type (脚本类型): %SCRIPT_TYPE%
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
if "%SCRIPT_TYPE%"=="feature" (
    echo 执行特征选择训练...
    cd FeatureSelect
    
    REM 根据K值选择相应的训练脚本
    if %K%==1 (
        set SCRIPT_NAME=pcapTrainVGG_ES3_1.py
    ) else if %K%==2 (
        set SCRIPT_NAME=pcapTrainVGG_ES3_2.py
    ) else (
        echo 警告: K=%K% 没有对应的预定义脚本，使用K=1的脚本
        set SCRIPT_NAME=pcapTrainVGG_ES3_1.py
    )
    
    REM 创建临时脚本文件，动态设置参数
    set TEMP_SCRIPT=temp_train_K%K%_%LOSS_TYPE%_%SEED%.py
    copy !SCRIPT_NAME! !TEMP_SCRIPT! >nul
    
    REM 替换脚本中的参数 (Windows版本使用PowerShell)
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'K = [0-9]+', 'K = %K%' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'ES_THRESHOLD = [0-9]+', 'ES_THRESHOLD = %ES_THRESHOLD%' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'SEED = [0-9]+', 'SEED = %SEED%' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace '\"cb_focal_loss\"', '\"%LOSS_TYPE%\"' | Set-Content !TEMP_SCRIPT!"
    
    echo 执行脚本: !TEMP_SCRIPT!
    python !TEMP_SCRIPT!
    
    REM 清理临时文件
    del !TEMP_SCRIPT!
    
) else if "%SCRIPT_TYPE%"=="loss" (
    echo 执行损失函数实验...
    cd Loss
    
    REM 根据损失函数类型选择脚本
    if "%LOSS_TYPE%"=="ce" (
        set SCRIPT_NAME=pcapTrainVGG_ce.py
    ) else if "%LOSS_TYPE%"=="cb" (
        set SCRIPT_NAME=pcapTrainVGG_cb.py
    ) else if "%LOSS_TYPE%"=="cb_focal_loss" (
        set SCRIPT_NAME=pcapTrainVGG_cb_focal.py
    ) else (
        echo 使用默认的CE损失脚本
        set SCRIPT_NAME=pcapTrainVGG_ce.py
    )
    
    REM 创建临时脚本文件
    set TEMP_SCRIPT=temp_loss_%LOSS_TYPE%_%SEED%.py
    if exist !SCRIPT_NAME! (
        copy !SCRIPT_NAME! !TEMP_SCRIPT! >nul
    ) else (
        REM 如果特定脚本不存在，使用通用模板
        copy pcapTrainVGG_ce.py !TEMP_SCRIPT! >nul
    )
    
    REM 替换脚本中的参数
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'SEED = [0-9]+', 'SEED = %SEED%' | Set-Content !TEMP_SCRIPT!"
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace '\"ce\"', '\"%LOSS_TYPE%\"' | Set-Content !TEMP_SCRIPT!"
    
    echo 执行脚本: !TEMP_SCRIPT!
    python !TEMP_SCRIPT!
    
    REM 清理临时文件
    del !TEMP_SCRIPT!
    
) else if "%SCRIPT_TYPE%"=="beyes" (
    echo 执行贝叶斯优化...
    cd beyes
    
    REM 创建临时脚本文件
    set TEMP_SCRIPT=temp_beyes_%SEED%.py
    copy BeyesVGG.py !TEMP_SCRIPT! >nul
    
    REM 替换脚本中的参数
    powershell -Command "(Get-Content !TEMP_SCRIPT!) -replace 'SEED = [0-9]+', 'SEED = %SEED%' | Set-Content !TEMP_SCRIPT!"
    
    echo 执行脚本: !TEMP_SCRIPT!
    python !TEMP_SCRIPT!
    
    REM 清理临时文件
    del !TEMP_SCRIPT!
    
) else (
    echo 错误: 未知的脚本类型 '%SCRIPT_TYPE%'
    echo 支持的类型: feature, loss, beyes
    pause
    exit /b 1
)

echo =========================================
echo 训练完成！
echo =========================================
pause
