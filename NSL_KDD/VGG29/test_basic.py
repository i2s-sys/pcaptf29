# 基础测试脚本（不依赖TensorFlow）
import os
import sys

def test_directory_structure():
    """测试目录结构"""
    print("=== 测试目录结构 ===")
    
    required_dirs = [
        'beyes',
        'FeatureSelect', 
        'FeatureSelect2',
        'Loss'
    ]
    
    required_files = [
        'beyes/BeyesVGG.py',
        'beyes/VGG2.py',
        'FeatureSelect/pcapVGGSeed.py',
        'FeatureSelect/pcapTrainVGG_ES3_1.py',
        'FeatureSelect/pcapTrainVGG_ES3_2.py',
        'Loss/pcapVGGSeed.py',
        'Loss/pcapTrainVGG_ce.py',
        'requirements.txt',
        'README.md',
        'README_CN.md',
        'run_vgg_training.sh',
        'run_vgg_training.bat'
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"✅ 目录 {dir_name} 存在")
        else:
            print(f"❌ 目录 {dir_name} 不存在")
            all_good = False
    
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"✅ 文件 {file_name} 存在")
        else:
            print(f"❌ 文件 {file_name} 不存在")
            all_good = False
    
    return all_good

def test_file_contents():
    """测试关键文件内容"""
    print("\n=== 测试文件内容 ===")
    
    # 检查随机种子函数是否存在
    files_to_check = [
        'beyes/VGG2.py',
        'FeatureSelect/pcapVGGSeed.py',
        'Loss/pcapVGGSeed.py'
    ]
    
    all_good = True
    
    for file_path in files_to_check:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 检查是否包含随机种子设置函数
            if 'set_deterministic_seed' in content:
                print(f"✅ {file_path} 包含随机种子设置函数")
            else:
                print(f"❌ {file_path} 缺少随机种子设置函数")
                all_good = False
                
            # 检查是否包含TensorFlow 2.9.0相关设置
            if 'tf.keras.utils.set_random_seed' in content:
                print(f"✅ {file_path} 包含TensorFlow 2.9.0随机种子设置")
            else:
                print(f"❌ {file_path} 缺少TensorFlow 2.9.0随机种子设置")
                all_good = False
                
            # 检查是否移除了tf.compat.v1
            if 'tf.compat.v1.disable_eager_execution' in content:
                print(f"❌ {file_path} 仍包含tf.compat.v1.disable_eager_execution")
                all_good = False
            else:
                print(f"✅ {file_path} 已移除tf.compat.v1.disable_eager_execution")
                
        except Exception as e:
            print(f"❌ 读取文件 {file_path} 失败: {e}")
            all_good = False
    
    return all_good

def test_script_parameters():
    """测试脚本参数配置"""
    print("\n=== 测试脚本参数 ===")
    
    try:
        with open('run_vgg_training.bat', 'r', encoding='utf-8') as f:
            bat_content = f.read()
        
        with open('run_vgg_training.sh', 'r', encoding='utf-8') as f:
            sh_content = f.read()
        
        # 检查参数说明
        required_params = ['K', 'loss_type', 'ES_THRESHOLD', 'SEED', 'script_type']
        
        all_good = True
        
        for param in required_params:
            if param in bat_content and param in sh_content:
                print(f"✅ 参数 {param} 在脚本中有说明")
            else:
                print(f"❌ 参数 {param} 在脚本中缺少说明")
                all_good = False
        
        # 检查脚本类型
        script_types = ['feature', 'loss', 'beyes']
        for script_type in script_types:
            if script_type in bat_content and script_type in sh_content:
                print(f"✅ 脚本类型 {script_type} 支持")
            else:
                print(f"❌ 脚本类型 {script_type} 不支持")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"❌ 测试脚本参数失败: {e}")
        return False

def test_requirements():
    """测试requirements.txt"""
    print("\n=== 测试依赖包配置 ===")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        required_packages = [
            'tensorflow==2.9.0',
            'numpy',
            'scikit-learn',
            'matplotlib',
            'hyperopt',
            'pandas'
        ]
        
        all_good = True
        
        for package in required_packages:
            if package.split('>=')[0].split('==')[0] in requirements:
                print(f"✅ 依赖包 {package.split('>=')[0].split('==')[0]} 已配置")
            else:
                print(f"❌ 依赖包 {package.split('>=')[0].split('==')[0]} 未配置")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"❌ 测试requirements.txt失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始基础测试VGG29代码...")
    
    tests = [
        ("目录结构测试", test_directory_structure),
        ("文件内容测试", test_file_contents),
        ("脚本参数测试", test_script_parameters),
        ("依赖包配置测试", test_requirements)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 执行失败: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("📊 测试结果汇总")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有基础测试通过！VGG29代码结构正确！")
        print("\n📋 下一步:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 准备数据: 确保train_data.csv和test_data.csv在../../目录")
        print("3. 运行训练: ./run_vgg_training.sh 或 run_vgg_training.bat")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关问题。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
