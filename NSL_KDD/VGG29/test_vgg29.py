# 测试VGG29代码的基本功能
import sys
import os
import numpy as np

def test_imports():
    """测试所有必要的导入"""
    print("=== 测试导入 ===")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow版本: {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy版本: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy导入失败: {e}")
        return False
    
    try:
        from sklearn.metrics import f1_score
        print("✅ Scikit-learn导入成功")
    except ImportError as e:
        print(f"❌ Scikit-learn导入失败: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib导入成功")
    except ImportError as e:
        print(f"❌ Matplotlib导入失败: {e}")
        return False
    
    return True

def test_random_seed():
    """测试随机种子设置"""
    print("\n=== 测试随机种子设置 ===")
    
    try:
        import tensorflow as tf
        import numpy as np
        import random
        
        def set_deterministic_seed(seed):
            tf.keras.utils.set_random_seed(seed)
            tf.config.experimental.enable_op_determinism()
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # 测试随机种子
        set_deterministic_seed(42)
        rand1 = np.random.rand(3)
        
        set_deterministic_seed(42)
        rand2 = np.random.rand(3)
        
        if np.allclose(rand1, rand2):
            print("✅ 随机种子设置正确")
            return True
        else:
            print("❌ 随机种子设置失败")
            return False
            
    except Exception as e:
        print(f"❌ 随机种子测试失败: {e}")
        return False

def test_vgg_model_creation():
    """测试VGG模型创建"""
    print("\n=== 测试VGG模型创建 ===")
    
    try:
        # 尝试导入VGG模型
        sys.path.append('./FeatureSelect')
        from pcapVGGSeed import VGG, set_deterministic_seed
        
        print("✅ VGG模型导入成功")
        
        # 设置随机种子
        set_deterministic_seed(25)
        
        # 创建模拟数据来测试模型创建
        print("创建测试数据...")
        
        # 创建临时CSV文件用于测试
        import csv
        
        # 创建训练数据
        train_data = []
        for i in range(100):
            row = [np.random.rand() for _ in range(41)] + [np.random.randint(0, 2)]
            train_data.append(row)
        
        with open('../../train_data_test.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(train_data)
        
        # 创建测试数据
        test_data = []
        for i in range(50):
            row = [np.random.rand() for _ in range(41)] + [np.random.randint(0, 2)]
            test_data.append(row)
        
        with open('../../test_data_test.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(test_data)
        
        print("✅ 测试数据创建成功")
        
        # 暂时修改文件路径
        import pcapVGGSeed
        original_train_file = pcapVGGSeed.TRAIN_FILE
        original_test_file = pcapVGGSeed.TEST_FILE
        pcapVGGSeed.TRAIN_FILE = '../../train_data_test.csv'
        pcapVGGSeed.TEST_FILE = '../../test_data_test.csv'
        
        try:
            # 创建VGG模型
            print("创建VGG模型...")
            model = VGG(K=1, lossType="ce", ES_THRESHOLD=3, seed=25)
            print("✅ VGG模型创建成功")
            
            # 测试模型调用
            test_input = np.random.rand(2, 41).astype(np.float32)
            output = model(test_input, training=False)
            print(f"✅ 模型前向传播成功，输出形状: {output.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ VGG模型创建失败: {e}")
            return False
        finally:
            # 恢复原始文件路径
            pcapVGGSeed.TRAIN_FILE = original_train_file
            pcapVGGSeed.TEST_FILE = original_test_file
            
            # 清理测试文件
            try:
                os.remove('../../train_data_test.csv')
                os.remove('../../test_data_test.csv')
            except:
                pass
        
    except ImportError as e:
        print(f"❌ VGG模型导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ VGG模型测试失败: {e}")
        return False

def test_shell_scripts():
    """测试shell脚本是否存在"""
    print("\n=== 测试Shell脚本 ===")
    
    scripts = [
        'run_vgg_training.sh',
        'run_vgg_training.bat'
    ]
    
    all_exist = True
    for script in scripts:
        if os.path.exists(script):
            print(f"✅ {script} 存在")
        else:
            print(f"❌ {script} 不存在")
            all_exist = False
    
    return all_exist

def test_directory_structure():
    """测试目录结构"""
    print("\n=== 测试目录结构 ===")
    
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
        'README_CN.md'
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

def main():
    """主测试函数"""
    print("🚀 开始测试VGG29代码...")
    
    tests = [
        ("导入测试", test_imports),
        ("随机种子测试", test_random_seed),
        ("目录结构测试", test_directory_structure),
        ("Shell脚本测试", test_shell_scripts),
        ("VGG模型创建测试", test_vgg_model_creation)
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
        print("🎉 所有测试通过！VGG29代码迁移成功！")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关问题。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
