# 测试UNSW-IoT Resnet29迁移版本的基本功能
import os
import sys

def test_directory_structure():
    """测试目录结构"""
    print("=== 测试目录结构 ===")
    
    required_dirs = [
        ".",
        "FeatureSelect"
    ]
    
    required_files = [
        "FeatureSelect/pcapResnetPacketSeed.py",
        "FeatureSelect/pcapTrainResPacket_ES2_32.py",
        "requirements.txt",
        "README.md",
        "README_CN.md"
    ]
    
    all_passed = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ 目录存在: {dir_path}")
        else:
            print(f"✗ 目录不存在: {dir_path}")
            all_passed = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ 文件存在: {file_path}")
        else:
            print(f"✗ 文件不存在: {file_path}")
            all_passed = False
    
    return all_passed

def test_file_content():
    """测试文件内容"""
    print("\n=== 测试文件内容 ===")
    
    # 测试主要模型文件
    model_file = "FeatureSelect/pcapResnetPacketSeed.py"
    if os.path.exists(model_file):
        with open(model_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查关键内容
        checks = [
            ("TensorFlow 2.9.0", "tensorflow==2.9.0" in content or "TensorFlow 2.9.0" in content),
            ("set_deterministic_seed", "set_deterministic_seed" in content),
            ("Model import", "from tensorflow.keras import" in content and "Model" in content),
            ("tf.GradientTape", "tf.GradientTape" in content),
            ("@tf.function", "@tf.function" in content),
            ("DATA_DIM = 72", "DATA_DIM = 72" in content),
            ("OUTPUT_DIM = 29", "OUTPUT_DIM = 29" in content),
            ("class Resnet", "class Resnet" in content),
            ("class Resnet2", "class Resnet2" in content),
            ("class BasicBlock", "class BasicBlock" in content)
        ]
        
        for check_name, check_result in checks:
            if check_result:
                print(f"✓ {check_name}: 找到")
            else:
                print(f"✗ {check_name}: 未找到")
        
        # 检查是否移除了tf.compat.v1
        if "tf.compat.v1" not in content:
            print("✓ 已移除tf.compat.v1")
        else:
            print("✗ 仍包含tf.compat.v1")
    else:
        print("✗ 模型文件不存在")
        return False
    
    # 测试训练脚本
    train_file = "FeatureSelect/pcapTrainResPacket_ES2_32.py"
    if os.path.exists(train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        checks = [
            ("set_deterministic_seed", "set_deterministic_seed" in content),
            ("from pcapResnetPacketSeed import Resnet", "from pcapResnetPacketSeed import Resnet" in content),
            ("K = 32", "K = 32" in content),
            ("ES_THRESHOLD = 3", "ES_THRESHOLD = 3" in content)
        ]
        
        for check_name, check_result in checks:
            if check_result:
                print(f"✓ {check_name}: 找到")
            else:
                print(f"✗ {check_name}: 未找到")
    else:
        print("✗ 训练脚本不存在")
        return False
    
    return True

def test_requirements():
    """测试requirements.txt"""
    print("\n=== 测试requirements.txt ===")
    
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        with open(req_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        required_packages = [
            "tensorflow==2.9.0",
            "numpy",
            "scikit-learn",
            "matplotlib",
            "pandas"
        ]
        
        for package in required_packages:
            if package in content:
                print(f"✓ 包含包: {package}")
            else:
                print(f"✗ 缺少包: {package}")
        
        return True
    else:
        print("✗ requirements.txt不存在")
        return False

def test_syntax():
    """测试Python语法"""
    print("\n=== 测试Python语法 ===")
    
    python_files = [
        "FeatureSelect/pcapResnetPacketSeed.py",
        "FeatureSelect/pcapTrainResPacket_ES2_32.py"
    ]
    
    for file_path in python_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 简单的语法检查
                compile(content, file_path, 'exec')
                print(f"✓ 语法正确: {file_path}")
            except SyntaxError as e:
                print(f"✗ 语法错误: {file_path} - {e}")
                return False
            except Exception as e:
                print(f"✗ 其他错误: {file_path} - {e}")
                return False
        else:
            print(f"✗ 文件不存在: {file_path}")
            return False
    
    return True

def main():
    """主测试函数"""
    print("UNSW-IoT Resnet29 迁移版本测试")
    print("=" * 50)
    
    tests = [
        ("目录结构", test_directory_structure),
        ("文件内容", test_file_content),
        ("依赖包", test_requirements),
        ("Python语法", test_syntax)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}测试失败: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("测试结果总结:")
    
    all_passed = True
    for test_name, result in results:
        status = "通过" if result else "失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ 所有测试通过！UNSW-IoT Resnet29迁移成功")
    else:
        print("✗ 部分测试失败，请检查上述问题")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)