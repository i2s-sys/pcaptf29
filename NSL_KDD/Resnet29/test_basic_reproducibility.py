# 简化的随机种子测试脚本（不依赖TensorFlow）
import numpy as np
import random

def test_basic_reproducibility():
    """测试基本的随机种子设置"""
    print("=== 测试基本随机种子设置 ===")
    
    # 测试NumPy随机数
    print("\n1. 测试NumPy随机数:")
    np.random.seed(42)
    np_rand1 = np.random.rand(3, 3)
    np.random.seed(42)
    np_rand2 = np.random.rand(3, 3)
    print(f"NumPy随机数是否相同: {np.allclose(np_rand1, np_rand2)}")
    print(f"第一次结果:\n{np_rand1}")
    print(f"第二次结果:\n{np_rand2}")
    
    # 测试Python random
    print("\n2. 测试Python random:")
    random.seed(42)
    py_rand1 = [random.random() for _ in range(5)]
    random.seed(42)
    py_rand2 = [random.random() for _ in range(5)]
    print(f"Python random是否相同: {py_rand1 == py_rand2}")
    print(f"第一次结果: {py_rand1}")
    print(f"第二次结果: {py_rand2}")
    
    print("\n=== 基本测试完成 ===")
    return np.allclose(np_rand1, np_rand2) and py_rand1 == py_rand2

if __name__ == "__main__":
    success = test_basic_reproducibility()
    if success:
        print("\n✅ 基本随机种子设置正确！")
        print("💡 提示：安装TensorFlow后运行完整测试脚本")
    else:
        print("\n❌ 基本随机种子设置有问题！")
