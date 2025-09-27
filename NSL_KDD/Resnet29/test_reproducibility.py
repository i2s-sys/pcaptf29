# 测试随机种子设置是否正确
import tensorflow as tf
import numpy as np
import random

def set_deterministic_seed(seed):
    """设置所有随机种子确保结果可复现"""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_reproducibility():
    """测试随机种子设置是否有效"""
    print("=== 测试随机种子设置 ===")
    
    # 测试1: TensorFlow随机数
    print("\n1. 测试TensorFlow随机数:")
    set_deterministic_seed(42)
    tf_rand1 = tf.random.normal([3, 3])
    set_deterministic_seed(42)
    tf_rand2 = tf.random.normal([3, 3])
    print(f"TensorFlow随机数是否相同: {tf.reduce_all(tf_rand1 == tf_rand2).numpy()}")
    
    # 测试2: NumPy随机数
    print("\n2. 测试NumPy随机数:")
    set_deterministic_seed(42)
    np_rand1 = np.random.rand(3, 3)
    set_deterministic_seed(42)
    np_rand2 = np.random.rand(3, 3)
    print(f"NumPy随机数是否相同: {np.allclose(np_rand1, np_rand2)}")
    
    # 测试3: Python random
    print("\n3. 测试Python random:")
    set_deterministic_seed(42)
    py_rand1 = [random.random() for _ in range(5)]
    set_deterministic_seed(42)
    py_rand2 = [random.random() for _ in range(5)]
    print(f"Python random是否相同: {py_rand1 == py_rand2}")
    
    # 测试4: 模型权重初始化
    print("\n4. 测试模型权重初始化:")
    set_deterministic_seed(42)
    model1 = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    weights1 = model1.get_weights()
    
    set_deterministic_seed(42)
    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    weights2 = model2.get_weights()
    
    weights_same = all(np.allclose(w1, w2) for w1, w2 in zip(weights1, weights2))
    print(f"模型权重是否相同: {weights_same}")
    
    print("\n=== 测试完成 ===")
    return tf.reduce_all(tf_rand1 == tf_rand2).numpy() and np.allclose(np_rand1, np_rand2) and py_rand1 == py_rand2 and weights_same

if __name__ == "__main__":
    success = test_reproducibility()
    if success:
        print("\n✅ 所有随机种子设置正确，结果可以复现！")
    else:
        print("\n❌ 随机种子设置有问题，结果可能无法复现！")
