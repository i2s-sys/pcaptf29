import tensorflow as tf
import heapq
import random
tf.compat.v1.disable_eager_execution()  # 禁用 eager execution


# 假设 self.scaling_factor 是一个形状为 [1, 72] 的张量
# scaling_factor = tf.Variable(tf.constant(1, dtype=tf.float32, shape=[1, 72]))
scaling_factor = tf.Variable(tf.random.normal([1, 72], mean=25.0, stddev=10.0, dtype=tf.float32))
# 将张量转换为 NumPy 数组
scaling_factor_array = scaling_factor.numpy().flatten()

# print(scaling_factor_array)
# 获取前 k 个最大值及其索引
k = 8
def top_k_with_indices(arr, k):
    heap = []
    for index, value in enumerate(arr):
        if len(heap) < k:
            heapq.heappush(heap, (value, index))
        else:
            if value > heap[0][0]:
                heapq.heappop(heap)
                heapq.heappush(heap, (value, index))
    top_k_sorted = sorted(heap, key=lambda x: x[0], reverse=True)
    top_values = [item[0] for item in top_k_sorted]
    top_indices = [item[1] for item in top_k_sorted]
    return top_values, top_indices
top_values, top_indices = top_k_with_indices(scaling_factor_array, k)

# 分别提取数值和下标

print(top_values)
print(top_indices)