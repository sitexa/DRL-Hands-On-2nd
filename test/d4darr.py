import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个四维数组示例：4x4x4x4
data = np.random.rand(4, 4, 4, 4)

# 选择一个固定的维度（比如，最后一个维度的第一个元素）
fixed_dimension = data[:, :, :, 0]

# 创建一个3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 为了可视化，我们将使用前三个维度的数据
x, y, z = np.indices(fixed_dimension.shape)

# 绘制散点图
ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=fixed_dimension.ravel())

plt.show()
