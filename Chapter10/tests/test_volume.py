import matplotlib.pyplot as plt

# 假设价格范围是 [0, 100]，成交量范围是 [0, 20000]
price_data = [50, 55, 60, 58, 62]
volume_data = [10000, 12000, 11000, 9000, 15000]

# 创建一个新的图表
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制价格数据的折线图
ax1.plot(price_data, color='b', label='Price')
ax1.set_ylabel('Price', color='b')

# 创建一个共享 x 轴的副坐标轴
ax2 = ax1.twinx()

# 缩放成交量数据到价格数据的范围内
scaled_volume_data = [(volume / max(volume_data)) * max(price_data) for volume in volume_data]

# 绘制缩放后的成交量数据的柱状图
ax2.bar(range(len(volume_data)), scaled_volume_data, alpha=0.3, color='r', label='Volume')
ax2.set_ylabel('Scaled Volume', color='r')

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.xlabel('Time')
plt.title('Price and Scaled Volume')
plt.grid(True)
plt.show()
