from Chapter10.lib.environ import data, State

prices = data.Prices(open=[50, 55, 60, 58, 62],
                     high=[52, 58, 62, 60, 65],
                     low=[48, 52, 58, 56, 59],
                     close=[51, 57, 61, 59, 63],
                     volume=[10000, 12000, 11000, 9000, 15000])
state = State(bars_count=5, commission_perc=0.01, reset_on_close=True, reward_on_close=True, volumes=True)
state.reset(prices, offset=4)

encoded_state = state.encode()
print("Encoded state:\n", encoded_state)

# 绘制图表
import matplotlib.pyplot as plt

# 提取历史高、低、收盘价和成交量数据
high_data = encoded_state[0::4]
low_data = encoded_state[1::4]
close_data = encoded_state[2::4]
volume_data = encoded_state[3::4]

# 绘制图表
# plt.figure(figsize=(10, 6))
# plt.plot(high_data, label='High')
# plt.plot(low_data, label='Low')
# plt.plot(close_data, label='Close')
# plt.bar(range(len(volume_data)), volume_data, alpha=0.3, label='Volume')
# plt.xlabel('Time')
# plt.ylabel('Price/Volume')
# plt.title('Historical Price Data and Volume')
# plt.legend()
# plt.grid(True)
# plt.show()


# 创建一个新的图表
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制价格数据的折线图
ax1.plot(high_data, color='b', label='High')
ax1.plot(low_data, color='r', label='Low')
ax1.plot(close_data, color='g', label='Close')
ax1.set_ylabel('Price', color='b')

# 创建一个共享 x 轴的副坐标轴
ax2 = ax1.twinx()

# 缩放成交量数据到价格数据的范围内
scaled_volume_data = [(volume / max(volume_data)) * max(high_data) for volume in volume_data]
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
