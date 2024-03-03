import torch
import torch.nn as nn

# 定义输入数据，假设为一张灰度图像，大小为(1, 1, 5, 5)
input_data = torch.tensor([[[[1, 2, 3, 4, 5],
                             [6, 7, 8, 9, 10],
                             [11, 12, 13, 14, 15],
                             [16, 17, 18, 19, 20],
                             [21, 22, 23, 24, 25]]]], dtype=torch.float32)

# 定义卷积层，输入通道数为1，输出通道数为1，卷积核大小为3x3
conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

# 打印卷积层的权重（卷积核参数）和偏置
print("卷积层权重：", conv_layer.weight)
print("卷积层偏置：", conv_layer.bias)

# 将输入数据传递给卷积层，进行卷积操作
output_data = conv_layer(input_data)

# 打印输入和输出的形状
print("输入数据形状：", input_data.shape)
print("输出数据形状：", output_data.shape)

# 打印输入和输出的数据
print("输入数据：", input_data)
print("输出数据：", output_data)