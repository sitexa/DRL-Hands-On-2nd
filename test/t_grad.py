import torch

# 创建一个张量，并设置 requires_grad=True 以追踪其梯度
x = torch.tensor([2.0], requires_grad=True)

# 定义一个简单的函数：y = x^2
y = x**2

# 计算 y 对 x 的梯度
y.backward()

# 输出梯度
print(x.grad)
