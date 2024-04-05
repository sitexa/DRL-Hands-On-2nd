import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # 定义权重和偏置
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # 定义噪声参数
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features) * sigma_init)
        self.sigma_bias = nn.Parameter(torch.Tensor(out_features) * sigma_init)

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化权重和偏置
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1 / math.sqrt(self.in_features), 1 / math.sqrt(self.in_features))

    def forward(self, input):
        # 生成噪声
        epsilon_weight = torch.normal(0, self.sigma_weight.size()).to(self.weight.device)
        epsilon_bias = torch.normal(0, self.sigma_bias.size()).to(self.bias.device)

        # 应用带噪声的权重和偏置
        return F.linear(input, self.weight + self.sigma_weight * epsilon_weight,
                        self.bias + self.sigma_bias * epsilon_bias)


# 测试NoisyLinear层
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.noisy_layer = NoisyLinear(10, 2)

    def forward(self, x):
        return self.noisy_layer(x)


# 创建模型和数据示例
model = TestNet()
x = torch.rand(5, 10)  # 生成一个5x10的随机输入张量

output = model(x)
print(output)
