import torch

# 假设状态张量有两个样本，每个样本有三个特征
states_v = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0]])

# 假设动作张量包含两个样本的动作索引
actions_v = torch.tensor([0, 2])

# 假设我们有一个神经网络模型
# 注意：这只是一个示例，实际网络可能更为复杂
net = torch.nn.Linear(3, 3)

# 使用神经网络计算 Q 值
state_action_values = net(states_v)

# 通过 gather 操作，根据动作索引提取对应的 Q 值
# 这里的 gather 操作相当于对每个样本提取对应动作的 Q 值
state_action_values = state_action_values.gather(1, actions_v.unsqueeze(-1))

# 最后，去除多余的维度，得到最终的 Q 值
state_action_values = state_action_values.squeeze(-1)

print(state_action_values)
