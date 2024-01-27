import torch
import torch.nn as nn

import ptan


class DQNNet(nn.Module):
    def __init__(self, n_actions: int):
        super(DQNNet, self).__init__()
        self.n_actions = n_actions

    def forward(self, x: torch.Tensor):
        # we always produce diagonal tensor of shape (batch_size, n_actions)
        return torch.eye(x.shape[0], self.n_actions)


class PolicyNet(nn.Module):
    def __init__(self, n_actions: int):
        super(PolicyNet, self).__init__()
        self.n_actions = n_actions

    def forward(self, x):
        # Now we produce the tensor with first two actions
        # having the same logit scores
        shape = (x.shape[0], self.n_actions)
        res = torch.zeros(shape, dtype=torch.float32)
        res[:, 0] = 1
        res[:, 1] = 1
        return res


if __name__ == "__main__":
    net = DQNNet(n_actions=3)
    net_out = net(torch.zeros(2, 10))
    print("dqn_net:")
    print(net_out)

    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(dqn_model=net, action_selector=selector)
    ag_out = agent(torch.zeros(2, 5))
    print("Argmax:", ag_out)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=1.0)
    agent = ptan.agent.DQNAgent(dqn_model=net, action_selector=selector)
    ag_out = agent(torch.zeros(10, 5))[0]
    print("eps=1.0:", ag_out)

    selector.epsilon = 0.5
    ag_out = agent(torch.zeros(10, 5))[0]
    print("eps=0.5:", ag_out)

    selector.epsilon = 0.1
    ag_out = agent(torch.zeros(10, 5))[0]
    print("eps=0.1:", ag_out)

    net = PolicyNet(n_actions=5)
    net_out = net(torch.zeros(6, 10))
    print("policy_net:")
    print(net_out)

    selector = ptan.actions.ProbabilityActionSelector()
    agent = ptan.agent.PolicyAgent(model=net, action_selector=selector, apply_softmax=True)
    ag_out = agent(torch.zeros(6, 5))[0]
    print(ag_out)
