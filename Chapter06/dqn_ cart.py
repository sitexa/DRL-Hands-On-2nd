import random
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

DEFAULT_ENV_NAME = "CartPole-v1"

GAMMA = 0.99
BATCH_SIZE = 128
BUFFER_SIZE = 10000

LEARNING_RATE = 1e-4
EPS_START = 0.9
EPS_END = 0.05
EPS_STEPS = 1000

TAU = 0.005

device = torch.device("mps")

Transition = namedtuple(typename="Transition", field_names=("state", "action", "reward", "next_state", "terminated"))


class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminations = map(np.array, zip(*transitions))
        return states, actions, rewards, next_states, terminations


class Qnet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Qnet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class DQN:
    def __init__(self, env: gym.Env, buffer: ReplayBuffer):
        self.env = env
        self.buffer = buffer
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.q_net = Qnet(self.state_dim, self.action_dim).to(device)
        self.target_net = Qnet(self.state_dim, self.action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.count = 0

    @staticmethod
    def exponential_schedule(start, end, decay):
        return end + (start - end) * np.exp(-1.0 * decay)

    @torch.no_grad()
    def get_action(self, state):
        eps_threshold = self.exponential_schedule(EPS_START, EPS_END, self.count / EPS_STEPS)
        self.count += 1
        if np.random.random() < eps_threshold:
            return env.action_space.sample()
        else:
            state_tensor = torch.tensor(np.array([state]), device=device, dtype=torch.float32)
            return self.q_net(state_tensor).argmax(dim=1).item()

    def train_episode(self):
        state, _ = self.env.reset()
        total_reward = 0.0
        while True:
            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            transition = Transition(state, action, reward, next_state, terminated)
            self.buffer.add(transition)
            state = next_state
            if terminated or truncated:
                break

            if len(self.buffer) < BATCH_SIZE:
                continue
            self.optimize_model()
            self.soft_update()
        return total_reward

    def optimize_model(self):
        states, actions, rewards, next_states, terminations = self.buffer.sample(BATCH_SIZE)
        states = torch.tensor(states, device=device, dtype=torch.float32)
        actions = torch.tensor(actions, device=device, dtype=torch.long).view(-1, 1)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float32).view(-1, 1)
        next_states = torch.tensor(next_states, device=device, dtype=torch.float32)
        terminations = torch.tensor(terminations, device=device, dtype=torch.bool).view(-1, 1)
        # Q(s_t)
        q_values = self.q_net(states).gather(dim=1, index=actions)
        with torch.no_grad():
            # Q'(s_{t+1})
            next_q_values = self.target_net(next_states).max(dim=1)[0].view(-1, 1)
            next_q_values[terminations] = 0.0
            next_q_values = next_q_values.detach()
        expected_q_values = rewards + GAMMA * next_q_values

        # loss = F.mse_loss(q_values, expected_q_values).mean()
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()

    def soft_update(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


if __name__ == "__main__":
    num_episodes = 600
    env = gym.make(DEFAULT_ENV_NAME)

    buffer = ReplayBuffer()
    dqn = DQN(env, buffer)
    writer = SummaryWriter(comment="-cartpole")
    for i in tqdm(range(num_episodes)):
        reward = dqn.train_episode()
        writer.add_scalar("reward", reward, i)
    writer.close()
