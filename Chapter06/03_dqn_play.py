#!/usr/bin/env python3
import argparse
import collections
import time

import gymnasium as gym
import numpy as np
import torch
from lib import dqn_model, wrappers

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model file to load")
    parser.add_argument(
        "-e", "--env", default=DEFAULT_ENV_NAME, help="Environment name to use, default=" + DEFAULT_ENV_NAME
    )
    parser.add_argument("-r", "--record", help="Directory for video")
    parser.add_argument("--no-vis", default=True, dest="vis", help="Disable visualization", action="store_false")
    args = parser.parse_args()

    env = wrappers.make_env(args.env)
    env.unwrapped.render_mode = "rgb_array"
    env.unwrapped.metadata["render_fps"] = FPS
    if args.record:
        env = gym.wrappers.RecordVideo(env, args.record)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    state = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(state)

    state, _ = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    for i in range(500):
        start_ts = time.time()
        if args.vis:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False)).to(device)
        q_vals = net(state_v).data.cpu().numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, terminated, truncated, _ = env.step(action)
        print(f"[{i}] reward: {reward}")
        total_reward += reward
        if terminated or truncated:
            break
        if args.vis:
            delta = 1 / FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.close()
