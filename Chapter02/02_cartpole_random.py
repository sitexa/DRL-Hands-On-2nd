import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    total_reward = 0.0
    total_steps = 0
    obs, info = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
