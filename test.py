import torch
import gymnasium as gym
import highway_env
import numpy as np
from dqn_agent import DQNAgent   # 你的智能体类

# ================== 配置 ==================
env = gym.make("highway-v0", render_mode="human",
               config={
                   "duration": 40,          # ← 改这里，单位秒，例如 80 秒 → 约 80 steps（如果 policy_frequency=1）
        # 可选：如果你想保持步数不变但时间变长，可以同时调 frequency
        # "policy_frequency": 1,   # 默认 1 或 5，根据你的版本
        # "simulation_frequency": 15,  # 物理模拟频率，通常不用动
    })  # 或者 "rgb_array" 录视频

state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

# 加载训练好的模型（推荐用 best_xxx.pth）
checkpoint = torch.load("models/latest_dqn_highway.pth", weights_only=False)
agent.policy_net.load_state_dict(checkpoint)
# 什么意思？为什么要加载模型的状态字典？如果不加载会怎样？如果加载了但不调用eval()会怎样？
agent.policy_net.eval()          # 重要！进入评估模式

print("✅ 模型加载成功，开始测试...")

num_test_episodes = 10
rewards = []
success_count = 0

for episode in range(num_test_episodes):
    state, _ = env.reset()
    state = state.flatten()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 1000:
        # 测试时用确定性动作（不加噪声）
        with torch.no_grad():
            action = agent.select_action(state, epsilon=0.0)   # 强制 epsilon=0

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state.flatten()
        total_reward += reward
        steps += 1
        if truncated:
            success_count += 1

    rewards.append(total_reward)
    print(f"Test Episode {episode:2d} | Reward: {total_reward:6.2f} | Steps: {steps}")

print(f"\n平均测试奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"成功次数: {success_count}/{num_test_episodes}")
env.close()