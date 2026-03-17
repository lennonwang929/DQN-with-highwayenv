import torch
import gymnasium as gym
import highway_env
import numpy as np
from dqn_agent import DQNAgent
from torch.utils.tensorboard import SummaryWriter  # ← 新增
import matplotlib.pyplot as plt
import os

# ================== 配置 ==================
env = gym.make("highway-v0")  # 训练时不渲染，速度快；想看效果时改成 "human" 或 "rgb_array"
# env = gym.make("highway-v0", render_mode="rgb_array")  # 可选：用于偶尔渲染视频

state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

num_episodes = 500
max_steps = 1000

# TensorBoard 日志
writer = SummaryWriter(log_dir="runs/dqn_highway")  # 日志存到 runs/ 文件夹

# 用于计算 moving average
rewards_history = []
moving_avg_window = 100

best_avg_reward = -float('inf')          
best_model_path = "models/best_dqn_highway.pth"

os.makedirs("models", exist_ok=True)
# 接着训练：加载之前的模型权重，继续训练（如果之前有保存的话）
# 加载模型权重，如果之前没有保存的模型，这里会报错，你可以先运行一次训练，保存一个模型后再继续训练，或者在这里加个判断，如果文件存在才加载。
if os.path.exists("models/latest_dqn_highway.pth"):
    state_dict = torch.load("models/latest_dqn_highway.pth", map_location='cpu')
    agent.policy_net.load_state_dict(state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent.policy_net.to(device)

if hasattr(agent, 'target_net'):
    agent.target_net.load_state_dict(state_dict)
    agent.target_net.to(device)

# 手动设置 epsilon（避免从头开始太高的探索）
agent.epsilon = 0.01  # 根据你之前曲线最后的值调整，比如 0.01~0.1

# 如果你有 epsilon decay 逻辑，确保当前 step 对应这个 epsilon
# agent.epsilon_decay_step = some_value  # 如果有的话

# 直接继续从 0 开始循环，或者你自己记录了上次的 episode 数
start_episode = 1000   # 或你知道的上次结束的 episode + 1

for episode in range(start_episode, num_episodes + start_episode):
    state, _ = env.reset()
    state = state.flatten()

    done = False
    total_reward = 0
    episode_loss = 0.0
    steps = 0

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = next_state.flatten()

        agent.store_transition(state, action, reward, next_state, done)
        
        loss = agent.train_step()  # ← 假设你的 train_step 返回 loss (float or tensor)，如果不返回就注释下面两行
        if loss is not None:
            episode_loss += loss
            steps += 1

        state = next_state
        total_reward += reward

        if done:
            break

    rewards_history.append(total_reward)
    #这的意思是从当前往前数100个episode的reward的平均值，如果当前episode数小于100，就从第0个开始算到当前episode的reward平均值 
    avg_reward = np.mean(rewards_history[-moving_avg_window:])

    # ================== Logging to TensorBoard ==================
    writer.add_scalar("Reward/Episode", total_reward, episode)
    writer.add_scalar("Reward/MovingAvg100", avg_reward, episode)
    writer.add_scalar("Epsilon", agent.epsilon, episode)
    
    if steps > 0 and loss is not None:
        writer.add_scalar("Loss/AvgPerEpisode", episode_loss / steps, episode)
    
    # 可选：记录其他指标（如Q值、梯度等），在 DQNAgent 里加 writer 即可

    if episode % 100 == 0:
        model_path = f"models/dqn_highway_ep{episode}.pth"
        torch.save(agent.policy_net.state_dict(), model_path)
        torch.save(agent.policy_net.state_dict(), "models/latest_dqn_highway.pth")
        print(f"✓ Saved model to {model_path}")

    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        torch.save(agent.policy_net.state_dict(), best_model_path)
        print(f"✓ New best model saved with avg reward {best_avg_reward:.2f} at episode {episode}")    

    print(f"Episode {episode:4d} | Reward {total_reward:6.2f} | "
          f"MovingAvg {avg_reward:6.2f} | Epsilon {agent.epsilon:.3f}")
    

# 训练结束：关闭writer，画静态图
writer.close()

# ================== 训练结束后画 matplotlib 图 ==================
plt.figure(figsize=(10, 5))
plt.plot(rewards_history, label="Episode Reward", alpha=0.6)
plt.plot(np.convolve(rewards_history, np.ones(moving_avg_window)/moving_avg_window, mode='valid'),
         label=f"Moving Avg ({moving_avg_window})", color="orange", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Training on highway-v0")
plt.legend()
plt.grid(True)
plt.savefig("dqn_highway_rewards.png")
plt.show()

print("训练完成！TensorBoard 日志在 runs/dqn_highway 文件夹")