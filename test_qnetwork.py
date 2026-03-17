import torch
from QNetwork import QNetwork

state_dim = 25
action_dim = 5

net = QNetwork(state_dim, action_dim)

# 随机生成一个state
state = torch.randn(1, state_dim)

q_values = net(state)
print(q_values.shape)
print("Input shape:", state.shape)
print("Output shape:", q_values.shape)
print("Q values:", q_values)