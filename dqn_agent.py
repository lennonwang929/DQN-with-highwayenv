from QNetwork import QNetwork
from replay_buffer import ReplayBuffer
import torch
import torch.optim as optim
# import torch.nn as nn ## nn是PyTorch中用于构建神经网络的模块，提供了各种层、损失函数和优化器等工具。
import torch.nn.functional as F## F是PyTorch中提供的函数式接口，包含了各种激活函数、损失函数等操作，可以直接调用而不需要定义为层。

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        # 初始化时复制策略网络的权重到目标网络，确保它们在开始时是相同的。这是DQN算法中的一个重要步骤，因为目标网络用于计算目标Q值，保持其稳定性对于训练过程非常重要。
        self.target_net.load_state_dict(self.policy_net.state_dict())#需要解释下语法
        self.replay_buffer = ReplayBuffer(10000)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)#需要解释，还有从哪里能看到网络有哪些元素可以或者需要供我调用？比如这里的parameters是什么？还有其他函数吗？
        self.gamma = 0.99#折扣因子，决定了未来奖励在当前决策中的重要性。值越接近1，未来奖励越重要；值越接近0，当前奖励越重要。
        self.epsilon = 1.0#greedy策略中的探索率，决定了智能体在选择动作时是更多地探索（选择随机动作）还是利用（选择当前估计的最佳动作）。初始值为1.0，表示完全探索。
        self.epsilon_decay = 0.995#每次更新epsilon时的衰减率，表示epsilon在每次更新后乘以这个值，从而逐渐减少探索率。值越接近1，衰减越慢；值越小，衰减越快。
        self.epsilon_min = 0.01#epsilon的最小值，确保智能体在训练过程中始终保留一定的探索能力。即使经过多次更新，epsilon也不会低于这个值。
        self.batch_size = 64
        

        
        self.train_step_count = 0
        self.target_update_freq = 1000

    def select_action(self, state, epsilon=None):
        #要生成一个随机数，决定要随机探索还是利用当前策略，随机探索直接随机选择一个动作，利用当前策略要调用当前策略网络计算Q值，选择Q值最大的那个动作
        if epsilon is not None:
            self.epsilon = epsilon
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()#随机选择一个动作，torch.randint(0, self.action_dim, (1,))生成一个在0到action_dim-1之间的随机整数，表示动作的索引。item()方法将这个单元素张量转换为一个Python整数。
        else:
            # 为什么需要专门写不需要计算梯度，所以默认是会计算的？如果没有这行代码会怎样？
            with torch.no_grad():#上下文管理器，表示在这个代码块中不需要计算梯度，这样可以节省内存和计算资源，因为在选择动作时我们不需要进行反向传播。
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)#将输入状态转换为PyTorch张量，并添加一个批次维度，使其形状变为(1, state_dim)，以适应神经网络的输入要求。
                q_values = self.policy_net(state)#通过策略网络计算当前状态下每个动作的Q值。q_values是一个包含所有动作Q值的张量。
                return q_values.argmax().item()#选择Q值最大的动作，argmax()返回最大值的索引，item()将其转换为Python整数，作为最终选择的动作。
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    # 需要有一个函数算下一步的状态，奖励还有是否done吧

    def train_step(self):
        # 先判断buffer里是否存够了一个batch，否则就不训练
        if len(self.replay_buffer) < self.batch_size:
            return
        # 从buffer里抽取batchsize个样本
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        # 将抽取的样本转换为PyTorch张量，并进行必要的维度调整，以便输入到神经网络中进行训练。
        # 具体来说，状态和下一个状态被转换为浮点张量，动作被转换为长整型张量并添加一个维度，奖励和完成标志也被转换为浮点张量并添加一个维度。这些张量将用于计算当前Q值、下一步Q值和目标Q值，从而进行损失计算和网络参数更新。
        # 这里需要举例说明一下，不够直观，
        # 计算当前Q值、下一步Q值和目标Q值。当前Q值是通过策略网络计算得到的，使用gather方法根据动作索引选择对应的Q值。一个Q值列表
        current_q_values = self.policy_net(states).gather(1, actions)
        # 下一步Q值是通过目标网络计算得到的，使用max方法选择每个状态下的最大Q值，并使用unsqueeze方法调整维度。为什么下一步Q值要用目标网络？
        with torch.no_grad():
           next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        # 目标Q值是根据奖励和折扣因子计算得到的，同时考虑了完成标志，以确保在终止状态时不考虑未来奖励。
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        # 计算损失，这里使用均方误差损失函数（MSELoss）来衡量当前Q值和目标Q值之间的差距。这个损失将用于反向传播，以更新策略网络的参数。
        loss = F.mse_loss(current_q_values, target_q_values)
        # 这一步是？在进行反向传播之前，我们需要先将优化器的梯度缓存清零，以避免累积之前的梯度。
        self.optimizer.zero_grad()
        # 调用loss.backward()计算当前损失相对于网络参数的梯度，
        loss.backward()
        # 最后调用optimizer.step()更新网络参数，使其朝着最小化损失的方向调整。
        self.optimizer.step()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # 为什么是.item()，loss是一个张量，为什么是一个张量？
        return loss.item()  # 返回当前训练步骤的损失值，供外部调用者使用（例如记录到TensorBoard）。如果不需要返回损失，可以将这行代码注释掉。

    # def update_target(self):
    #     self.target_net.load_state_dict(self.policy_net.state_dict())