from collections import deque
import random

# 经验存储：将上一组的state,action,reward,next_state,done标志存起来；

class ReplayBuffer:

    def __init__(self, capacity):
        # 双端队列，具有固定的最大长度。当达到容量时，最旧的元素会被自动删除。这种数据结构非常适合实现经验回放缓冲区，因为它允许我们高效地添加新经验并丢弃旧经验。
        # 初始化的时候不需要说明维度吗？因为我们存储的是元组(state, action, reward, next_state, done)，每个元素的维度可以不同，缓冲区只需要存储这些元组即可。
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # 为什么用元组？因为我们需要同时存储状态、动作、奖励、下一个状态和完成标志，这些信息是相关联的。使用元组可以将这些相关的信息打包在一起，方便存储和管理。就因为相关？
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # random是个Python内置模块，提供了生成随机数和进行随机选择的函数。这里的random.sample()函数用于从self.buffer中随机抽取batch_size个元素，返回一个新的列表。这些元素是之前存储在缓冲区中的经验（状态、动作、奖励、下一个状态和完成标志）。通过随机抽样，我们可以打破数据之间的相关性，提高训练的稳定性和效率。
        batch = random.sample(self.buffer, batch_size)
        # zip(*batch)的作用是将批次中的每个元素（state, action, reward, next_state, done）分别组合成一个新的元组。具体来说，假设batch中有N个元素，每个元素都是一个包含5个部分的元组，那么zip(*batch)会将所有的state组合成一个元组，所有的action组合成一个元组，以此类推。最终，我们得到5个元组，分别对应状态、动作、奖励、下一个状态和完成标志。
        # zip的数学解释是？
        states, actions, rewards, next_states, dones = zip(*batch)
        # 这里就是5个元组
        return states, actions, rewards, next_states, dones

    def __len__(self):

        return len(self.buffer)