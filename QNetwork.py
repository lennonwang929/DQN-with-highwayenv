import torch
import torch.nn as nn ## nn是PyTorch中用于构建神经网络的模块，提供了各种层、损失函数和优化器等工具。
import torch.nn.functional as F## F是PyTorch中提供的函数式接口，包含了各种激活函数、损失函数等操作，可以直接调用而不需要定义为层。

# 这个语法是定义了一个名为QNetwork的类，继承自nn.Module，这是PyTorch中所有神经网络模块的基类。通过继承nn.Module，我们可以利用PyTorch提供的各种功能来构建和训练神经网络。
# 神经网络都是要继承nn.Module的，这样才能使用PyTorch的功能，比如自动求导、参数管理等。通过继承nn.Module，我们可以定义自己的神经网络结构，并且在训练过程中利用PyTorch的工具来优化网络参数。
class QNetwork(nn.Module):
# 在类的初始化方法中，我们定义了神经网络的结构。这个网络有三个全连接层（fc1、fc2、fc3）。输入层的大小由state_dim决定，输出层的大小由action_dim决定，中间有两个隐藏层，每个隐藏层有128个神经元。
    def __init__(self, state_dim, action_dim):

        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
# 在前向传播方法中，我们定义了数据如何通过网络进行计算。输入state首先通过第一个全连接层（fc1），然后经过ReLU激活函数，再通过第二个全连接层（fc2）和ReLU激活函数，最后通过第三个全连接层（fc3）得到输出q_values。这个输出表示对于每个动作的Q值。
    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)

        return q_values