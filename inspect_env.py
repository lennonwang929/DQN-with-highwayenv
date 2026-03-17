import gymnasium as gym
import highway_env

env = gym.make("highway-v0")

print("Observation space:")
# 这是什么？state空间吗？神经网络的输入？是个矩阵？是输入吗？
# Box(-inf, inf, (5, 5), float32)代表5个vehicles(5个agent),每个vehicle有5个特征（位置、速度等），数据类型是float32，值的范围是负无穷到正无穷。
print(env.observation_space)

print("Action space:")
# 这是动作空间？也是个矩阵？是输出吧
print(env.action_space)

# 这一步是重置环境，返回初始观察和信息。观察是环境的状态，信息可能包含一些额外的调试或环境相关的信息。
# obs指的是当前状态
obs, info = env.reset()

print("Observation shape:")
# 和上面的env.observation_space区别是什么？这个是实际的观察数据的形状，而上面的是观察空间的定义，描述了观察数据的结构和范围。
# 5*5的矩阵，5个agent，每个agent有5个特征
print(obs.shape)
# x,y,vx,vy,heading,每一行代表一个agent的状态信息，包括位置（x, y）、速度（vx, vy）和朝向（heading）。所以obs是一个5行5列的矩阵，每行对应一个agent，每列对应一个特征。
print(obs)
