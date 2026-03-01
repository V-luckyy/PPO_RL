import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        初始化策略网络
        :param input_dim: 输入维度（状态空间的维度）
        :param output_dim: 输出维度（动作空间的维度）
        """
        super(PolicyNetwork, self).__init__()

        # 第一层全连接层
        self.fc1 = nn.Linear(input_dim, 64)
        # 第二层全连接层
        self.fc2 = nn.Linear(64, 64)
        # 输出层：不使用softmax，输出均值和标准差
        self.fc_mu = nn.Linear(64, output_dim)  # 均值
        self.fc_sigma = nn.Linear(64, output_dim)  # 标准差

    def forward(self, state):
        """
        前向传播
        :param state: 当前状态
        :return: 动作的均值和标准差
        """
        x = F.relu(self.fc1(state))  # 激活函数ReLU
        x = F.relu(self.fc2(x))  # 激活函数ReLU
        mu = self.fc_mu(x)  # 输出动作的均值
        sigma = self.fc_sigma(x)  # 输出动作的标准差
        sigma = F.softplus(sigma) + 1e-4  # 确保标准差为正且不会过小，防止 NaN
        # 限制输出范围，防止训练后期爆炸导致 NaN
        mu = torch.clamp(mu, -50.0, 50.0)
        sigma = torch.clamp(sigma, 1e-4, 10.0)
        return mu, sigma


class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        """
        初始化价值网络
        :param input_dim: 输入维度（状态空间的维度）
        """
        super(ValueNetwork, self).__init__()

        # 第一层全连接层
        self.fc1 = nn.Linear(input_dim, 64)
        # 第二层全连接层
        self.fc2 = nn.Linear(64, 64)
        # 输出层，返回当前状态的估计值（值函数V(s)）
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        """
        前向传播
        :param state: 当前状态
        :return: 当前状态的估计值
        """
        x = F.relu(self.fc1(state))  # 激活函数ReLU
        x = F.relu(self.fc2(x))  # 激活函数ReLU
        return self.fc3(x)  # 返回状态的估计值


class ActorCriticNetwork(nn.Module):
    """
    一个同时包含策略网络和价值网络的类
    用于PPO算法，同时学习策略和价值函数
    """

    def __init__(self, input_dim, output_dim):
        """
        初始化Actor-Critic网络
        :param input_dim: 输入维度（状态空间的维度）
        :param output_dim: 输出维度（动作空间的维度）
        """
        super(ActorCriticNetwork, self).__init__()

        # 定义策略网络
        self.policy_net = PolicyNetwork(input_dim, output_dim)
        # 定义价值网络
        self.value_net = ValueNetwork(input_dim)

    def forward(self, state):
        """
        前向传播
        :param state: 当前状态
        :return: 策略网络的输出（均值和标准差）和价值网络的输出（状态的价值）
        """
        action_mu, action_sigma = self.policy_net(state)  # 获取均值和标准差
        state_value = self.value_net(state)  # 获取状态的估计值
        return action_mu, action_sigma, state_value
