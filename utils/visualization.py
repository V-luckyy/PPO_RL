import matplotlib.pyplot as plt
import numpy as np


class Visualization:
    def __init__(self):
        """
        初始化可视化工具
        """
        self.losses = []  # 存储损失值
        self.rewards = []  # 存储奖励值
        self.steps = []  # 存储训练步骤

    def update_loss(self, step, loss):
        """
        更新损失曲线
        :param step: 当前训练步数
        :param loss: 当前损失值
        """
        self.steps.append(step)
        self.losses.append(loss)

    def update_reward(self, step, reward):
        """
        更新奖励曲线
        :param step: 当前训练步数
        :param reward: 当前奖励值
        """
        self.steps.append(step)
        self.rewards.append(reward)

    def update_episode(self, step, loss, reward):
        """
        同时更新损失和奖励（每个 episode 结束时调用一次）
        :param step: 当前训练步数
        :param loss: 当前损失值
        :param reward: 当前 episode 累计奖励
        """
        self.steps.append(step)
        self.losses.append(loss)
        self.rewards.append(reward)

    def plot_loss(self):
        """
        绘制损失曲线
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.losses, label='Loss', color='r')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_reward(self):
        """
        绘制奖励曲线
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.rewards, label='Reward', color='g')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.title('Training Reward')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_loss_and_reward(self):
        """
        同时绘制损失和奖励曲线
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss', color='r')
        ax1.plot(self.steps, self.losses, label='Loss', color='r')
        ax1.tick_params(axis='y', labelcolor='r')

        ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
        ax2.set_ylabel('Reward', color='g')
        ax2.plot(self.steps, self.rewards, label='Reward', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        fig.tight_layout()  # 调整布局
        plt.title('Training Loss and Reward')
        plt.grid(True)
        plt.show()
