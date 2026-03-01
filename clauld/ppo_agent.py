import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import gym


# 定义策略网络 (Actor) 和价值网络 (Critic)
class MLPPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(MLPPolicy, self).__init__()

        # Actor 网络 (用于计算动作均值)
        self.actor_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 输出动作均值
        )

        # Critic 网络 (用于计算状态值)
        self.critic_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出状态值
        )

    def forward(self, state):
        # 输出动作均值和状态值
        action_mean = self.actor_fc(state)
        state_value = self.critic_fc(state)

        return action_mean, state_value


# 定义 PPO 代理
class PPOAgent:
    def __init__(self, env, lr=2e-5, gamma=0.999, lam=0.95, clip_eps=0.15, update_epochs=10, batch_size=64):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        # 初始化策略网络 (Actor-Critic 网络)
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.shape[0]
        self.policy = MLPPolicy(input_dim, output_dim)

        # 定义优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def compute_advantages(self, rewards, values, next_values, dones):
        """
        计算优势估计（GAE）
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lam * (1 - dones[t]) * last_advantage
        return advantages

    def update_policy(self, states, actions, log_probs_old, returns, advantages):
        """
        更新策略
        """
        # 将数据转为 Tensor
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        # 优势标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 策略更新训练
        for _ in range(self.update_epochs):
            action_mean, state_value = self.policy(states)
            dist = Normal(action_mean, torch.ones_like(action_mean))  # 使用高斯分布

            log_probs = dist.log_prob(actions).sum(dim=-1)  # 计算动作的对数概率
            ratio = torch.exp(log_probs - log_probs_old)  # 比率，用于裁剪目标

            # 计算损失函数
            surrogate_loss = ratio * advantages
            clipped_loss = torch.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surrogate_loss, clipped_loss).mean()  # 最大化目标，取负值

            # 计算价值损失
            critic_loss = (returns - state_value).pow(2).mean()

            # 总损失
            loss = actor_loss + 0.5 * critic_loss

            # 更新网络参数
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, num_episodes=1000):
        """
        训练模型
        """
        for episode in range(num_episodes):
            states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
            state = self.env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_mean, state_value = self.policy(state_tensor)
                dist = Normal(action_mean, torch.ones_like(action_mean))  # 高斯分布
                action = dist.sample().squeeze(0).numpy()  # 从分布中采样动作
                log_prob = dist.log_prob(torch.tensor(action)).sum().item()

                # 存储数据
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(state_value.item())

                # 执行动作并返回下一状态
                next_state, reward, done, _ = self.env.step(action)

                rewards.append(reward)
                dones.append(done)

                state = next_state
                episode_reward += reward.item()

            # 计算优势和回报
            next_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            _, next_value = self.policy(next_state_tensor)
            advantages = self.compute_advantages(rewards, values, [next_value.item()] * len(rewards), dones)
            returns = advantages + np.array(values)

            # 更新策略
            self.update_policy(states, actions, log_probs, returns, advantages)

            # 打印每轮训练的详细信息
            if episode % 100 == 0:  # 每10轮打印一次
                print(f"Episode {episode + 1}/{num_episodes} completed. Episode Reward: {episode_reward:.2f}")
                # print(f"  Last 10 Episode Rewards: {rewards[-10:]}")
                # print(f"  Last 10 Episode Values: {values[-10:]}")
                # print(f"  Last 10 Episode Advantages: {advantages[-10:]}")
                # print(f"  Last 10 Episode Returns: {returns[-10:]}")

    def save_model(self, path):
        """
        保存训练好的模型
        """
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        """
        加载训练好的模型
        """
        self.policy.load_state_dict(torch.load(path))


# 示例：如何调用该 PPO 代理
if __name__ == "__main__":
    env = gym.make("BipedalRobotEnv")  # 你需要确保环境注册正确
    agent = PPOAgent(env)

    # 训练
    agent.train(num_episodes=1000)

    # 保存模型
    agent.save_model("ppo_model.pth")
