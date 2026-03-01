import torch
import torch.optim as optim
import torch.nn.functional as F
from .networks import ActorCriticNetwork
import numpy as np
from utils.visualization import Visualization
from utils.logger import Logger


class PPO:
    def __init__(self, env, config, visualizer=None):
        """
        初始化PPO算法
        :param env: 强化学习环境
        :param config: 配置文件（包括超参数等）
        :param visualizer: 可视化工具实例
        """
        self.env = env
        self.config = config
        self.visualizer = visualizer  # 可视化工具实例
        self.logger = Logger()
        self.last_loss = 0.0
        self._has_updated = False  # 是否已执行过 update，用于曲线只记录有效 loss

        # 创建策略和价值网络（Actor-Critic网络）
        self.actor_critic = ActorCriticNetwork(input_dim=config['state_dim'], output_dim=config['action_dim'])

        # 使用Adam优化器
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config['learning_rate'])

    def select_action(self, state):
        """
        根据当前状态选择动作
        :param state: 当前状态
        :return: 选择的动作（包含tau_hip, tau_knee, f_swing）和该动作的log概率
        """
        state = torch.tensor(state, dtype=torch.float32)
        action_mu, action_sigma, _ = self.actor_critic(state)  # 获取动作概率分布
        # 防止 NaN 传入分布
        action_mu = torch.clamp(action_mu, -50.0, 50.0)
        action_sigma = torch.clamp(action_sigma, 1e-4, 10.0)
        dist = torch.distributions.Normal(action_mu, action_sigma)  # 使用正态分布生成连续动作
        action = dist.sample()  # 从分布中采样动作

        # 返回连续动作及其log概率
        return action, dist.log_prob(action)

    def compute_advantages(self, trajectory):
        """
        使用广义优势估计（GAE）计算优势函数
        :param trajectory: 存储的状态-动作-奖励-下一个状态的轨迹
        :return: 优势函数和目标值
        """
        advantages = []
        target_values = []
        gamma, lambd = self.config['gamma'], self.config['lambda']

        # 使用GAE计算每个时间步的优势和目标值
        for t in reversed(range(len(trajectory))):
            state, action, reward, next_state, done, log_prob = trajectory[t]
            state_value = self.actor_critic.value_net(torch.tensor(state, dtype=torch.float32))
            next_state_value = self.actor_critic.value_net(torch.tensor(next_state, dtype=torch.float32))
            delta = reward + gamma * next_state_value * (1 - done) - state_value
            advantage = delta + gamma * lambd * (1 - done) * (advantages[0] if advantages else 0)
            advantages.insert(0, advantage)
            target_values.insert(0, advantage + state_value)

        return advantages, target_values

    def update(self, trajectory, timestep=0):
        """
        更新PPO策略（批量更新，避免多次 backward + retain_graph 导致的原地修改错误）
        :param trajectory: 存储的状态-动作-奖励-下一个状态的轨迹
        :param timestep: 当前训练步数，用于日志和可视化
        """
        advantages, target_values = self.compute_advantages(trajectory)

        # 转为 Tensor 并 detach，避免计算图中张量被后续 backward 修改
        advantages = torch.tensor(
            [a.detach().item() if torch.is_tensor(a) else float(a) for a in advantages],
            dtype=torch.float32
        )
        target_values = torch.tensor(
            [t.detach().item() if torch.is_tensor(t) else float(t) for t in target_values],
            dtype=torch.float32
        )
        # 优势标准化，防止梯度爆炸导致 NaN
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 从轨迹中取出批量数据，全部 detach 避免梯度回传到旧图
        states = torch.tensor([t[0] for t in trajectory], dtype=torch.float32)
        actions = torch.stack([t[1].detach().clone() for t in trajectory])
        old_log_probs = torch.stack([t[5].detach().clone() for t in trajectory])  # (T, action_dim)

        # 多轮 PPO 更新，每轮一次批量前向 + 一次反向，不使用 retain_graph
        for _ in range(10):
            action_mu, action_sigma, state_values = self.actor_critic(states)
            # 防止 NaN 传入分布
            action_mu = torch.clamp(action_mu, -50.0, 50.0)
            action_sigma = torch.clamp(action_sigma, 1e-4, 10.0)
            dist = torch.distributions.Normal(action_mu, action_sigma)

            log_probs = dist.log_prob(actions)  # (T, action_dim)
            # 连续动作：对 action 维度求和得到每个 step 的 log_prob
            log_prob_per_step = log_probs.sum(dim=1)
            old_log_prob_per_step = old_log_probs.sum(dim=1)

            ratio = torch.exp(log_prob_per_step - old_log_prob_per_step)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config['epsilon'], 1.0 + self.config['epsilon']) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(state_values.squeeze(-1), target_values)
            entropy_loss = -dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=0.5)
            self.optimizer.step()

        # 记录损失和奖励（使用 policy_loss 作为曲线，与 reward 趋势更符合“loss 降 reward 升”）
        episode_reward = sum(t[2] for t in trajectory)
        policy_loss_val = policy_loss.item()
        self.last_loss = loss.item()
        self.logger.log_loss(self.last_loss, timestep)
        self._has_updated = True
        # 曲线数据仅在 train() 中每 50 步统一写入

    def train(self, total_timesteps, stop_callback=None):
        """
        训练PPO模型。训练曲线仅在此处、每 plot_interval 步统一写入一次，避免重复/归零。
        """
        state = self.env.reset()
        trajectory = []
        timestep = 0
        plot_interval = self.config.get('plot_interval', 50)
        current_ep_reward = 0.0  # 当前 episode 累计奖励

        while timestep < total_timesteps:
            if stop_callback and stop_callback():
                break

            action, log_prob = self.select_action(state)
            action_np = action.detach().cpu().numpy()
            next_state, reward, done, _ = self.env.step(action_np)

            trajectory.append((state, action, reward, next_state, done, log_prob))
            current_ep_reward += float(reward)
            state = next_state
            timestep += 1

            episode_ends_now = done or len(trajectory) >= self.config['max_episode_length']

            if episode_ends_now:
                self.update(trajectory, timestep)
                if self.visualizer and self._has_updated and timestep % plot_interval == 0:
                    ep_rew = sum(t[2] for t in trajectory)
                    self.visualizer.update_episode(timestep, self.last_loss, ep_rew)
                current_ep_reward = 0.0
                trajectory = []
                state = self.env.reset()
            else:
                if self.visualizer and self._has_updated and timestep % plot_interval == 0:
                    self.visualizer.update_episode(timestep, self.last_loss, current_ep_reward)

            if timestep % self.config['log_interval'] == 0:
                self.logger.print_logs(timestep)

        if self.visualizer:
            self.visualizer.plot_loss_and_reward()

    def save(self, path):
        """
        保存模型参数
        :param path: 保存路径
        """
        torch.save(self.actor_critic.state_dict(), path)

    def load(self, path):
        """
        加载模型参数
        :param path: 加载路径
        """
        self.actor_critic.load_state_dict(torch.load(path))
