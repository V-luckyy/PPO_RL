import gym
from gym import spaces
import numpy as np
import math


class BipedalEnv(gym.Env):
    def __init__(self, config):
        super(BipedalEnv, self).__init__()

        self.config = config

        # 状态空间定义：11维
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']

        # 定义动作空间和观察空间（显式 float32 避免 Box bound precision 警告）
        low = np.array([-self.config['max_torque'], -self.config['max_torque'], 0.0], dtype=np.float32)
        high = np.array([self.config['max_torque'], self.config['max_torque'],
                        self.config['max_swing_force']], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        low_obs = np.float32(-np.inf)
        high_obs = np.float32(np.inf)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, shape=(self.state_dim,), dtype=np.float32)

        # 初始状态
        self.state = np.array(self.config['initial_state'], dtype=np.float32)
        self.timestep = 0

    def reset(self):
        """重置环境"""
        self.state = np.array(self.config['initial_state'], dtype=np.float32)
        self.timestep = 0
        return self.state

    def step(self, action):
        """执行一步操作，并返回新的状态、奖励、是否完成和额外信息"""
        # 解构动作
        tau_hip, tau_knee, f_swing = action  # 确保这里的action是一个包含三个值的可迭代对象

        # 其余的环境更新逻辑...
        next_state = self._update_state(tau_hip, tau_knee, f_swing)

        reward = self._compute_reward(next_state, tau_hip, tau_knee)

        self.state = next_state
        self.timestep += 1
        done = self._check_done()

        return next_state, reward, done, {}

    def _update_state(self, tau_hip, tau_knee, f_swing):
        """
        基于动作（关节力矩、摆动腿推力）更新状态
        这个方法应该根据你提供的动力学方程来更新状态。
        目前是简化形式。
        """
        # 在此示例中，状态更新是一个简单的加权累加，实际情况需要根据动力学公式推导
        q_torso_x, dot_q_torso_x, q_torso_z, dot_q_torso_z, theta_hip, dot_theta_hip, theta_knee, dot_theta_knee, F_foot_x, F_foot_z, phi_foot = self.state

        # 假设机器人简单地更新这些状态
        next_state = np.copy(self.state)
        next_state[0] += dot_q_torso_x * 0.02  # 假设躯干x位置根据速度更新
        next_state[2] += dot_q_torso_z * 0.02  # 假设躯干z位置根据速度更新
        next_state[4] += dot_theta_hip * 0.02  # 假设髋关节角度根据速度更新
        next_state[6] += dot_theta_knee * 0.02  # 假设膝关节角度根据速度更新

        # 简化动力学模型：关节力矩影响角速度，外部力影响接触力
        next_state[1] += (tau_hip * 0.01)  # 更新髋关节速度
        next_state[3] += (tau_knee * 0.01)  # 更新膝关节速度
        next_state[9] += f_swing  # 摆动腿的推力影响足端接触力

        # 限制状态范围，防止发散导致评估曲线异常（如 torso_z 涨到 7）
        next_state[1] = np.clip(next_state[1], -5.0, 5.0)   # dot_q_torso_x
        next_state[3] = np.clip(next_state[3], -5.0, 5.0)   # dot_q_torso_z
        next_state[2] = np.clip(next_state[2], 0.3, 2.0)    # q_torso_z 躯干高度
        next_state[4] = np.clip(next_state[4], -np.pi, np.pi)   # theta_hip
        next_state[6] = np.clip(next_state[6], -np.pi, np.pi)   # theta_knee
        next_state[5] = np.clip(next_state[5], -5.0, 5.0)   # dot_theta_hip
        next_state[7] = np.clip(next_state[7], -5.0, 5.0)   # dot_theta_knee
        next_state[9] = np.clip(next_state[9], 0.0, 20.0)   # F_foot_z
        next_state[10] = 1.0 if next_state[10] > 0.5 else 0.0  # phi_foot 离散化

        return next_state

    def _compute_reward(self, state, tau_hip, tau_knee):
        """
        根据给定的状态计算奖励。
        包括平衡奖励、能量效率、步态周期性、安全奖励等。
        """
        q_torso_x, dot_q_torso_x, q_torso_z, dot_q_torso_z, theta_hip, dot_theta_hip, theta_knee, dot_theta_knee, F_foot_x, F_foot_z, phi_foot = state

        # 平衡奖励
        r_balance = math.exp(-2.0 * abs(q_torso_x)) + math.exp(-1.0 * abs(q_torso_z - 0.8))

        # 能量效率奖励（假设关节力矩平方作为消耗）
        r_energy = -0.005 * (tau_hip ** 2 + tau_knee ** 2)

        # 步态奖励（基于接触相位，周期性控制）
        T_stride = 1.0  # 假设步态周期为1秒
        r_gait = math.cos(2 * math.pi * self.timestep / T_stride) * (1 if phi_foot == 1 else 0)

        # 安全奖励（确保足端力不超过最大值，并限制关节角度）
        r_safety = -0.1 * max(0, F_foot_z - 10.0) - 0.2 * max(0, theta_knee - 2.0)

        # 总奖励 = 各项奖励的加权和
        reward = (1.0 * r_balance + 0.005 * r_energy + 1.0 * r_gait + 1.0 * r_safety)

        return reward

    def _check_done(self):
        """检查终止条件"""
        q_torso_x, dot_q_torso_x, q_torso_z, dot_q_torso_z, theta_hip, dot_theta_hip, theta_knee, dot_theta_knee, F_foot_x, F_foot_z, phi_foot = self.state

        # 躯干倾斜角度过大
        if abs(q_torso_x) > 15.0:
            return True

        # 质心高度过低
        if q_torso_z < 0.3:
            return True

        # 连续10步未触发摆动腿动作
        if self.timestep > 10 and phi_foot == 0:
            return True

        return False

    def render(self):
        """可视化环境状态"""
        print(f"Step: {self.timestep}, State: {self.state}")
