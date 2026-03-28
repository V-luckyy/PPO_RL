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
        # 6维：[tau_hip_L, tau_knee_L, f_swing_L, tau_hip_R, tau_knee_R, f_swing_R]
        low = np.array([-self.config['max_torque'], -self.config['max_torque'], 0.0,
                        -self.config['max_torque'], -self.config['max_torque'], 0.0], dtype=np.float32)
        high = np.array([self.config['max_torque'], self.config['max_torque'], self.config['max_swing_force'],
                         self.config['max_torque'], self.config['max_torque'], self.config['max_swing_force']], dtype=np.float32)
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
        # 解构动作 (分为左右腿)
        tau_hip_L, tau_knee_L, f_swing_L, tau_hip_R, tau_knee_R, f_swing_R = action

        # 其余的环境更新逻辑...
        next_state = self._update_state(tau_hip_L, tau_knee_L, f_swing_L, tau_hip_R, tau_knee_R, f_swing_R)

        reward = self._compute_reward(next_state, tau_hip_L, tau_knee_L, tau_hip_R, tau_knee_R)

        self.state = next_state
        self.timestep += 1
        done = self._check_done()

        return next_state, reward, done, {}

    def _update_state(self, tau_hip_L, tau_knee_L, f_swing_L, tau_hip_R, tau_knee_R, f_swing_R):
        """
        基于左右腿动作更新15维完整双足状态
        """
        # 解构所有15维状态
        (q_torso_x, dot_q_torso_x, q_torso_z, dot_q_torso_z,
         theta_hip_L, dot_theta_hip_L, theta_knee_L, dot_theta_knee_L,
         theta_hip_R, dot_theta_hip_R, theta_knee_R, dot_theta_knee_R,
         F_foot_z_L, F_foot_z_R, phi_foot) = self.state

        next_state = np.copy(self.state)
        dt = 0.02
        damping = 0.90
        g = 9.81
        
        # --- 1. 左腿关节更新 ---
        next_state[5] = next_state[5] * damping + (tau_hip_L * 0.1 * dt)
        next_state[7] = next_state[7] * damping + (tau_knee_L * 0.1 * dt)
        next_state[4] += next_state[5] * dt
        next_state[6] += next_state[7] * dt

        # --- 2. 右腿关节更新 ---
        next_state[9] = next_state[9] * damping + (tau_hip_R * 0.1 * dt)
        next_state[11] = next_state[11] * damping + (tau_knee_R * 0.1 * dt)
        next_state[8] += next_state[9] * dt
        next_state[10] += next_state[11] * dt

        # --- 3. 计算双脚落点高度与推力支持 ---
        # 左腿坐标
        leg_z_L = 0.4 * math.cos(next_state[4]) + 0.4 * math.cos(next_state[4] + next_state[6])
        ankle_height_L = next_state[2] - leg_z_L 
        
        # 右腿坐标
        leg_z_R = 0.4 * math.cos(next_state[8]) + 0.4 * math.cos(next_state[8] + next_state[10])
        ankle_height_R = next_state[2] - leg_z_R 
        
        force_z = -g
        force_x = 0.0
        
        next_state[12] = 0.0  # L foot force
        next_state[13] = 0.0  # R foot force

        # 左腿判定与发力
        if ankle_height_L < 0.05:
            total_angle_L = next_state[4] + next_state[6] * 0.5 
            base_support_z_L = g * 0.45  # 分担一半重力
            scaled_f_swing_L = f_swing_L * 15.0 
            Fz_L = base_support_z_L + scaled_f_swing_L * math.cos(total_angle_L)
            Fx_L = scaled_f_swing_L * math.sin(total_angle_L)
            force_z += Fz_L
            force_x += Fx_L
            next_state[12] = Fz_L
            
        # 右腿判定与发力
        if ankle_height_R < 0.05:
            total_angle_R = next_state[8] + next_state[10] * 0.5 
            base_support_z_R = g * 0.45  # 分担一半重力
            scaled_f_swing_R = f_swing_R * 15.0 
            Fz_R = base_support_z_R + scaled_f_swing_R * math.cos(total_angle_R)
            Fx_R = scaled_f_swing_R * math.sin(total_angle_R)
            force_z += Fz_R
            force_x += Fx_R
            next_state[13] = Fz_R

        # --- 4. 躯干全向动力学更新 ---
        next_state[1] = next_state[1] * damping + force_x * dt  # dot_q_torso_x
        next_state[3] = next_state[3] * damping + force_z * dt  # dot_q_torso_z
        next_state[0] += next_state[1] * dt  # torso_x
        next_state[2] += next_state[3] * dt  # torso_z

        # 限制状态范围防爆炸
        next_state[1] = np.clip(next_state[1], -5.0, 5.0)
        next_state[3] = np.clip(next_state[3], -5.0, 5.0)
        next_state[2] = np.clip(next_state[2], 0.3, 2.0)
        
        # 截断所有关节
        for i in [4, 6, 8, 10]:
            next_state[i] = np.clip(next_state[i], -np.pi, np.pi)
        for i in [5, 7, 9, 11]:
            next_state[i] = np.clip(next_state[i], -5.0, 5.0)

        # 稍微更新一下交替步态相位作为网络观察量（基于步数余弦）
        next_state[14] = math.cos(2 * math.pi * self.timestep / 40.0)

        return next_state

    def _compute_reward(self, state, tau_hip_L, tau_knee_L, tau_hip_R, tau_knee_R):
        """
        评估完整双足状态并计算奖励
        """
        (q_torso_x, dot_q_torso_x, q_torso_z, dot_q_torso_z,
         theta_hip_L, dot_theta_hip_L, theta_knee_L, dot_theta_knee_L,
         theta_hip_R, dot_theta_hip_R, theta_knee_R, dot_theta_knee_R,
         F_foot_z_L, F_foot_z_R, phi_foot) = state

        # 平衡与前移奖励
        r_balance = math.exp(-1.0 * abs(q_torso_z - 0.8))
        r_forward = dot_q_torso_x * 0.5  # 鼓励向前走

        # 能量效率
        r_energy = -0.002 * (tau_hip_L**2 + tau_knee_L**2 + tau_hip_R**2 + tau_knee_R**2)

        # 步态分离奖励（强烈惩罚双腿像僵尸一样贴在一起）
        angle_diff = abs(theta_hip_L - theta_hip_R)
        r_gait = 0.5 * angle_diff  # 两腿劈开越远奖励越大

        # 安全惩罚
        r_safety = -0.1 * max(0, F_foot_z_L - 10.0) - 0.1 * max(0, F_foot_z_R - 10.0)

        # 总奖励
        reward = (1.5 * r_balance + r_forward + r_energy + r_gait + r_safety)
        return reward

    def _check_done(self):
        """检查终止条件"""
        q_torso_z = self.state[2]
        F_foot_z_L = self.state[12]
        F_foot_z_R = self.state[13]

        # 质心高度过低判定跌倒
        if q_torso_z < 0.35:
            return True


        # 躯干前进跑得太远或者落后太远判定失败（可选）
        if self.state[0] < -2.0 or self.state[0] > 100.0:
            return True

        # 脚部长时间悬空判定为摔倒
        if self.timestep > 20 and F_foot_z_L < 0.1 and F_foot_z_R < 0.1:
            return True

        return False

    def render(self):
        """可视化环境状态"""
        print(f"Step: {self.timestep}, State: {self.state}")
