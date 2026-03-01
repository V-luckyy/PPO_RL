import torch
import gym
from ppo_agent import PPOAgent  # 导入 PPO 代理
from torch.distributions import Normal
from stable_baselines3.common.vec_env import DummyVecEnv
from bipedal_robot_env import BipedalRobotEnv  # 导入自定义的 BipedalRobotEnv 环境
import pygame


# 可视化函数
def visualize(model_path, num_episodes=10):
    # 创建和包装自定义环境
    env = BipedalRobotEnv(render_mode="human")  # 使用自定义环境
    env = DummyVecEnv([lambda: env])  # 使环境与 PPO 兼容
    agent = PPOAgent(env)

    pygame.init()
    pygame.display.set_mode((1200, 400))

    # 加载训练好的模型
    agent.load_model(model_path)
    print(f"加载模型 {model_path}")

    # 进行多轮评估
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        print(f"开始第 {episode + 1} 轮评估...")

        while not done:
            # 获取模型输出的动作
            state_tensor = torch.tensor(state[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 调整维度为 (1, 1, 11)
            action_mean, _ = agent.policy(state_tensor)
            dist = Normal(action_mean, torch.ones_like(action_mean))  # 高斯分布
            action = dist.sample().squeeze(0).numpy()  # 从分布中采样动作

            # 执行动作并获取下一个状态
            next_state, reward, done, _ = env.step(action)

            # 渲染机器人动作
            env.render()

            # 更新状态
            state = next_state


if __name__ == "__main__":
    model_path = "ppo_model.pth"  # 加载的训练好的模型路径
    visualize(model_path, num_episodes=1000)
