import gym
import torch
from ppo_agent import PPOAgent  # 导入 PPO 代理
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from bipedal_robot_env import BipedalRobotEnv  # 导入自定义的 BipedalRobotEnv 环境


# 创建训练函数
def train(num_episodes=1000, save_model_path="ppo_model.pth"):
    # 创建和包装自定义环境
    env = BipedalRobotEnv(render_mode="human")  # 使用自定义的环境
    env = DummyVecEnv([lambda: env])  # 使环境与 PPO 兼容

    # 初始化 PPO 代理
    agent = PPOAgent(env)

    # 训练
    print("开始训练...")
    agent.train(num_episodes=num_episodes)

    # 保存模型
    print(f"训练完成，保存模型到 {save_model_path}")
    agent.save_model(save_model_path)


if __name__ == "__main__":
    num_episodes = 10000  # 训练的轮次
    save_model_path = "ppo_model.pth"  # 保存模型的路径
    train(num_episodes=num_episodes, save_model_path=save_model_path)
