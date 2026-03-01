import torch
import gym
import numpy as np
import pygame  # 添加 pygame 导入
from ppo_agent import PPOAgent  # 导入 PPO 代理
from bipedal_robot_env import BipedalRobotEnv  # 导入自定义的 BipedalRobotEnv 环境


# 评估函数
def evaluate(model_path, num_episodes=100):
    # 创建和包装自定义环境
    env = BipedalRobotEnv(render_mode="human")  # 使用自定义环境
    agent = PPOAgent(env)

    # 初始化 pygame 显示模式
    pygame.init()
    pygame.display.init()

    # 加载训练好的模型
    agent.load_model(model_path)
    print(f"加载模型 {model_path}")

    # 进行多轮评估
    total_rewards = 0.0
    for episode in range(num_episodes):
        state, _ = env.reset()  # 解包reset返回的元组，获取state
        done = False
        episode_reward = 0.0

        while not done:
            # 获取模型输出的动作
            action_mean, _ = agent.policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action = action_mean.detach().numpy()
            # 执行动作并获取下一个状态
            next_state, reward, done, _, _ = env.step(action[0])
            episode_reward += reward  # 累加奖励
            state = next_state

            # 渲染机器人动作
            env.render()

        total_rewards += episode_reward
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    # 计算平均奖励
    avg_reward = total_rewards / num_episodes
    print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")


if __name__ == "__main__":
    model_path = "ppo_model.pth"  # 加载的训练好的模型路径
    evaluate(model_path, num_episodes=100)
