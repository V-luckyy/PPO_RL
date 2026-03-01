import torch
from envs.bipedal_env import BipedalEnv
from models.ppo import PPO
from models.networks import PolicyNetwork, ValueNetwork
from configs.default import CONFIG
from utils.visualization import Visualization


def test():
    # 初始化环境
    env = BipedalEnv(config=CONFIG)

    # 创建可视化工具实例
    visualizer = Visualization()

    # 创建PPO代理并加载训练好的模型
    ppo_agent = PPO(env, CONFIG, visualizer=visualizer)
    ppo_agent.load("ppo_bipedal_model.pth")  # 加载训练好的模型

    # 设置测试的最大步数
    max_steps = 1000
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    # 进行测试
    while not done and steps < max_steps:
        # 使用训练好的策略选择动作
        action, _ = ppo_agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        # 累积奖励
        total_reward += reward
        state = next_state
        steps += 1

        # 更新可视化数据
        visualizer.update_reward(steps, total_reward)

        # 每100步绘制一次奖励图
        if steps % 100 == 0:
            visualizer.plot_reward()

    print(f"Test completed. Total reward: {total_reward}, Steps: {steps}")

    # 绘制最终的奖励曲线
    visualizer.plot_reward()


if __name__ == "__main__":
    test()
