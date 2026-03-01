from envs.bipedal_env import BipedalEnv
from models.ppo import PPO
from utils.visualization import Visualization
from configs.default import CONFIG


def main():
    # 初始化训练环境
    env = BipedalEnv(config=CONFIG)

    # 创建可视化工具实例
    visualizer = Visualization()

    # 创建PPO代理并传入可视化工具
    ppo_agent = PPO(env, CONFIG, visualizer=visualizer)

    # 开始训练
    print(f"Starting training for {CONFIG['total_timesteps']} timesteps...")
    ppo_agent.train(total_timesteps=CONFIG['total_timesteps'])

    # 保存训练好的模型
    ppo_agent.save("ppo_bipedal_model.pth")
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
