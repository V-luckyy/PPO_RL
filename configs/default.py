# default.py

CONFIG = {
    # PPO算法相关超参数
    'learning_rate': 0.0003,  # 学习率
    'gamma': 0.99,  # 折扣因子
    'epsilon': 0.2,  # Clipped Surrogate Objective的epsilon参数
    'lambda': 0.95,  # GAE参数
    'batch_size': 64,  # 每批次的样本数量
    'total_timesteps': 10000,  # 总的训练步数
    'log_interval': 10,  # 训练过程中日志打印间隔
    'plot_interval': 50,  # GUI 训练曲线更新步长（步数越小曲线越密）
    'save_interval': 10000,  # 模型保存间隔步数

    # 状态和动作空间的维度
    'state_dim': 15,  # 状态空间维度，15维 (含左右腿各2关节角度/速度及脚底接触标志)
    'action_dim': 6,  # 动作空间维度，6维 (左右腿各：hip torque, knee torque, swing force)

    # 动作空间相关限制
    'max_torque': 10,  # 最大关节力矩
    'max_swing_force': 1.0,  # 最大摆动腿力

    # 奖励函数的权重
    'reward_weights': {
        'balance': 1.0,
        'energy': 0.005,
        'gait': 1.0,
        'safety': 1.0
    },

    # 训练环境的相关配置
    'max_episode_length': 1000,  # 每个episode的最大步数
    # 初始状态: [0]躯干x, [1]躯干x速度, [2]躯干z(高度), [3]躯干z速度,
    # [4]左髋角, [5]左髋角速度, [6]左膝角, [7]左膝角速度,
    # [8]右髋角, [9]右髋角速度, [10]右膝角, [11]右膝角速度,
    # [12]左足端力z, [13]右足端力z, [14]步态相位(辅助判断或占位)
    'initial_state': [0.0, 0.0, 0.8, 0.0, 
                      0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 1.0]
}
