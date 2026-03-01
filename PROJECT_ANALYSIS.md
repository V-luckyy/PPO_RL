# PPO_RL 项目结构分析报告

## 一、项目概述

本项目是一个基于 **PPO (Proximal Policy Optimization)** 算法的强化学习仿真项目，用于模拟双足机器人行走。项目包含两套可并行的实现体系。

---

## 二、目录结构

```
PPO_RL/
├── configs/
│   └── default.py             # 主项目配置文件
├── envs/
│   ├── __init__.py
│   └── bipedal_env.py         # 主项目双足环境（简化版）
├── models/
│   ├── ppo.py                 # 主项目 PPO 核心实现
│   └── networks.py            # Actor-Critic 网络定义
├── utils/
│   ├── visualization.py       # 主项目可视化（matplotlib）
│   └── logger.py              # 训练日志
├── logs/                      # 训练日志输出目录
├── train.py                   # 主项目训练入口
├── test.py                    # 主项目测试/评估入口
├── requirements.txt
└── PROJECT_ANALYSIS.md
```

---

## 三、各文件作用说明

### 主体成分（主项目）

| 文件 | 作用 |
|------|------|
| `train.py` | 主项目入口，初始化环境、PPO、可视化并执行训练 |
| `test.py` | 加载训练好的模型并评估效果 |
| `configs/default.py` | 存储 PPO 超参数、状态/动作维度、奖励权重等配置 |
| `models/ppo.py` | PPO 算法核心：策略更新、GAE 优势估计、训练循环 |
| `models/networks.py` | PolicyNetwork、ValueNetwork、ActorCriticNetwork 定义 |
| `envs/bipedal_env.py` | 双足行走环境：状态空间、动作空间、动力学、奖励函数 |
| `utils/visualization.py` | 训练过程中 loss/reward 的 matplotlib 绘图 |
| `utils/logger.py` | 将训练 loss/reward 写入 logs/training_log.txt |

### 次要/辅助文件

| 文件 | 作用 |
|------|------|
| `envs/__init__.py` | 包初始化（当前为空） |
| `requirements.txt` | 依赖列表 |

### 冗余/重复实现（clauld 目录）

| 文件 | 作用 | 与主项目关系 |
|------|------|--------------|
| `clauld/bipedal_robot_env.py` | 带 Pygame 渲染的双足环境 | 与 `envs/bipedal_env.py` 功能重叠，API 和动力学不同 |
| `clauld/ppo_agent.py` | 另一套 PPO 实现 | 与 `models/ppo.py` 功能重叠 |
| `clauld/train.py` | 该体系的训练入口 | 与根目录 `train.py` 功能重叠 |
| `clauld/evaluate.py` | 模型评估 | 与根目录 `test.py` 功能重叠 |
| `clauld/visualize.py` | Pygame 渲染模型效果 | 主项目无对应可视化，可单独保留用于演示 |

### 不需要/可清理的文件

- **`clauld/` 整个目录**：若以主项目为主，可整体移除或归档；若需要 Pygame 实时渲染，可保留 `bipedal_robot_env.py` 和 `visualize.py` 作为可选扩展。
- **`logs/training_log.txt`**：由 Logger 自动生成，可加入 `.gitignore`，不纳入版本控制。

---

## 四、主项目核心流程

```
train.py
  └─> BipedalEnv(config)           # envs/bipedal_env.py
  └─> Visualization()              # utils/visualization.py
  └─> PPO(env, config, visualizer) # models/ppo.py
       └─> ActorCriticNetwork      # models/networks.py
  └─> ppo_agent.train()
       └─> select_action() -> env.step() -> update() [GAE + PPO clip]
  └─> ppo_agent.save()
```

---

## 五、可调节的主要参数（来自 configs/default.py）

| 参数 | 含义 | 默认值 |
|------|------|--------|
| learning_rate | 学习率 | 0.0003 |
| gamma | 折扣因子 | 0.99 |
| epsilon | PPO clip 范围 | 0.2 |
| lambda | GAE 参数 | 0.95 |
| batch_size | 批次大小 | 64 |
| total_timesteps | 总训练步数 | 10000 |
| max_torque | 最大关节力矩 | 10 |
| max_swing_force | 最大摆动腿力 | 1.0 |
| max_episode_length | 单 episode 最大步数 | 1000 |

---

## 六、依赖关系

主项目依赖：`gym`, `numpy`, `torch`, `matplotlib`  
clauld 额外依赖：`stable_baselines3`, `pygame`（用于渲染）

---

## 七、建议

1. **以主项目为主体**：`train.py` + `models/` + `envs/` + `configs/` + `utils/` 作为核心代码。
2. **clauld 目录**：若不需要 Pygame 可视化或 stable_baselines3，可删除；否则保留为可选扩展。
3. **Logger**：`utils/logger.py` 中 `_get_current_step()` 固定返回 100，建议改为接收真实训练步数，或由 PPO 传入。
