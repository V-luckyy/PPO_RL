# -*- coding: utf-8 -*-
"""
PPO 双足机器人行走 - GUI 训练与评估界面
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import sys
import os
import traceback

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('TkAgg')
# 修复中文显示异常：指定支持中文的字体
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
except Exception:
    pass
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


# 默认配置（与 configs/default.py 一致）
DEFAULT_CONFIG = {
    'learning_rate': 0.0003,
    'gamma': 0.99,
    'epsilon': 0.2,
    'lambda': 0.95,
    'batch_size': 64,
    'total_timesteps': 10000,
    'log_interval': 10,
    'save_interval': 10000,
    'state_dim': 11,
    'action_dim': 3,
    'max_torque': 10,
    'max_swing_force': 1.0,
    'reward_weights': {
        'balance': 1.0,
        'energy': 0.005,
        'gait': 1.0,
        'safety': 1.0
    },
    'max_episode_length': 1000,
    'initial_state': [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
}


class PPOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PPO 双足机器人行走 - 训练与评估")
        self.root.geometry("1000x750")
        self.root.minsize(800, 600)

        # 训练状态
        self.training = False
        self.worker_thread = None
        self.visualizer = None
        self.ppo_agent = None
        self.env = None
        self.config = None
        self.model_path = "ppo_bipedal_model.pth"
        self.refresh_job = None
        self.eval_state_history = None  # 评估时采集的机器人状态序列，用于可视化
        self.train_visualizer = None  # 训练结束后的曲线数据，与评估分开避免叠加

        self._build_ui()

    def _build_ui(self):
        # 主布局：左侧参数，右侧图表
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)

        # === 左侧：参数设置 ===
        left_frame = ttk.LabelFrame(main_paned, text="模型参数", padding=10)
        main_paned.add(left_frame, weight=0)

        # 参数输入
        params = [
            ("learning_rate", "学习率", 0.0001, 0.01, DEFAULT_CONFIG['learning_rate']),
            ("gamma", "折扣因子 γ", 0.9, 1.0, DEFAULT_CONFIG['gamma']),
            ("epsilon", "PPO clip ε", 0.05, 0.5, DEFAULT_CONFIG['epsilon']),
            ("lambda", "GAE λ", 0.9, 1.0, DEFAULT_CONFIG['lambda']),
            ("batch_size", "批次大小", 16, 256, DEFAULT_CONFIG['batch_size']),
            ("total_timesteps", "总训练步数", 1000, 100000, DEFAULT_CONFIG['total_timesteps']),
            ("max_torque", "最大关节力矩", 1, 50, DEFAULT_CONFIG['max_torque']),
            ("max_swing_force", "最大摆动腿力", 0.1, 5.0, DEFAULT_CONFIG['max_swing_force']),
            ("max_episode_length", "Episode 最大步数", 100, 2000, DEFAULT_CONFIG['max_episode_length']),
        ]
        self.param_vars = {}
        for i, (key, label, low, high, default) in enumerate(params):
            row = ttk.Frame(left_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            self.param_vars[key] = (var, low, high)
            spin = ttk.Spinbox(row, textvariable=var, from_=low, to=high, width=10)
            spin.pack(side=tk.LEFT, padx=5)

        # 按钮区
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=15)

        self.btn_train = ttk.Button(btn_frame, text="开始训练", command=self._on_start_train)
        self.btn_train.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_stop = ttk.Button(btn_frame, text="停止训练", command=self._on_stop_train, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5, pady=5)

        ttk.Button(btn_frame, text="保存模型", command=self._on_save_model).pack(side=tk.LEFT, padx=5, pady=5)

        # 评估区
        eval_frame = ttk.LabelFrame(left_frame, text="模型评估", padding=5)
        eval_frame.pack(fill=tk.X, pady=10)

        self.eval_steps_var = tk.StringVar(value="500")
        ttk.Label(eval_frame, text="评估步数:").pack(anchor=tk.W)
        ttk.Spinbox(eval_frame, textvariable=self.eval_steps_var, from_=100, to=5000, width=10).pack(anchor=tk.W)

        ttk.Button(eval_frame, text="加载模型并评估", command=self._on_evaluate).pack(anchor=tk.W, pady=5)
        ttk.Button(eval_frame, text="查看训练曲线", command=lambda: self._on_show_curves(show_robot=True)).pack(anchor=tk.W, pady=2)
        ttk.Button(eval_frame, text="查看机器人姿态", command=self._on_show_robot_pose).pack(anchor=tk.W, pady=2)
        ttk.Button(eval_frame, text="实时仿真回放", command=self._on_robot_replay).pack(anchor=tk.W, pady=2)

        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(left_frame, textvariable=self.status_var, foreground="gray").pack(anchor=tk.W, pady=5)

        # === 右侧：训练过程 / 评估结果 ===
        right_frame = ttk.Frame(main_paned, padding=5)
        main_paned.add(right_frame, weight=1)

        # Matplotlib 图表
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax_loss = self.fig.add_subplot(211)
        self.ax_reward = self.fig.add_subplot(212, sharex=self.ax_loss)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _get_config(self):
        """从界面读取配置"""
        cfg = dict(DEFAULT_CONFIG)
        for key, (var, low, high) in self.param_vars.items():
            try:
                v = float(var.get())
                v = max(low, min(high, v))
                cfg[key] = v
                if key == 'batch_size' or key == 'total_timesteps' or key == 'max_episode_length':
                    cfg[key] = int(v)
            except ValueError:
                pass
        return cfg

    def _run_training(self):
        """在后台线程中运行训练"""
        try:
            from envs.bipedal_env import BipedalEnv
            from models.ppo import PPO
            from utils.visualization import Visualization

            self.config = self._get_config()
            self.env = BipedalEnv(config=self.config)
            self.visualizer = Visualization()
            self.ppo_agent = PPO(self.env, self.config, visualizer=self.visualizer)

            total = self.config['total_timesteps']
            stop_callback = lambda: not self.training
            self.ppo_agent.train(total_timesteps=total, stop_callback=stop_callback)
            self.ppo_agent.save(self.model_path)

            self.root.after(0, lambda: self._training_done(success=True))
        except Exception as e:
            err_msg = str(e).strip() if e else ""
            if not err_msg:
                err_msg = repr(e) or "未知错误"
            tb_str = traceback.format_exc()
            self.root.after(0, lambda e=err_msg, t=tb_str: self._training_done(success=False, err=e, tb=t))

    def _training_done(self, success=True, err=None, tb=None):
        self.training = False
        self.btn_train.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if success:
            self.status_var.set("训练完成，模型已保存至 " + self.model_path)
            self.train_visualizer = self.visualizer  # 保留训练曲线
            self._refresh_plot()
            # 提示：本环境奖励尺度较小，0.0003 附近属正常，关注趋势即可
        else:
            msg = (err or "未知错误").strip() or "未知错误"
            self.status_var.set("训练异常: " + msg)
            detail = msg if not tb else msg + "\n\n" + tb
            messagebox.showerror("训练失败", detail)

    def _on_start_train(self):
        if self.training:
            return
        self.training = True
        self.btn_train.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_var.set("训练中...")
        self.eval_state_history = None  # 新训练时清除上次评估的状态
        self.worker_thread = threading.Thread(target=self._run_training, daemon=True)
        self.worker_thread.start()
        self._schedule_refresh()

    def _on_stop_train(self):
        self.training = False
        self.status_var.set("正在停止...")

    def _schedule_refresh(self):
        if self.refresh_job:
            self.root.after_cancel(self.refresh_job)
        if self.training and self.visualizer:
            self._refresh_plot()
        if self.training:
            self.refresh_job = self.root.after(500, self._schedule_refresh)
        else:
            self.refresh_job = None

    def _refresh_plot(self, show_robot_state=False):
        """刷新图表。show_robot_state=True 时，若有评估状态数据则在上方显示机器人状态"""
        # 先移除可能存在的 twin 轴，避免多次评估后曲线叠加
        for ax in list(self.fig.axes):
            if ax not in (self.ax_loss, self.ax_reward):
                ax.remove()
        self.ax_loss.clear()
        self.ax_reward.clear()

        # 有评估结果时用评估数据，否则用训练数据
        if self.eval_state_history is not None and len(self.eval_state_history) > 0:
            steps = list(getattr(self.visualizer, 'steps', [])) if self.visualizer else []
            losses = list(getattr(self.visualizer, 'losses', [])) if self.visualizer else []
            rewards = list(getattr(self.visualizer, 'rewards', [])) if self.visualizer else []
        else:
            v = self.train_visualizer or self.visualizer
            steps = list(getattr(v, 'steps', [])) if v else []
            losses = list(getattr(v, 'losses', [])) if v else []
            rewards = list(getattr(v, 'rewards', [])) if v else []
        has_state = self.eval_state_history and len(self.eval_state_history) > 0
        if not steps and not has_state:
            return

        # 上图：评估且有状态时显示机器人状态；否则显示 Loss
        if (show_robot_state or not (losses and len(losses) == len(steps))) and has_state:
            self._plot_robot_state(self.ax_loss)
        elif losses and len(losses) == len(steps):
            self.ax_loss.plot(steps, losses, 'r-', label='Loss')
            self.ax_loss.set_ylabel('Loss')
        else:
            self.ax_loss.text(0.5, 0.5, '无 Loss 数据\n(训练后可见)', ha='center', va='center', transform=self.ax_loss.transAxes)
            self.ax_loss.set_ylabel('Loss')
        self.ax_loss.legend(loc='upper right')
        self.ax_loss.grid(True, alpha=0.3)

        # 下图：奖励曲线
        if rewards:
            r_steps = steps[:len(rewards)] if len(rewards) < len(steps) else steps
            self.ax_reward.plot(r_steps, rewards, 'g-', label='Reward')
        self.ax_reward.set_xlabel('Steps')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.legend(loc='upper right')
        self.ax_reward.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _plot_robot_state(self, ax):
        """在 ax 上绘制评估过程中机器人关键状态随时间变化"""
        hist = self.eval_state_history
        if not hist or len(hist) < 2:
            ax.text(0.5, 0.5, '无机器人状态数据', ha='center', va='center', transform=ax.transAxes)
            return
        steps = np.arange(len(hist))
        arr = np.array(hist)
        # 状态: q_torso_x, dot_x, q_torso_z, dot_z, theta_hip, dot_hip, theta_knee, dot_knee, F_x, F_z, phi
        theta_hip = np.degrees(arr[:, 4])
        theta_knee = np.degrees(arr[:, 6])
        torso_z = arr[:, 2]
        ax.plot(steps, theta_hip, 'b-', label='Hip (deg)', alpha=0.8)
        ax.plot(steps, theta_knee, 'm-', label='Knee (deg)', alpha=0.8)
        ax_twin = ax.twinx()
        ax_twin.plot(steps, torso_z, 'c--', label='Torso z (m)', alpha=0.8)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Joint angle (deg)', color='b')
        ax_twin.set_ylabel('Torso height (m)', color='c')
        # 合并为一个图例，避免与 Torso z 重叠
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_twin.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    def _on_save_model(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch 模型", "*.pth")],
            initialfile="ppo_bipedal_model.pth"
        )
        if path and self.ppo_agent:
            try:
                self.ppo_agent.save(path)
                self.status_var.set("模型已保存: " + path)
                messagebox.showinfo("保存成功", f"模型已保存至:\n{path}")
            except Exception as e:
                messagebox.showerror("保存失败", str(e))
        elif not self.ppo_agent:
            messagebox.showwarning("提示", "请先完成训练或加载模型")

    def _on_evaluate(self):
        """加载模型并评估，展示效果"""
        path = self.model_path
        if not os.path.exists(path):
            path = filedialog.askopenfilename(
                filetypes=[("PyTorch 模型", "*.pth")],
                title="选择模型文件"
            )
        if not path:
            return
        try:
            steps = int(self.eval_steps_var.get())
            steps = max(100, min(5000, steps))
        except ValueError:
            steps = 500

        def run_eval():
            try:
                from envs.bipedal_env import BipedalEnv
                from models.ppo import PPO
                from utils.visualization import Visualization

                config = self._get_config()
                env = BipedalEnv(config=config)
                viz = Visualization()
                agent = PPO(env, config, visualizer=viz)
                agent.load(path)

                state = env.reset()
                total_reward = 0
                done = False
                eval_steps = 0
                state_history = [state.copy()]  # 记录每步机器人状态，用于可视化
                while not done and eval_steps < steps:
                    action, _ = agent.select_action(state)
                    action_np = action.detach().cpu().numpy()
                    next_state, reward, done, _ = env.step(action_np)
                    total_reward += reward
                    state = next_state
                    state_history.append(state.copy())
                    eval_steps += 1
                    viz.update_reward(eval_steps, total_reward)

                self.visualizer = viz
                self.eval_state_history = state_history
                self.root.after(0, lambda: self._eval_done(total_reward, eval_steps))
            except Exception as e:
                self.root.after(0, lambda: self._eval_done(None, None, str(e)))

        def do_eval():
            # 每次评估前清空上次的评估曲线，避免叠加显示
            from utils.visualization import Visualization
            self.eval_state_history = None
            self.visualizer = Visualization()
            self._refresh_plot()
            self.status_var.set("评估中...")
            t = threading.Thread(target=run_eval, daemon=True)
            t.start()

        do_eval()

    def _eval_done(self, total_reward=None, eval_steps=None, err=None):
        if err:
            self.status_var.set("评估失败")
            messagebox.showerror("评估失败", err)
            return
        self.status_var.set(f"评估完成: 总奖励={total_reward:.2f}, 步数={eval_steps}")
        self._refresh_plot(show_robot_state=True)
        messagebox.showinfo("评估完成", f"总奖励: {total_reward:.2f}\n步数: {eval_steps}")

    def _on_show_curves(self, show_robot=False):
        """仅展示当前已有的训练曲线，若有评估状态则一并显示"""
        if self.train_visualizer or self.visualizer or self.eval_state_history:
            self._refresh_plot(show_robot_state=show_robot)
        else:
            messagebox.showinfo("提示", "请先完成训练或评估，才能查看曲线")

    def _on_show_robot_pose(self):
        """静态 2D 连杆图展示当前评估末态"""
        if not self.eval_state_history or len(self.eval_state_history) == 0:
            messagebox.showinfo("提示", "请先完成模型评估，才能查看机器人姿态")
            return
        state = self.eval_state_history[-1]
        self._show_robot_stick_figure(state)

    def _on_robot_replay(self):
        """参考 clauld：Pygame 实时回放评估状态序列"""
        if not self.eval_state_history or len(self.eval_state_history) == 0:
            messagebox.showinfo("提示", "请先完成模型评估，再点击「实时仿真回放」")
            return
        try:
            from utils.robot_replay import run_playback
        except Exception as e:
            messagebox.showerror("错误", "加载回放模块失败: " + str(e))
            return
        def run():
            run_playback(self.eval_state_history, title="PPO 双足机器人 - 实时回放")
        threading.Thread(target=run, daemon=True).start()
        self.status_var.set("已启动实时回放窗口，关闭窗口或按 ESC 结束")

    def _show_robot_stick_figure(self, state):
        """绘制 2D 连杆机器人姿态（确保机器人在视图内可见）"""
        state = np.asarray(state, dtype=np.float64)
        q_torso_x, _, q_torso_z, _, theta_hip, _, theta_knee, _, _, _, _ = state
        thigh_len, calf_len = 0.4, 0.4
        hip_x, hip_z = float(q_torso_x), float(q_torso_z)
        knee_x = hip_x - thigh_len * np.sin(theta_hip)
        knee_z = hip_z + thigh_len * np.cos(theta_hip)
        ankle_x = knee_x - calf_len * np.sin(theta_hip + theta_knee)
        ankle_z = knee_z + calf_len * np.cos(theta_hip + theta_knee)
        torso_top_z = hip_z + 0.3

        win = tk.Toplevel(self.root)
        win.title("机器人姿态")
        fig = Figure(figsize=(5, 5), dpi=100)
        ax = fig.add_subplot(111)
        # 先设范围再画，保证机器人一定在视野内
        margin = 0.15
        x_min = min(hip_x, knee_x, ankle_x) - margin
        x_max = max(hip_x, knee_x, ankle_x) + margin
        z_min = max(0, min(hip_z, knee_z, ankle_z) - margin)
        z_max = max(torso_top_z, hip_z, knee_z, ankle_z) + margin
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_aspect('equal')
        # 躯干、大腿、小腿
        ax.plot([hip_x, hip_x], [hip_z, torso_top_z], 'k-', linewidth=4, label='Torso')
        ax.plot([hip_x, knee_x], [hip_z, knee_z], 'b-', linewidth=3, label='Thigh')
        ax.plot([knee_x, ankle_x], [knee_z, ankle_z], 'g-', linewidth=3, label='Calf')
        ax.plot(hip_x, torso_top_z, 'ko', markersize=10)
        ax.plot(hip_x, hip_z, 'bo', markersize=8)
        ax.plot(knee_x, knee_z, 'go', markersize=8)
        ax.plot(ankle_x, ankle_z, 'ro', markersize=6)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('z (m)')
        ax.set_title('姿态: theta_hip={:.1f} deg  theta_knee={:.1f} deg'.format(float(np.degrees(theta_hip)), float(np.degrees(theta_knee))))
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
        ax.grid(True, alpha=0.3)
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main():
    root = tk.Tk()
    app = PPOApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
