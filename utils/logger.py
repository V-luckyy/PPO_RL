import os
import time


class Logger:
    def __init__(self, log_dir="logs"):
        """
        初始化Logger
        :param log_dir: 日志文件保存的目录
        """
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, "training_log.txt")

        # 创建日志目录（如果不存在的话）
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 打开日志文件并写入初始日志
        with open(self.log_file, 'w') as f:
            f.write("Training Log Started at {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    def log_loss(self, loss_value, step=None):
        """
        记录损失值
        :param loss_value: 当前训练中的损失值
        :param step: 可选，当前训练步数；若不传则使用内部计数
        """
        s = step if step is not None else self._get_current_step()
        with open(self.log_file, 'a') as f:
            f.write("Step {}: Loss = {}\n".format(s, loss_value))

    def log_reward(self, reward_value):
        """
        记录奖励值
        :param reward_value: 当前训练中的奖励值
        """
        with open(self.log_file, 'a') as f:
            f.write("Step {}: Reward = {}\n".format(self._get_current_step(), reward_value))

    def print_logs(self, step):
        """
        打印当前训练步数的日志
        :param step: 当前训练步数
        """
        print(f"Step {step}: Logs updated successfully.")

    def _get_current_step(self):
        """
        获取当前训练步数
        :return: 当前训练步数
        """
        # 假设从训练过程中提供当前步数，实际中可以从其他部分获取
        # 此处返回一个模拟步数
        return 100  # 模拟返回一个步数，实际代码中你可以根据训练进度返回真实的步数
