# -*- coding: utf-8 -*-
"""
基于评估状态序列的机器人实时回放（参考 clauld 可视化方式）
使用 Pygame 将状态历史逐帧绘制为 2D 连杆图。
"""
import numpy as np
import pygame
import sys
import os

# 项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 显示参数（与 clauld 的 BipedalRobotEnv 类似）
SCALE = 100  # 像素/米
SCREEN_W = 800
SCREEN_H = 500
GROUND_Y = SCREEN_H - 80
FPS = 30


def state_to_poses(state):
    """将 11 维状态转为 (躯干顶, 髋, 膝, 足) 像素坐标"""
    state = np.asarray(state, dtype=np.float64)
    q_torso_x, _, q_torso_z, _, theta_hip, _, theta_knee, _, _, _, _ = state
    thigh_len, calf_len = 0.4, 0.4
    hip_x = q_torso_x
    hip_z = q_torso_z
    knee_x = hip_x - thigh_len * np.sin(theta_hip)
    knee_z = hip_z + thigh_len * np.cos(theta_hip)
    ankle_x = knee_x - calf_len * np.sin(theta_hip + theta_knee)
    ankle_z = knee_z + calf_len * np.cos(theta_hip + theta_knee)
    cx = SCREEN_W // 2
    x_px = cx + int(hip_x * SCALE)
    z_px = GROUND_Y - int(hip_z * SCALE)
    kx_px = cx + int(knee_x * SCALE)
    kz_px = GROUND_Y - int(knee_z * SCALE)
    ax_px = cx + int(ankle_x * SCALE)
    az_px = GROUND_Y - int(ankle_z * SCALE)
    tx_px = cx + int(hip_x * SCALE)
    tz_px = GROUND_Y - int((hip_z + 0.3) * SCALE)
    return (tx_px, tz_px), (x_px, z_px), (kx_px, kz_px), (ax_px, az_px)


def run_playback(state_history, title="Robot Pose - Replay"):
    """在 Pygame 窗口中回放 state_history。"""
    if not state_history or len(state_history) == 0:
        return
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    idx = 0
    n = len(state_history)
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
        if idx >= n:
            idx = 0
        state = state_history[idx]
        (torso_top, hip, knee, ankle) = state_to_poses(state)
        screen.fill((255, 255, 255))
        pygame.draw.line(screen, (0, 0, 0), (0, GROUND_Y), (SCREEN_W, GROUND_Y), 2)
        pygame.draw.line(screen, (0, 0, 0), hip, torso_top, 5)
        pygame.draw.line(screen, (0, 0, 255), hip, knee, 4)
        pygame.draw.line(screen, (0, 180, 0), knee, ankle, 4)
        pygame.draw.circle(screen, (0, 0, 0), torso_top, 8)
        pygame.draw.circle(screen, (0, 0, 255), hip, 6)
        pygame.draw.circle(screen, (0, 180, 0), knee, 6)
        pygame.draw.circle(screen, (200, 0, 0), ankle, 5)
        text = font.render("Step %d / %d" % (idx, n), True, (0, 0, 0))
        screen.blit(text, (10, 10))
        pygame.display.flip()
        clock.tick(FPS)
        idx += 1
    pygame.quit()
