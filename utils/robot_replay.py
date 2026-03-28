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
    """将 15 维状态转为 (躯干顶, 髋, L膝, L足, R膝, R足) 像素坐标，并让摄像机沿X轴跟随左脚进度"""
    state = np.asarray(state, dtype=np.float64)
    (q_torso_x, _, q_torso_z, _, 
     theta_hip_L, _, theta_knee_L, _, 
     theta_hip_R, _, theta_knee_R, _, 
     _, _, _) = state
     
    thigh_len, calf_len = 0.4, 0.4
    hip_x = q_torso_x
    hip_z = q_torso_z
    
    # 左腿 (L)
    knee_x_L = hip_x - thigh_len * np.sin(theta_hip_L)
    knee_z_L = hip_z - thigh_len * np.cos(theta_hip_L)
    ankle_x_L = knee_x_L - calf_len * np.sin(theta_hip_L + theta_knee_L)
    ankle_z_L = knee_z_L - calf_len * np.cos(theta_hip_L + theta_knee_L)
    
    # 右腿 (R)
    knee_x_R = hip_x - thigh_len * np.sin(theta_hip_R)
    knee_z_R = hip_z - thigh_len * np.cos(theta_hip_R)
    ankle_x_R = knee_x_R - calf_len * np.sin(theta_hip_R + theta_knee_R)
    ankle_z_R = knee_z_R - calf_len * np.cos(theta_hip_R + theta_knee_R)
    
    # 调整 cx，让摄像机水平跟随髋关节
    cx = SCREEN_W // 2 - int(hip_x * SCALE)
    
    x_px = cx + int(hip_x * SCALE)
    z_px = GROUND_Y - int(hip_z * SCALE)
    
    kx_px_L = cx + int(knee_x_L * SCALE)
    kz_px_L = GROUND_Y - int(knee_z_L * SCALE)
    ax_px_L = cx + int(ankle_x_L * SCALE)
    az_px_L = GROUND_Y - int(ankle_z_L * SCALE)
    
    kx_px_R = cx + int(knee_x_R * SCALE)
    kz_px_R = GROUND_Y - int(knee_z_R * SCALE)
    ax_px_R = cx + int(ankle_x_R * SCALE)
    az_px_R = GROUND_Y - int(ankle_z_R * SCALE)

    tx_px = cx + int(hip_x * SCALE)
    tz_px = GROUND_Y - int((hip_z + 0.3) * SCALE)
    
    return ((tx_px, tz_px), (x_px, z_px), 
            (kx_px_L, kz_px_L), (ax_px_L, az_px_L), 
            (kx_px_R, kz_px_R), (ax_px_R, az_px_R), cx)


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
        (torso_top, hip, knee_L, ankle_L, knee_R, ankle_R, cx) = state_to_poses(state)
        screen.fill((255, 255, 255))
        pygame.draw.line(screen, (0, 0, 0), (0, GROUND_Y), (SCREEN_W, GROUND_Y), 2)
        
        # 绘制移动的地面网格（视觉提示机器人在前进）
        offset = cx % 100
        for x in range(offset - 100, SCREEN_W + 100, 100):
            pygame.draw.line(screen, (150, 150, 150), (x, GROUND_Y), (x, GROUND_Y + 10), 2)
            
        # 绘制右腿 (R) 靠后一层, 浅色
        pygame.draw.line(screen, (173, 216, 230), hip, knee_R, 6)    # 浅蓝
        pygame.draw.line(screen, (144, 238, 144), knee_R, ankle_R, 6) # 浅绿
        pygame.draw.circle(screen, (144, 238, 144), knee_R, 8)
        pygame.draw.circle(screen, (250, 128, 114), ankle_R, 7)      # 浅红足底
        
        # 绘制躯干和髋关节轴心 (黑)
        pygame.draw.line(screen, (0, 0, 0), hip, torso_top, 8)
        pygame.draw.circle(screen, (0, 0, 0), torso_top, 10)
        pygame.draw.circle(screen, (0, 0, 0), hip, 8)                # 黑底髋关节
        
        # 绘制左腿 (L) 靠前一层, 深色
        pygame.draw.line(screen, (0, 0, 255), hip, knee_L, 6)        # 深蓝
        pygame.draw.line(screen, (0, 180, 0), knee_L, ankle_L, 6)    # 深绿
        pygame.draw.circle(screen, (0, 180, 0), knee_L, 8)
        pygame.draw.circle(screen, (200, 0, 0), ankle_L, 7)          # 深红足底
        
        text = font.render(f"Step {idx} / {n}  | Dim: 15 BIPEDAL", True, (0, 0, 0))
        screen.blit(text, (10, 10))
        pygame.display.flip()
        clock.tick(FPS)
        idx += 1
    pygame.quit()
