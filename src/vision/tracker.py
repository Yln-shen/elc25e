# src/vision/tracker.py
import math
import time
import cv2
import numpy as np
from collections import deque
from .Kalman import AdaptiveEKF3D  # 新的 3D EKF

RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180


def time_diff(last_time=[None]):
    """计算两次调用之间的时间差（秒）"""
    current_time = time.time_ns()
    if last_time[0] is None:
        last_time[0] = current_time
        return 1e-9
    else:
        diff = current_time - last_time[0]
        last_time[0] = current_time
        return diff / 1e9


class Tracker:
    def __init__(self, vfov=100, img_width=640, use_kf=True, frame_add=35):
        self.vfov = vfov
        self.img_width = img_width
        self.use_kf = use_kf
        self.frame_add = frame_add
        
        # ===== 跟踪状态变量 =====
        self.lost = 0
        self.predict = False
        self.if_find = False
        
        # ===== 3D 卡尔曼滤波器（位置 + 速度） =====
        self.kf_3d = AdaptiveEKF3D(Q_base=0.5, R=1.0)
        
        # ===== 存储滤波后的 3D 位置 =====
        self.position_filtered = None  # (x, y, z) 滤波后
        self.position_raw = None       # (x, y, z) 原始观测
        
        # ===== 轨迹存储（用于绘制） =====
        self.trajectory = deque(maxlen=30)  # 最多保存 30 帧
        
        # ===== 预测状态 =====
        self.predicted_position = None  # 丢失时的预测位置
        
        # ===== 角度平滑 =====
        self.yaw_filtered = 0.0
        self.pitch_filtered = 0.0
        self.yaw_raw = 0.0
        self.pitch_raw = 0.0

    def track_3d(self, target_position_3d):
        """
        3D 域跟踪：对 (x, y, z) 进行滤波
        
        参数:
            target_position_3d: 激光目标 3D 位置 (x, y, z)，来自 LaserCompensator
        
        返回:
            (x, y, z): 滤波后的 3D 位置
        """
        dt = time_diff()
        
        # 保存原始观测值（用于绘制对比）
        self.position_raw = target_position_3d
        
        # ==========================================
        # 情况1：没有检测到目标
        # ==========================================
        if target_position_3d is None:
            if self.use_kf:
                self.lost += 1
                
                if self.lost <= self.frame_add and self.predict:
                    # ----- 预测模式 -----
                    self.kf_3d.predict(dt)
                    filtered = self.kf_3d.get_state()
                    self.position_filtered = filtered
                    self.predicted_position = filtered
                    self.if_find = True
                    
                    # 保存到轨迹
                    self.trajectory.append(filtered)
                    
                    return filtered
                else:
                    # ----- 完全丢失 -----
                    self.reset_kf()
                    self.lost = 0
                    self.predict = False
                    self.if_find = False
                    self.position_filtered = None
                    self.predicted_position = None
                    return None
            else:
                self.if_find = False
                self.position_filtered = None
                return None
        
        # ==========================================
        # 情况2：检测到目标
        # ==========================================
        else:
            self.predict = True
            self.if_find = True
            self.lost = 0
            self.predicted_position = None
            
            if self.use_kf:
                # 首次检测：初始化
                if not self.kf_3d.is_initialized:
                    self.kf_3d.set_initial_state(target_position_3d)
                    self.position_filtered = target_position_3d
                else:
                    # 正常 EKF 流程
                    self.kf_3d.predict(dt)
                    self.kf_3d.update(target_position_3d)
                    self.position_filtered = self.kf_3d.get_state()
            else:
                self.position_filtered = target_position_3d
            
            # 保存到轨迹
            if self.position_filtered is not None:
                self.trajectory.append(self.position_filtered)
            
            return self.position_filtered
    
    def get_yaw_pitch(self):
        """从滤波后的 3D 位置计算 (yaw, pitch)"""
        if self.position_filtered is None:
            return 0.0, 0.0
        
        x, y, z = self.position_filtered
        
        # 计算角度（注意：z 是距离，x 是左右，y 是上下）
        # 这里假设激光笔指向 z 轴正方向
        yaw = math.atan2(x, z) * RAD2DEG
        pitch = math.atan2(y, z) * RAD2DEG
        
        # 保存用于绘制
        self.yaw_filtered = yaw
        self.pitch_filtered = pitch
        
        return yaw, pitch
    
    def get_raw_yaw_pitch(self):
        """从原始 3D 位置计算 (yaw, pitch)（用于对比）"""
        if self.position_raw is None:
            return 0.0, 0.0
        
        x, y, z = self.position_raw
        yaw = math.atan2(x, z) * RAD2DEG
        pitch = math.atan2(y, z) * RAD2DEG
        
        self.yaw_raw = yaw
        self.pitch_raw = pitch
        
        return yaw, pitch
    
    def reset_kf(self):
        """重置滤波器"""
        self.kf_3d.reset()
        self.position_filtered = None
        self.predicted_position = None
        self.trajectory.clear()
    
    # ===== 绘制方法 =====
    def draw_debug(self, frame, laser_pixel):
        """
        在图像上绘制调试信息
        
        参数:
            frame: 原始图像
            laser_pixel: 激光笔在图像中的像素坐标 (x, y)
        
        绘制内容:
            1. 原始观测值（红色圆点）
            2. 滤波后的位置（绿色圆点）
            3. 预测位置（粉色叉号）
            4. 历史轨迹（橙色曲线）
            5. 原始 vs 滤波的连线（蓝色）
        """
        result = frame.copy()
        h, w = result.shape[:2]
        
        # 安全检查
        if self.position_raw is None and self.position_filtered is None:
            return result
        
        # ===== 1. 绘制历史轨迹（橙色曲线） =====
        if len(self.trajectory) > 1:
            pts = []
            for pos in self.trajectory:
                # 将 3D 位置映射到图像坐标（需要相机模型）
                # 简单映射：使用 laser_pixel 作为参考点
                # 这里假设 trajectory 中的位置是相对偏移
                x_img = int(laser_pixel[0] + pos[0] * 100)  # 缩放因子
                y_img = int(laser_pixel[1] + pos[1] * 100)
                if 0 <= x_img < w and 0 <= y_img < h:
                    pts.append((x_img, y_img))
            
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    cv2.line(result, pts[i-1], pts[i], (0, 165, 255), 2)
        
        # ===== 2. 绘制原始观测值（红色圆点） =====
        if self.position_raw is not None:
            # 将 3D 位置映射到图像坐标
            raw_x = int(laser_pixel[0] + self.position_raw[0] * 100)
            raw_y = int(laser_pixel[1] + self.position_raw[1] * 100)
            if 0 <= raw_x < w and 0 <= raw_y < h:
                cv2.circle(result, (raw_x, raw_y), 8, (0, 0, 255), -1)  # 红色实心
                cv2.putText(result, "RAW", (raw_x + 10, raw_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # ===== 3. 绘制滤波后的位置（绿色圆点） =====
        if self.position_filtered is not None:
            filt_x = int(laser_pixel[0] + self.position_filtered[0] * 100)
            filt_y = int(laser_pixel[1] + self.position_filtered[1] * 100)
            if 0 <= filt_x < w and 0 <= filt_y < h:
                cv2.circle(result, (filt_x, filt_y), 10, (0, 255, 0), -1)  # 绿色实心
                cv2.putText(result, "FILT", (filt_x + 12, filt_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
                
                # 连接原始和滤波（蓝色连线）
                if self.position_raw is not None:
                    raw_x = int(laser_pixel[0] + self.position_raw[0] * 100)
                    raw_y = int(laser_pixel[1] + self.position_raw[1] * 100)
                    if 0 <= raw_x < w and 0 <= raw_y < h:
                        cv2.line(result, (raw_x, raw_y), (filt_x, filt_y), (255, 200, 0), 1)
        
        # ===== 4. 绘制预测位置（粉色叉号） =====
        if self.predicted_position is not None:
            pred_x = int(laser_pixel[0] + self.predicted_position[0] * 100)
            pred_y = int(laser_pixel[1] + self.predicted_position[1] * 100)
            if 0 <= pred_x < w and 0 <= pred_y < h:
                cv2.drawMarker(result, (pred_x, pred_y), (255, 0, 255),
                              markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
                cv2.putText(result, "PRED", (pred_x + 12, pred_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # ===== 5. 显示角度信息 =====
        if self.position_filtered is not None:
            yaw, pitch = self.get_yaw_pitch()
            if self.position_raw is not None:
                raw_yaw, raw_pitch = self.get_raw_yaw_pitch()
                cv2.putText(result, f"Raw Y/P: ({raw_yaw:.2f}, {raw_pitch:.2f})", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(result, f"Filt Y/P: ({yaw:.2f}, {pitch:.2f})", 
                       (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 状态显示
        status = "TRACKING" if self.if_find else "LOST"
        color = (0, 255, 0) if self.if_find else (0, 0, 255)
        cv2.putText(result, f"Status: {status}", (10, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result