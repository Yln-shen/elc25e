# src/vision/tracker.py
import math
import time
import cv2
import numpy as np
from collections import deque
from .Kalman import AdaptiveEKF1D

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
    """
    2D 目标跟踪器 (两个独立1D EKF)
    
    状态: 对 cx 和 cy 分别滤波
    模型: 匀速 (Constant Velocity)
    特点: 自适应Q，算力极低
    """
    def __init__(self, vfov=100, img_width=640, use_kf=True, frame_add=35):
        self.vfov = vfov
        self.img_width = img_width
        self.use_kf = use_kf
        self.frame_add = frame_add
        
        # ===== 跟踪状态 =====
        self.lost = 0
        self.predict = False
        self.if_find = False
        
        # ===== 两个独立的1D自适应EKF =====
        # X方向滤波器 (水平)
        self.kf_cx = AdaptiveEKF1D(Q_base=0.5, R=1.0)
        # Y方向滤波器 (垂直)
        self.kf_cy = AdaptiveEKF1D(Q_base=0.5, R=1.0)
        
        # ===== 数据存储 =====
        self.position_filtered = None  # (cx, cy) 滤波后
        self.position_raw = None       # (cx, cy) 原始观测
        
        # ===== 轨迹存储 (用于绘制) =====
        self.trajectory = deque(maxlen=30)  # 最多保存30帧
        
        # ===== 预测状态 =====
        self.predicted_position = None  # 丢失时的预测位置
        
        # ===== 角度 =====
        self.yaw_filtered = 0.0
        self.pitch_filtered = 0.0

    def track(self, center_pixel):
        """
        2D 域跟踪：对 (cx, cy) 进行滤波
        
        参数:
            center_pixel: 目标中心像素坐标 (cx, cy)
        
        返回:
            (cx, cy): 滤波后的像素坐标，或 None
        """
        dt = time_diff()
        
        # 保存原始观测值（用于绘制对比）
        self.position_raw = center_pixel
        
        # ==========================================
        # 情况1：没有检测到目标
        # ==========================================
        if center_pixel is None:
            if self.use_kf:
                self.lost += 1
                
                if self.lost <= self.frame_add and self.predict:
                    # ----- 预测模式 -----
                    pred_cx = self.kf_cx.predict(dt)
                    pred_cy = self.kf_cy.predict(dt)
                    self.position_filtered = (pred_cx, pred_cy)
                    self.predicted_position = self.position_filtered
                    self.if_find = True
                    
                    # 保存到轨迹
                    self.trajectory.append(self.position_filtered)
                    
                    return self.position_filtered
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
            
            cx, cy = center_pixel
            
            if self.use_kf:
                # 首次检测：直接初始化
                if not self.kf_cx.is_initialized:
                    self.kf_cx.set_initial_state(cx)
                    self.kf_cy.set_initial_state(cy)
                    self.position_filtered = (cx, cy)
                else:
                    # 正常 EKF 流程: 先预测，再更新
                    self.kf_cx.predict(dt)
                    self.kf_cy.predict(dt)
                    filt_cx = self.kf_cx.update(cx)
                    filt_cy = self.kf_cy.update(cy)
                    self.position_filtered = (filt_cx, filt_cy)
            else:
                self.position_filtered = (cx, cy)
            
            # 保存到轨迹
            if self.position_filtered is not None:
                self.trajectory.append(self.position_filtered)
            
            return self.position_filtered

    def pixel_to_yaw_pitch(self, laser_center):
        """将像素坐标转换为偏航角和俯仰角"""
        if laser_center is None:
            return 0.0, 0.0
        
        cx, cy = laser_center
        
        vfov_rad = self.vfov * DEG2RAD
        focal = (self.img_width / 2) / math.tan(vfov_rad / 2)
        
        if abs(focal) < 1e-6:
            return 0.0, 0.0
        
        yaw = math.atan(cx / focal) * RAD2DEG
        pitch = math.atan(cy / focal) * RAD2DEG
        
        return yaw, pitch
    
    def get_yaw_pitch(self):
        """从滤波后的像素坐标计算 (yaw, pitch)"""
        if self.position_filtered is None:
            return 0.0, 0.0
        
        yaw, pitch = self.pixel_to_yaw_pitch(self.position_filtered)
        self.yaw_filtered = yaw
        self.pitch_filtered = pitch
        return yaw, pitch
    
    def get_raw_yaw_pitch(self):
        """从原始观测值计算 (yaw, pitch)（用于对比）"""
        if self.position_raw is None:
            return 0.0, 0.0
        
        return self.pixel_to_yaw_pitch(self.position_raw)
    
    def reset_kf(self):
        """重置所有滤波器"""
        self.kf_cx.reset()
        self.kf_cy.reset()
        self.position_filtered = None
        self.predicted_position = None
        self.trajectory.clear()

    # ===== 绘制调试信息 =====
    def draw_debug(self, frame, board_center_pixel):
        """
        在图像上绘制调试信息
        
        参数:
            frame: 原始图像
            board_center_pixel: 棋盘格中心的像素坐标 (用于参考点)
        
        绘制内容:
            1. 原始观测值 (红色圆点)
            2. 滤波后的位置 (绿色圆点)
            3. 预测位置 (粉色叉号)
            4. 历史轨迹 (橙色曲线)
            5. 原始 vs 滤波的连线 (蓝色)
        """
        result = frame.copy()
        h, w = result.shape[:2]
        
        # 如果没有参考点，使用画面中心
        if board_center_pixel is None:
            board_center_pixel = (w // 2, h // 2)
        
        ref_x, ref_y = board_center_pixel
        
        # ===== 1. 绘制历史轨迹（橙色曲线） =====
        if len(self.trajectory) > 1:
            pts = []
            for pos in self.trajectory:
                if pos is None:
                    continue
                # 轨迹是相对偏移，加上参考点
                x_img = int(ref_x + pos[0])
                y_img = int(ref_y + pos[1])
                if 0 <= x_img < w and 0 <= y_img < h:
                    pts.append((x_img, y_img))
            
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    cv2.line(result, pts[i-1], pts[i], (0, 165, 255), 2)
        
        # ===== 2. 绘制原始观测值（红色圆点） =====
        if self.position_raw is not None:
            raw_x = int(ref_x + self.position_raw[0])
            raw_y = int(ref_y + self.position_raw[1])
            if 0 <= raw_x < w and 0 <= raw_y < h:
                cv2.circle(result, (raw_x, raw_y), 8, (0, 0, 255), -1)
                cv2.putText(result, "RAW", (raw_x + 10, raw_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # ===== 3. 绘制滤波后的位置（绿色圆点） =====
        if self.position_filtered is not None:
            filt_x = int(ref_x + self.position_filtered[0])
            filt_y = int(ref_y + self.position_filtered[1])
            if 0 <= filt_x < w and 0 <= filt_y < h:
                cv2.circle(result, (filt_x, filt_y), 10, (0, 255, 0), -1)
                cv2.putText(result, "FILT", (filt_x + 12, filt_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
                
                # 连接原始和滤波（蓝色连线）
                if self.position_raw is not None:
                    raw_x = int(ref_x + self.position_raw[0])
                    raw_y = int(ref_y + self.position_raw[1])
                    if 0 <= raw_x < w and 0 <= raw_y < h:
                        cv2.line(result, (raw_x, raw_y), (filt_x, filt_y), (255, 200, 0), 1)
        
        # ===== 4. 绘制预测位置（粉色叉号） =====
        if self.predicted_position is not None:
            pred_x = int(ref_x + self.predicted_position[0])
            pred_y = int(ref_y + self.predicted_position[1])
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
        
        # 显示滤波器残差（调试用）
        if self.kf_cx.is_initialized:
            res_cx = self.kf_cx.get_residual()
            res_cy = self.kf_cy.get_residual()
            cv2.putText(result, f"Residual: ({res_cx:.1f}, {res_cy:.1f})", 
                       (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return result