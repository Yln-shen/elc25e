# src/control/laser.py
import numpy as np
import cv2

class LaserCompensator:
    """激光笔位姿补偿器"""
    
    def __init__(self):
        # ===== 需要标定的参数 =====
        # 平移补偿（激光笔相对摄像头的位置，单位：米）
        self.translation = np.array([0.03, 0.01, 0.0])  # 右3cm，上1cm
        
        # 角度补偿（激光笔相对摄像头的旋转，单位：弧度）
        self.rotation = np.array([0.0, 0.0, 0.0])  # (roll, pitch, yaw)
        
        # 保存上次计算的补偿结果
        self.last_compensated_position = None
    
    def compensate(self, camera_position, rvec):
        """
        将相机位姿转换为激光笔位姿
        
        参数:
            camera_position: PNP 解算的相机位置 (3,)
            rvec: PNP 解算的旋转向量 (3,)
        
        返回:
            laser_position: 激光笔需要指向的位置 (3,)
        """
        if camera_position is None:
            return None
        
        # 1. 旋转向量 → 旋转矩阵
        R_camera, _ = cv2.Rodrigues(rvec)
        
        # 2. 计算激光笔的旋转
        R_offset = self._euler_to_rotation_matrix(self.rotation)
        R_laser = R_camera @ R_offset
        
        # 3. 计算激光笔的平移
        # 注意：平移偏移是在相机坐标系中定义的，需要旋转到世界坐标系
        t_laser = camera_position + R_camera @ self.translation
        
        self.last_compensated_position = t_laser
        
        return t_laser, R_laser
    
    def _euler_to_rotation_matrix(self, euler):
        """欧拉角 → 旋转矩阵"""
        roll, pitch, yaw = euler
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx
    
    def set_translation(self, dx, dy, dz):
        """手动设置平移补偿"""
        self.translation = np.array([dx, dy, dz])
    
    def set_rotation(self, roll, pitch, yaw):
        """手动设置角度补偿"""
        self.rotation = np.array([roll, pitch, yaw])