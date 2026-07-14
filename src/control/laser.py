# src/control/laser_compensator.py
import numpy as np
import cv2
import json
import os


class LaserCompensator:
    """激光笔位姿补偿器"""
    
    def __init__(self, params_file="laser_params.json"):
        """
        参数:
            params_file: 标定参数文件路径
        """
        # 默认参数（未标定时使用）
        self.translation = np.array([0.00, 0.03, 0.0])  # 相机正上方3cm
        self.rotation = np.array([0.0, 0.0, 0.0])       # 假设平行
        
        # 尝试加载标定参数
        self.params_file = params_file
        if os.path.exists(params_file):
            self.load_params(params_file)
        else:
            print(f"警告: 标定参数文件 {params_file} 不存在，使用默认参数")
            print("请运行 laser_calibration.py 进行标定")
        
        # 保存上次计算的补偿结果
        self.last_compensated_position = None
        self.last_compensated_rotation = None
    
    def compensate(self, camera_position, rvec, target_point=None):
        """
        将相机位姿转换为激光笔位姿
        
        参数:
            camera_position: PNP 解算的靶心位置 (tx, ty, tz) 米
            rvec: PNP 解算的旋转向量 (3,)
            target_point: 目标点在激光坐标系中的位置（可选）
        
        返回:
            laser_target: 激光笔需要指向的位置 (3,)
            laser_rotation: 激光笔的旋转矩阵
        """
        if camera_position is None:
            return None, None
        
        # 确保是numpy数组
        if isinstance(camera_position, tuple):
            camera_position = np.array(camera_position, dtype=np.float32)
        if isinstance(rvec, tuple):
            rvec = np.array(rvec, dtype=np.float32)
        
        # 1. 旋转向量 → 旋转矩阵
        R_camera, _ = cv2.Rodrigues(rvec)
        
        # 2. 计算激光笔的旋转补偿
        R_offset = self._euler_to_rotation_matrix(self.rotation)
        R_laser = R_camera @ R_offset
        
        # 3. 计算激光笔的平移补偿
        # 平移偏移是在相机坐标系中定义的，需要旋转到世界坐标系
        t_laser = camera_position + R_camera @ self.translation
        
        # 保存结果
        self.last_compensated_position = t_laser
        self.last_compensated_rotation = R_laser
        
        return t_laser, R_laser
    
    def get_angle_command(self, camera_position, rvec):
        """
        直接获取云台角度指令
        
        返回:
            yaw: 水平偏航角（度）
            pitch: 垂直俯仰角（度）
        """
        laser_pos, _ = self.compensate(camera_position, rvec)
        
        if laser_pos is None:
            return None, None
        
        x, y, z = laser_pos
        if abs(z) < 1e-6:
            return 0.0, 0.0
        
        yaw = np.arctan2(x, z) * 180 / np.pi
        pitch = np.arctan2(-y, z) * 180 / np.pi
        
        return yaw, pitch
    
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
        """手动设置平移补偿（单位：米）"""
        self.translation = np.array([dx, dy, dz])
        print(f"平移补偿已更新: ({dx:.3f}, {dy:.3f}, {dz:.3f}) m")
    
    def set_rotation(self, roll, pitch, yaw):
        """手动设置角度补偿（单位：弧度）"""
        self.rotation = np.array([roll, pitch, yaw])
        print(f"角度补偿已更新: ({roll:.3f}, {pitch:.3f}, {yaw:.3f}) rad")
    
    def load_params(self, filename):
        """加载标定参数"""
        try:
            with open(filename, 'r') as f:
                params = json.load(f)
            
            self.translation = np.array([
                params['translation']['dx'],
                params['translation']['dy'],
                params['translation']['dz']
            ])
            self.rotation = np.array([
                params['rotation']['roll'],
                params['rotation']['pitch'],
                params['rotation']['yaw']
            ])
            
            print(f"标定参数已加载: {filename}")
            print(f"  平移: ({self.translation[0]:.4f}, {self.translation[1]:.4f}, {self.translation[2]:.4f}) m")
            print(f"  旋转: ({self.rotation[0]*180/np.pi:.2f}, {self.rotation[1]*180/np.pi:.2f}, {self.rotation[2]*180/np.pi:.2f})°")
            
        except Exception as e:
            print(f"加载参数失败: {e}")
    
    def save_params(self, filename):
        """保存当前参数"""
        params = {
            'translation': {
                'dx': float(self.translation[0]),
                'dy': float(self.translation[1]),
                'dz': float(self.translation[2])
            },
            'rotation': {
                'roll': float(self.rotation[0]),
                'pitch': float(self.rotation[1]),
                'yaw': float(self.rotation[2])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(params, f, indent=4)
        
        print(f"参数已保存到: {filename}")