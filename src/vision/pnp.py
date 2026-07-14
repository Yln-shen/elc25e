# src/vision/pnp.py
import cv2
import numpy as np
import math

RAD2DEG = 180 / math.pi

class PNPSolver:
    """
    PnP 解算器：将 2D 角点 + 3D 模型 + 相机内参 → 6DOF 位姿
    
    输入：
        - 4 个 2D 角点（原始像素坐标，原点在图像左上角）
        - 3D 世界模型（靶心为原点）
        - 相机内参 K + 畸变系数 D
    
    输出：
        - tvec: 靶心在相机坐标系下的位置 (x, y, z)，单位：米
        - rvec: 靶子平面的姿态（旋转向量）
        - yaw, pitch: 云台控制角
        - distance: 靶子到相机的欧氏距离
    """
    
    def __init__(self, target_width=0.262, target_height=0.174):
        """
        参数：
            camera_matrix: 3x3 内参矩阵 K
            dist_coeffs:   畸变系数 (k1,k2,p1,p2,k3)
            target_width:  靶子物理宽度（米），默认 0.262
            target_height: 靶子物理高度（米），默认 0.174
        """
        self.camera_matrix = np.array([
            [581.516, 0.0,    385.208],
            [0.0,    582.147, 181.477],
            [0.0,    0.0,     1.0    ]
        ], dtype=np.float32)

        self.dist_coeffs =  np.array([
            [0.263, -0.361, -0.023, 0.052, 0.395]
        ], dtype=np.float32)

        self.W = target_width
        self.H = target_height
        
        # 3D 世界模型：原点在靶心，矩形在 z=0 平面
        self.object_points = self._build_object_model()
        
        # 状态
        self.tvec = None
        self.rvec = None
        self.success = False
        self.reprojection_error = -1.0
        self.center_projected = None  # 添加这一行
        
    def _build_object_model(self):
        """
        构建靶子的 3D 模型。
        原点 (0,0,0) 在靶心，z=0 为靶子平面。
        
        返回顺序必须与 order_points 一致：
        左上、右上、右下、左下
        """
        W_half = self.W / 2
        H_half = self.H / 2
        
        pts = np.array([
            [-W_half,  H_half, 0],  # 左上
            [ W_half,  H_half, 0],  # 右上
            [ W_half, -H_half, 0],  # 右下
            [-W_half, -H_half, 0]   # 左下
        ], dtype=np.float32)
        
        return pts
    
    def solve(self, image_points):
        """
        核心解算函数。
        
        参数：
            image_points: np.array, shape (4, 2), dtype=np.float32
                          4 个角点的原始像素坐标
                          顺序：左上、右上、右下、左下
        
        返回：
            dict: {
                'success': bool,
                'tvec': (x, y, z),    # 米，靶心在相机坐标系的位置
                'rvec': (rx, ry, rz), # 旋转向量
                'yaw': float,         # 水平偏转角（度）
                'pitch': float,       # 垂直俯仰角（度）
                'distance': float,    # 欧氏距离（米）
                'error': float,       # 重投影误差（像素）
                'center_projected': (cx, cy)  # 靶心在图像上的投影坐标
            }
        """
        result = {
            'success': False,
            'tvec': None,
            'rvec': None,
            'yaw': 0.0,
            'pitch': 0.0,
            'distance': 0.0,
            'error': -1.0,
            'center_projected': None  # 添加这一行
        }
        
        # ===== 1. 基础检查 =====
        if self.camera_matrix is None or self.dist_coeffs is None:
            return result
        
        if image_points is None or len(image_points) != 4:
            return result
        
        # ===== 2. PnP 解算 =====
        # 使用 SOLVEPNP_IPPE：专门为共面 4 点设计，精度更高
        try:
            success, rvec, tvec = cv2.solvePnP(
                self.object_points, 
                image_points.reshape(4, 2), 
                self.camera_matrix, 
                self.dist_coeffs, 
                flags=cv2.SOLVEPNP_IPPE
            )
        except cv2.error:
            return result
        
        if not success:
            return result
        
        # ===== 3. 合理性检查 =====
        tvec = tvec.flatten()
        
        # 距离必须为正（靶子不能在相机后面）
        if tvec[2] <= 0.01:
            return result
        
        # 距离不能太远（超过 20 米一般是角点误匹配）
        if tvec[2] > 20.0:
            return result
        
        # ===== 4. 计算重投影误差 =====
        reproj_points, _ = cv2.projectPoints(
            self.object_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        reproj_points = reproj_points.reshape(-1, 2)
        error = np.mean(np.linalg.norm(image_points - reproj_points, axis=1))
        
        # 重投影误差过大，说明解算不可靠
        if error > 5.0:
            return result
        
        # ===== 5. 计算靶心投影点（中心投影）=====
        # 物体坐标系中的原点 (0,0,0) 就是靶心
        center_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        center_projected, _ = cv2.projectPoints(
            center_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        center_projected = center_projected.reshape(-1, 2)  # shape (2,)
        
        # ===== 6. 输出结果 =====
        self.tvec = tvec
        self.rvec = rvec.flatten()
        self.success = True
        self.reprojection_error = error
        self.center_projected = center_projected  # 保存到实例
        
        x, y, z = tvec
        distance = np.sqrt(x*x + y*y + z*z)
        yaw = math.atan2(x, z) * RAD2DEG
        pitch = math.atan2(-y, z) * RAD2DEG  # 注意符号：y 向上为正，pitch 抬头为正
        
        result = {
            'success': True,
            'tvec': (x, y, z),
            'rvec': tuple(self.rvec),
            'yaw': yaw,
            'pitch': pitch,
            'distance': distance,
            'error': error,
            'center_projected': center_projected  # 添加到返回结果
        }
        
        return result
    
    def is_valid(self):
        """检查上次解算是否有效"""
        return self.success
    
    def get_tvec(self):
        """获取靶心在相机坐标系的位置"""
        return self.tvec.copy() if self.tvec is not None else None