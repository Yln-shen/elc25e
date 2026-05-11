import cv2
import numpy as np

class PNPSolver:
    """PNP位姿解算器"""
    
    def __init__(self, target_width=0.260, target_height=0.173, img_width = 640, img_height = 480, fov = 100 ):
        self.target_width = target_width
        self.target_height = target_height
        self.img_width = img_width
        self.img_height = img_height
        self.fov = fov
        
        half_w = target_width / 2
        half_h = target_height / 2
        self.object_points = np.array([
            [-half_w, -half_h, 0],  # 左上
            [-half_w,  half_h, 0],  # 左下
            [ half_w,  half_h, 0],  # 右下
            [ half_w, -half_h, 0],  # 右上
        ], dtype=np.float32)
        
        self.camera_matrix = self._calc_camera_matrix()
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32) # 无畸变
        
        self.position = None    # 3D位置 (x, y, z)
        self.distance = None    # 距离（米）
        self.yaw = None         # 偏航角（度）
        self.pitch = None       # 俯仰角（度）
    def _calc_camera_matrix(self):
        fov_rad = self.fov * np.pi / 180
        fx = (self.img_width / 2) / np.tan(fov_rad / 2)
        fy = fx
        cx = self.img_width / 2
        cy = self.img_height / 2
        
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0,  1 ]
        ], dtype=np.float32)

    def solve(self, image_points):
        if len(image_points) != 4:
            self.position = None
            self.distance = None
            self.yaw = None
            self.pitch = None
            return False
        
        img_pts = np.array(image_points, dtype=np.float32).reshape(-1, 2)
        
        #success：求解是否成功
        #rvec：旋转向量，表示靶子相对相机的姿态
        #tvec：平移向量，表示靶子中心在相机坐标系下的3D位置（单位：米）
        success, rvec, tvec = cv2.solvePnP(
            self.object_points,    # 3D点：靶子四个角的真实坐标（米）
            img_pts,               # 2D点：图像上四个角的像素坐标
            self.camera_matrix,    # 相机内参矩阵
            self.dist_coeffs,      # 畸变系数（全零）
            flags=cv2.SOLVEPNP_SQPNP  # 求解方法
        )
        
        if not success:
            self.position = None
            self.distance = None
            self.yaw = None
            self.pitch = None
            return False
        
        self.position = tvec.flatten()
        x, y, z = self.position
        self.distance = np.linalg.norm(self.position)
        self.yaw = np.degrees(np.arctan2(x, z))
        self.pitch = np.degrees(np.arctan2(y, z))

        print("角点:", [(f"{p[0]:.1f},{p[1]:.1f}") for p in image_points])
        print("左上:", image_points[0], "右下:", image_points[2])
        
        return True