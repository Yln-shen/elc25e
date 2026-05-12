import cv2
import numpy as np

class PNPSolver:
    """PNP位姿解算器"""
    
    def __init__(self, target_width=0.260, target_height=0.173):
        self.target_width = target_width
        self.target_height = target_height
        
        half_w = target_width / 2
        half_h = target_height / 2
        self.object_points = np.array([
            [-half_w, -half_h, 0],  # 左上
            [-half_w,  half_h, 0],  # 左下
            [ half_w,  half_h, 0],  # 右下
            [ half_w, -half_h, 0],  # 右上
        ], dtype=np.float32)
        
        # ============================================================
        # 🔴 重要：使用你实际标定的参数
        # ============================================================
        
        # 方法1：手动输入标定结果（推荐用于测试）
        self.camera_matrix = np.array([
            [499.356, 0.0,    334.895],
            [0.0,    496.552, 231.149],
            [0.0,    0.0,     1.0    ]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.array([
            [0.233, -0.445, -0.022, -0.000, 0.347]
        ], dtype=np.float32)
        
        # # 方法2：从标定文件加载（推荐用于部署）
        # self._load_calibration()
        
        self.position = None    # 3D位置 (x, y, z)
        self.distance = None    # 距离（米）
        self.yaw = None         # 偏航角（度）
        self.pitch = None       # 俯仰角（度）
    
    def _load_calibration(self):
        """从保存的标定文件加载参数"""
        try:
            calib_data = np.load("camera_calib_target.npz")
            self.camera_matrix = calib_data['camera_matrix']
            self.dist_coeffs = calib_data['dist_coeffs']
            print("✓ 成功加载标定参数")
            print(f"  内参:\n{self.camera_matrix}")
            print(f"  畸变: {self.dist_coeffs.ravel()}")
        except Exception as e:
            print(f"✗ 加载标定文件失败: {e}")
            print("  使用默认参数")
    
    def solve(self, image_points):
        if len(image_points) != 4:
            self.position = None
            self.distance = None
            self.yaw = None
            self.pitch = None
            return False
        
        img_pts = np.array(image_points, dtype=np.float32).reshape(-1, 2)
        
        # success：求解是否成功
        # rvec：旋转向量，表示靶子相对相机的姿态
        # tvec：平移向量，表示靶子中心在相机坐标系下的3D位置（单位：米）
        success, rvec, tvec = cv2.solvePnP(
            self.object_points,    # 3D点：靶子四个角的真实坐标（米）
            img_pts,               # 2D点：图像上四个角的像素坐标
            self.camera_matrix,    # 相机内参矩阵
            self.dist_coeffs,      # ⚠️ 现在是正确的畸变系数
            flags=cv2.SOLVEPNP_IPPE  # 或者用 SOLVEPNP_ITERATIVE
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
        
        # 调试输出（可选）
        # print(f"位置: x={x:.3f}, y={y:.3f}, z={z:.3f}m")
        # print(f"距离: {self.distance:.3f}m")
        # print(f"偏航: {self.yaw:.1f}°, 俯仰: {self.pitch:.1f}°")
        
        return True