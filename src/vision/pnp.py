# pnp.py - PNP位姿解算器（添加中心点反投影）
import cv2
import numpy as np

class PNPSolver:
    """PNP位姿解算器"""
    
    def __init__(self, target_width=0.262, target_height=0.174):
        self.target_width = target_width
        self.target_height = target_height
        
        half_w = target_width / 2
        half_h = target_height / 2
        
        # 3D点顺序：左上→右上→右下→左下
        self.object_points = np.array([
            [-half_w, -half_h, 0],  # 左上
            [ half_w, -half_h, 0],  # 右上
            [ half_w,  half_h, 0],  # 右下
            [-half_w,  half_h, 0],  # 左下
        ], dtype=np.float32)
        
        # 靶子中心3D点（原点）
        self.center_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        
        # pnp.py 中更新为：
        self.camera_matrix = np.array([
            [581.516, 0.0,    385.208],
            [0.0,    582.147, 181.477],
            [0.0,    0.0,     1.0    ]
        ], dtype=np.float32)

        self.dist_coeffs = np.array([
            [0.263, -0.361, -0.023, 0.052, 0.395]
        ], dtype=np.float32)
        
        self.position = None
        self.distance = None
        self.yaw = None
        self.pitch = None
        
        # ===== 【新增】中心点投影坐标 =====
        self.center_projected = None  # PNP计算出的中心在图像上的投影
        self.center_error = None      # PNP中心与实际检测中心的偏差
        
        self._debug_count = 0
        self._debug_max = 5

    def solve(self, image_points):
        if len(image_points) != 4:
            self._reset()
            return False
        
        img_pts = np.array(image_points, dtype=np.float32).reshape(-1, 2)
        
        # ===== 调试输出 =====
        if self._debug_count < self._debug_max:
            print(f"\n=== PNP Debug Frame {self._debug_count} ===")
            print(f"  角点0(TL): ({img_pts[0][0]:.0f}, {img_pts[0][1]:.0f})")
            print(f"  角点1(TR): ({img_pts[1][0]:.0f}, {img_pts[1][1]:.0f})")
            print(f"  角点2(BR): ({img_pts[2][0]:.0f}, {img_pts[2][1]:.0f})")
            print(f"  角点3(BL): ({img_pts[3][0]:.0f}, {img_pts[3][1]:.0f})")
        
        # PNP求解
        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            img_pts,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE
        )
        
        if not success:
            self._reset()
            if self._debug_count < self._debug_max:
                print("  ❌ PNP求解失败!")
            self._debug_count += 1
            return False
        
        # 计算位置和角度
        self.position = tvec.flatten()
        x, y, z = self.position
        self.distance = np.linalg.norm(self.position)
        self.yaw = np.degrees(np.arctan2(x, z))
        self.pitch = np.degrees(np.arctan2(y, z))
        
        # ===== 【新增】将靶子中心3D点投影到图像上 =====
        center_proj, _ = cv2.projectPoints(
            self.center_3d,      # 靶子中心 (0,0,0)
            rvec, tvec,          # 旋转和平移向量
            self.camera_matrix,
            self.dist_coeffs
        )
        self.center_projected = center_proj[0][0]  # (cx_p, cy_p)
        
        # ===== 【新增】计算与实际检测中心的偏差 =====
        # 实际检测中心 = 四个角点的平均值
        real_center = np.mean(img_pts, axis=0)
        self.center_error = np.linalg.norm(self.center_projected - real_center)
        
        # 调试输出
        if self._debug_count < self._debug_max:
            print(f"  实际检测中心: ({real_center[0]:.1f}, {real_center[1]:.1f})")
            print(f"  PNP投影中心: ({self.center_projected[0]:.1f}, {self.center_projected[1]:.1f})")
            print(f"  中心偏差: {self.center_error:.1f} 像素")
            print(f"  位置: x={x:.3f}, y={y:.3f}, z={z:.3f}m")
            print(f"  距离: {self.distance:.3f}m")
            print(f"  Yaw: {self.yaw:.1f}°, Pitch: {self.pitch:.1f}°")
            self._debug_count += 1
        
        return True
    
    def _reset(self):
        self.position = None
        self.distance = None
        self.yaw = None
        self.pitch = None
        self.center_projected = None
        self.center_error = None