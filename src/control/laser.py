# src/control/laser_compensator.py
import numpy as np
import cv2
import json
import os


class LaserCompensator:
    """激光笔位姿补偿器 - 简化版（只调X/Y）"""
    
    def __init__(self, params_file="laser_params.json"):
        # ===== 固定补偿参数（从标定获得，一般不变） =====
        # 激光相对摄像头的固定偏移（米）
        # 注意：摄像头在下，激光在上，所以dy为负值
        self.translation = np.array([0.00, -0.028, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])
        
        # ===== 距离自适应参数（用户调参） =====
        # 补偿量 = base_offset + distance * slope_offset
        # 单位：毫米 (mm)
        self.dx_base = 0.0      # 基础X偏移 (mm)
        self.dx_slope = 0.0     # X偏移随距离变化率 (mm/m)
        self.dy_base = 0.0      # 基础Y偏移 (mm)
        self.dy_slope = 0.0     # Y偏移随距离变化率 (mm/m)
        
        # 调参范围
        self.limits = {
            'dx_base': (-50, 50),
            'dx_slope': (-10, 10),
            'dy_base': (-50, 50),
            'dy_slope': (-10, 10),
        }
        
        # 尝试加载参数
        self.params_file = params_file
        if os.path.exists(params_file):
            self.load_params(params_file)
        else:
            print(f"使用默认参数，请通过滑块调参")
            print(f"  固定平移: dx={self.translation[0]*1000:.1f}mm, dy={self.translation[1]*1000:.1f}mm")
        
        self.last_compensated_position = None
        self.last_raw_position = None
        self.last_distance = None
        self.slider_window_open = False
    
    def compensate(self, camera_position, rvec):
        """带距离自适应的补偿"""
        if camera_position is None:
            return None, None
        
        if isinstance(camera_position, tuple):
            camera_position = np.array(camera_position, dtype=np.float32)
        if isinstance(rvec, tuple):
            rvec = np.array(rvec, dtype=np.float32)
        
        self.last_raw_position = camera_position.copy()
        
        # 计算当前距离
        dist = np.linalg.norm(camera_position)
        self.last_distance = dist
        
        # 旋转向量 → 旋转矩阵
        R_camera, _ = cv2.Rodrigues(rvec)
        
        # ===== 计算补偿量 (mm → m) =====
        dx_comp = (self.dx_base + dist * self.dx_slope) / 1000.0
        dy_comp = (self.dy_base + dist * self.dy_slope) / 1000.0
        
        # 应用补偿（只补偿X和Y）
        translation_comp = self.translation + np.array([dx_comp, dy_comp, 0.0])
        
        # 旋转补偿（固定）
        R_offset = self._euler_to_rotation_matrix(self.rotation)
        R_laser = R_camera @ R_offset
        
        # 最终位置
        t_laser = camera_position + R_camera @ translation_comp
        
        self.last_compensated_position = t_laser
        
        return t_laser, R_laser
    
    def get_angle_command(self, camera_position, rvec):
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
    
    # ========== 滑块调参 ==========
    
    def create_slider_window(self):
        """创建调参窗口（4个滑块）"""
        if self.slider_window_open:
            return
        
        cv2.namedWindow("Laser Adjust", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Laser Adjust", 350, 250)
        
        # dx_base: -50~50mm
        cv2.createTrackbar("dx_base", "Laser Adjust", 
                           int(self.dx_base + 50), 100, 
                           lambda v: setattr(self, 'dx_base', float(v - 50)))
        
        # dx_slope: -10~10mm/m
        cv2.createTrackbar("dx_slope", "Laser Adjust", 
                           int(self.dx_slope + 10), 20, 
                           lambda v: setattr(self, 'dx_slope', float(v - 10)))
        
        # dy_base: -50~50mm
        cv2.createTrackbar("dy_base", "Laser Adjust", 
                           int(self.dy_base + 50), 100, 
                           lambda v: setattr(self, 'dy_base', float(v - 50)))
        
        # dy_slope: -10~10mm/m
        cv2.createTrackbar("dy_slope", "Laser Adjust", 
                           int(self.dy_slope + 10), 20, 
                           lambda v: setattr(self, 'dy_slope', float(v - 10)))
        
        self.slider_window_open = True
        print("\n滑块调参窗口已打开")
        print("  dx_base: 基础左右偏移 (mm)")
        print("  dx_slope: 左右偏移随距离变化 (mm/m)")
        print("  dy_base: 基础上下偏移 (mm)")
        print("  dy_slope: 上下偏移随距离变化 (mm/m)")
    
    def update_slider_window(self):
        """更新滑块窗口显示"""
        if not self.slider_window_open:
            return
        
        img = np.zeros((150, 350, 3), dtype=np.uint8)
        y = 25
        
        # 显示当前距离
        dist_text = f"Distance: {self.last_distance:.2f}m" if self.last_distance else "Distance: --"
        cv2.putText(img, dist_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        
        # 显示当前实际补偿值
        if self.last_distance:
            dx = self.dx_base + self.last_distance * self.dx_slope
            dy = self.dy_base + self.last_distance * self.dy_slope
            cv2.putText(img, f"Offset: dx={dx:+.1f}mm, dy={dy:+.1f}mm", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y += 25
        
        cv2.putText(img, "Press 'S' to save, 'H' to hide", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow("Laser Adjust", img)
    
    def save_params(self, filename=None):
        """保存参数"""
        if filename is None:
            filename = self.params_file
        
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
            },
            'adaptive': {
                'dx_base': float(self.dx_base),
                'dx_slope': float(self.dx_slope),
                'dy_base': float(self.dy_base),
                'dy_slope': float(self.dy_slope),
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"\n参数已保存到: {filename}")
        self.print_status()
    
    def load_params(self, filename):
        """加载参数"""
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
            
            if 'adaptive' in params:
                self.dx_base = params['adaptive']['dx_base']
                self.dx_slope = params['adaptive']['dx_slope']
                self.dy_base = params['adaptive']['dy_base']
                self.dy_slope = params['adaptive']['dy_slope']
            
            print(f"参数已加载: {filename}")
            self.print_status()
            
        except Exception as e:
            print(f"加载参数失败: {e}")
    
    def print_status(self):
        """打印当前状态"""
        print("\n" + "=" * 50)
        print("激光补偿参数:")
        print(f"  固定平移: dx={self.translation[0]*1000:.1f}mm, dy={self.translation[1]*1000:.1f}mm")
        print(f"  距离自适应:")
        print(f"    dx = {self.dx_base:+.1f} + dist * {self.dx_slope:+.1f} mm")
        print(f"    dy = {self.dy_base:+.1f} + dist * {self.dy_slope:+.1f} mm")
        print("=" * 50)