# src/control/laser_calibration.py
"""
激光笔位姿标定工具

用途：通过实验测量激光笔相对于相机的平移和旋转偏移
使用方法：
    1. 将靶子放在固定位置，用激光笔照射靶心
    2. 运行此脚本，记录多组数据
    3. 脚本会自动计算最优的补偿参数
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path


class LaserCalibrator:
    """激光笔位姿标定器"""
    
    def __init__(self):
        self.measurements = []  # 存储多组测量数据
        self.camera_matrix = np.array([
            [581.516, 0.0, 385.208],
            [0.0, 582.147, 181.477],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        self.dist_coeffs = np.array([
            [0.263, -0.361, -0.023, 0.052, 0.395]
        ], dtype=np.float32)
        
    def add_measurement(self, camera_position, rvec, laser_pixel):
        """
        添加一组标定数据
        
        参数:
            camera_position: PNP解算的靶心位置 (tx, ty, tz) 米
            rvec: 靶子姿态旋转向量 (rx, ry, rz)
            laser_pixel: 激光点在图像上的像素坐标 (u, v)
        """
        self.measurements.append({
            'camera_position': np.array(camera_position, dtype=np.float32),
            'rvec': np.array(rvec, dtype=np.float32),
            'laser_pixel': np.array(laser_pixel, dtype=np.float32)
        })
        print(f"已记录第 {len(self.measurements)} 组数据")
    
    def calibrate(self):
        """
        使用所有测量数据计算最优补偿参数
        
        返回:
            translation: (dx, dy, dz) 平移补偿
            rotation: (roll, pitch, yaw) 角度补偿（弧度）
        """
        if len(self.measurements) < 3:
            print("需要至少3组数据才能标定！")
            return None, None
        
        print(f"使用 {len(self.measurements)} 组数据开始标定...")
        
        # 初始猜测：平移和旋转都接近0
        initial_params = np.zeros(6)  # [dx, dy, dz, roll, pitch, yaw]
        
        # 使用最小二乘法优化
        from scipy.optimize import minimize
        
        result = minimize(
            self._objective_function,
            initial_params,
            method='BFGS',
            options={'maxiter': 1000, 'disp': True}
        )
        
        if result.success:
            params = result.x
            translation = params[0:3]
            rotation = params[3:6]
            
            print("\n" + "="*60)
            print("标定结果:")
            print(f"  平移补偿: dx={translation[0]:.4f} m, dy={translation[1]:.4f} m, dz={translation[2]:.4f} m")
            print(f"  角度补偿: roll={rotation[0]:.4f} rad, pitch={rotation[1]:.4f} rad, yaw={rotation[2]:.4f} rad")
            print(f"  角度补偿: roll={rotation[0]*180/np.pi:.2f}°, pitch={rotation[1]*180/np.pi:.2f}°, yaw={rotation[2]*180/np.pi:.2f}°")
            print(f"  重投影误差: {result.fun:.4f} 像素")
            print("="*60)
            
            return translation, rotation
        else:
            print("标定失败！")
            return None, None
    
    def _objective_function(self, params):
        """
        目标函数：最小化激光投影点与测量点的误差
        """
        dx, dy, dz, roll, pitch, yaw = params
        translation = np.array([dx, dy, dz])
        rotation = np.array([roll, pitch, yaw])
        
        total_error = 0.0
        
        for data in self.measurements:
            camera_pos = data['camera_position']
            rvec = data['rvec']
            measured_pixel = data['laser_pixel']
            
            # 计算补偿后的位置
            compensated_pos = self._compensate(camera_pos, rvec, translation, rotation)
            
            # 投影到图像平面
            projected_pixel = self._project_to_image(compensated_pos)
            
            # 计算误差
            error = np.linalg.norm(projected_pixel - measured_pixel)
            total_error += error**2
        
        return total_error / len(self.measurements)
    
    def _compensate(self, camera_pos, rvec, translation, rotation):
        """
        计算激光笔指向的位置（与laser_compensator.py中的逻辑一致）
        """
        # 旋转向量 → 旋转矩阵
        R_camera, _ = cv2.Rodrigues(rvec)
        
        # 计算激光笔的旋转
        R_offset = self._euler_to_rotation_matrix(rotation)
        R_laser = R_camera @ R_offset
        
        # 计算激光笔的平移
        t_laser = camera_pos + R_camera @ translation
        
        return t_laser
    
    def _project_to_image(self, position_3d):
        """将3D点投影到图像平面"""
        pt_3d = np.array([[position_3d[0]], [position_3d[1]], [position_3d[2]]], dtype=np.float32)
        pt_2d = self.camera_matrix @ pt_3d
        u = pt_2d[0, 0] / pt_2d[2, 0]
        v = pt_2d[1, 0] / pt_2d[2, 0]
        return np.array([u, v])
    
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
    
    def save_params(self, translation, rotation, filename="laser_params.json"):
        """保存标定参数到文件"""
        params = {
            'translation': {
                'dx': float(translation[0]),
                'dy': float(translation[1]),
                'dz': float(translation[2])
            },
            'rotation': {
                'roll': float(rotation[0]),
                'pitch': float(rotation[1]),
                'yaw': float(rotation[2])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(params, f, indent=4)
        
        print(f"参数已保存到: {filename}")
    
    def load_params(self, filename="laser_params.json"):
        """加载标定参数"""
        with open(filename, 'r') as f:
            params = json.load(f)
        
        translation = np.array([
            params['translation']['dx'],
            params['translation']['dy'],
            params['translation']['dz']
        ])
        rotation = np.array([
            params['rotation']['roll'],
            params['rotation']['pitch'],
            params['rotation']['yaw']
        ])
        
        return translation, rotation


# ===== 命令行交互工具 =====
def interactive_calibration():
    """
    交互式标定工具
    
    使用方法：
        1. 运行此函数
        2. 将靶子放在一个位置，用激光笔照射靶心
        3. 在画面中点击激光点位置（或按 's' 自动检测）
        4. 移动靶子到另一个位置，重复步骤2-3
        5. 至少采集3组数据后，按 'c' 进行标定
        6. 标定完成后按 'q' 退出
    """
    from src.vision.camera import Camera
    from src.vision.detector import Detector
    from src.vision.pnp import PNPSolver
    from src.vision.tracker import Tracker
    
    print("="*60)
    print("激光笔位姿标定工具")
    print("="*60)
    print("操作说明:")
    print("  1. 将靶子放在相机视野内")
    print("  2. 用激光笔照射靶心")
    print("  3. 按 's' 记录当前数据（自动检测激光点）")
    print("  4. 按 'm' 手动点击激光点位置")
    print("  5. 移动靶子到不同位置，重复步骤2-4")
    print("  6. 至少采集3组数据后，按 'c' 进行标定")
    print("  7. 按 'q' 退出")
    print("="*60)
    
    # 初始化
    cam = Camera(index=3, width=640, height=480, fps=120)
    pnp = PNPSolver()
    detector = Detector(rectangle_min_area=1000, rectangle_max_area=50000, pnp_solver=pnp)
    calibrator = LaserCalibrator()
    
    # 激光点检测参数
    laser_threshold = 200  # 激光点亮度阈值
    
    def detect_laser_point(frame):
        """自动检测激光点位置"""
        # 转换到HSV空间，检测红色/近红色区域
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 红色范围（激光笔通常是红色）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2
        
        # 找最大轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 10:
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return (cx, cy)
        return None
    
    print("\n等待摄像头启动...")
    collected_data = 0
    laser_pixel = None
    manual_mode = False
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("摄像头读取失败")
            break
        
        # 检测靶子
        binary, board = detector.detect(frame)
        result = detector.draw_boards(frame)
        
        # 显示当前状态
        status_y = 60
        cv2.putText(result, f"数据组数: {len(calibrator.measurements)}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(result, f"手动模式: {'ON' if manual_mode else 'OFF'}", 
                   (10, status_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 检测激光点
        auto_laser = detect_laser_point(frame)
        if auto_laser:
            cv2.circle(result, auto_laser, 8, (0, 0, 255), -1)
            cv2.putText(result, f"Laser: ({auto_laser[0]}, {auto_laser[1]})", 
                       (10, status_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            if not manual_mode:
                laser_pixel = auto_laser
        
        # 手动点击模式
        if manual_mode and laser_pixel:
            cv2.circle(result, laser_pixel, 8, (255, 0, 255), -1)
            cv2.putText(result, "Manual Laser Point", 
                       (laser_pixel[0]+15, laser_pixel[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # 显示靶子信息
        if board is not None and hasattr(detector.pnp, 'tvec') and detector.pnp.tvec is not None:
            tvec = detector.pnp.tvec
            if isinstance(tvec, tuple):
                tvec = np.array(tvec)
            cv2.putText(result, f"Target: ({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f}) m", 
                       (10, status_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 显示操作提示
        cv2.putText(result, "s: 记录  m:手动模式  c:标定  q:退出", 
                   (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Calibration", result)
        cv2.imshow("Binary", binary)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('m'):
            manual_mode = not manual_mode
            print(f"手动模式: {'开启' if manual_mode else '关闭'}")
            
            if manual_mode:
                def mouse_callback(event, x, y, flags, param):
                    global click_pos
                    if event == cv2.EVENT_LBUTTONDOWN:
                        click_pos = (x, y)
                
                click_pos = None
                cv2.setMouseCallback("Calibration", mouse_callback)
        
        elif key == ord('s'):
            # 记录数据
            if board is None:
                print("未检测到靶子！请确保靶子在视野内。")
                continue
            
            if laser_pixel is None:
                print("未检测到激光点！请确保激光笔照射在靶心附近。")
                continue
            
            # 获取PNP结果
            if not hasattr(detector.pnp, 'tvec') or detector.pnp.tvec is None:
                print("PNP解算失败！")
                continue
            
            tvec = detector.pnp.tvec
            if isinstance(tvec, tuple):
                tvec = np.array(tvec)
            
            rvec = detector.pnp.rvec
            if isinstance(rvec, tuple):
                rvec = np.array(rvec)
            
            # 添加到标定器
            calibrator.add_measurement(tvec, rvec, laser_pixel)
            collected_data += 1
            print(f"已记录第 {collected_data} 组数据")
        
        elif key == ord('c'):
            # 执行标定
            if len(calibrator.measurements) < 3:
                print(f"数据不足！需要至少3组，当前有 {len(calibrator.measurements)} 组")
                continue
            
            translation, rotation = calibrator.calibrate()
            if translation is not None and rotation is not None:
                # 保存参数
                calibrator.save_params(translation, rotation, "laser_params.json")
                print("\n标定完成！参数已保存到 laser_params.json")
                print("请将此文件放到主程序目录下使用")
    
    cam.cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    interactive_calibration()