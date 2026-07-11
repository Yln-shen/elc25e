# src/entrypoints/main_detector.py
import cv2
import time
import numpy as np
from src.vision.camera import Camera
from src.vision.detector import Detector
from src.vision.pnp import PNPSolver
from src.vision.tracker import Tracker
from src.control.laser import LaserCompensator
from src.utils.decorators import measure_fps


@measure_fps
def run_loop(cam, detector, pnp, laser, tracker):
    """
    单帧处理逻辑
    
    数据流:
    1. 摄像头读取帧
    2. Detector 检测棋盘格
    3. PNP 解算位姿
    4. LaserCompensator 计算激光目标 3D 位置
    5. 将 3D 位置投影到图像平面 → (cx, cy)
    6. Tracker 在图像域用 2 个 1D EKF 滤波
    7. 从滤波后的 (cx, cy) 计算 (yaw, pitch)
    8. 显示调试信息
    """
    # 1. 读取帧
    ret, frame = cam.read()
    if not ret:
        return False, None
    
    # 2. 检测棋盘格
    binary, board = detector.detect(frame)
    result = detector.draw_boards(frame)
    
    # 3. 显示 FPS
    cv2.putText(result, f"FPS: {run_loop.fps:.1f}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # ===== 4. 初始化激光目标和像素中心 =====
    laser_pos = None
    center_pixel = None
    
    # 5. PNP 解算 + 激光补偿
    if board is not None and pnp.position is not None:
        camera_pos = pnp.position
        rvec = pnp.rvec
        
        # 计算激光目标 3D 位置
        laser_pos, laser_rot = laser.compensate(camera_pos, rvec)
        
        # 将 3D 位置投影到图像平面
        # 注意: 这里假设激光目标在相机坐标系中，需要投影到像素坐标
        # 实际投影需要使用 camera_matrix 和 dist_coeffs
        # 这里用简化方式: 直接从 board.center 获取
        if board.center is not None:
            center_pixel = board.center  # (cx, cy) 像素坐标
            
            # 显示原始激光目标（未滤波）
            cv2.putText(result, f"Raw: ({center_pixel[0]:.1f}, {center_pixel[1]:.1f})", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # ===== 6. 图像域滤波（2个1D EKF） =====
    # 如果 board 没有被检测到，center_pixel 为 None
    if board is None:
        center_pixel = None
    
    filtered_pixel = tracker.track(center_pixel)
    
    # 7. 从滤波后的像素坐标计算角度
    if filtered_pixel is not None:
        yaw, pitch = tracker.get_yaw_pitch()
        
        # 显示滤波后的位置
        cv2.putText(result, f"Filt: ({filtered_pixel[0]:.1f}, {filtered_pixel[1]:.1f})", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(result, f"Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # TODO: 发送角度给云台
        # servo.set_target(yaw, pitch)
    
    # ===== 8. 绘制调试信息 =====
    # 使用 board.center 作为参考点（如果没有，用画面中心）
    ref_point = board.center if board is not None else None
    result = tracker.draw_debug(result, ref_point)
    
    # 9. 显示
    cv2.imshow("Result", result)
    cv2.imshow("Binary", binary)
    
    return True, result


def main():
    # ===== 1. 初始化所有组件 =====
    try:
        cam = Camera(index=3, width=640, height=480, fps=120)
    except Exception as e:
        print(f"摄像头初始化失败: {e}，尝试默认摄像头...")
        cam = Camera(index=1)
    
    pnp = PNPSolver()
    detector = Detector(
        rectangle_min_area=1000,
        rectangle_max_area=50000,
        pnp_solver=pnp
    )
    
    laser = LaserCompensator()
    laser.set_translation(dx=0.03, dy=0.01, dz=0.0)
    laser.set_rotation(roll=0.0, pitch=0.0, yaw=0.0)
    
    # ===== Tracker（2个1D自适应EKF） =====
    tracker = Tracker(
        vfov=100,
        img_width=640,
        use_kf=True,      # 启用滤波
        frame_add=35      # 丢失后预测35帧
    )

    # ===== 2. 主循环 =====
    while True:
        running, _ = run_loop(cam, detector, pnp, laser, tracker)
        if not running:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()