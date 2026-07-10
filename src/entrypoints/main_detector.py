# src/entrypoints/main_detector.py
import cv2
import time
import numpy as np
from src.vision.camera import Camera
from src.vision.detector import Detector
from src.vision.pnp import PNPSolver
from src.vision.tracker import Tracker  # ← 新增
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
    5. Tracker 在 3D 域滤波
    6. 从滤波后的 3D 位置计算 (yaw, pitch)
    7. 显示调试信息
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
    
    # 4. 初始化激光目标
    laser_pos = None
    
    # 5. PNP 解算 + 激光补偿
    if board is not None and pnp.position is not None:
        camera_pos = pnp.position
        rvec = pnp.rvec
        
        # 计算激光目标 3D 位置
        laser_pos, laser_rot = laser.compensate(camera_pos, rvec)
        
        # 显示原始激光目标（未滤波）
        if laser_pos is not None:
            cv2.putText(result, f"Raw Laser: ({laser_pos[0]:.3f}, {laser_pos[1]:.3f}, {laser_pos[2]:.3f})", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # ===== 6. 3D 域滤波（核心） =====
    # Tracker.track_3d() 接收 3D 位置，返回滤波后的 3D 位置
    filtered_pos = tracker.track_3d(laser_pos)
    
    # 7. 从滤波后的 3D 位置计算角度
    if filtered_pos is not None:
        yaw, pitch = tracker.get_yaw_pitch()
        
        # 显示滤波后的激光目标
        cv2.putText(result, f"Filt Laser: ({filtered_pos[0]:.3f}, {filtered_pos[1]:.3f}, {filtered_pos[2]:.3f})", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 显示滤波后的角度
        cv2.putText(result, f"Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°", 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # TODO: 发送角度给云台
        # servo.set_target(yaw, pitch)
    
    # ===== 8. 绘制调试信息（滤波前后对比 + 轨迹） =====
    # 获取激光笔在图像中的像素位置（用于投影）
    # 这里简化：使用画面中心作为参考点
    h, w = result.shape[:2]
    laser_pixel = (w // 2, h // 2)  # 可以使用 board.center 作为参考
    
    # 绘制 Tracker 调试信息（原始值、滤波值、轨迹、预测）
    result = tracker.draw_debug(result, laser_pixel)
    
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
    
    # 核心组件
    pnp = PNPSolver()
    detector = Detector(
        rectangle_min_area=1000,
        rectangle_max_area=50000,
        pnp_solver=pnp
    )
    
    laser = LaserCompensator()
    laser.set_translation(dx=0.03, dy=0.01, dz=0.0)
    laser.set_rotation(roll=0.0, pitch=0.0, yaw=0.0)
    
    # ===== Tracker（3D 域滤波） =====
    tracker = Tracker(
        vfov=100,
        img_width=640,
        use_kf=True,      # 启用滤波
        frame_add=35      # 丢失后预测 35 帧
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