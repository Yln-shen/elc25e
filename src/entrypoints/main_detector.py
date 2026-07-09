# src/entrypoints/main.py
import cv2
import time
from src.vision.camera import Camera
from src.vision.detector import Detector
from src.vision.pnp import PNPSolver
from src.control.laser import LaserCompensator

def main():
    # ===== 1. 初始化所有组件 =====
    try:
        cam = Camera(index=3, width=640, height=480, fps=120)
    except Exception as e:
        print(f"摄像头初始化失败: {e}，尝试默认摄像头...")
        cam = Camera(index=1)
    
    pnp = PNPSolver()
    detector = Detector(
        rectangle_min_area=100,
        rectangle_max_area=500000,
        pnp_solver=pnp
    )
    
    # 激光补偿器（独立组件）
    laser = LaserCompensator()
    
    # 设置补偿参数（需要实际标定）
    laser.set_translation(dx=0.03, dy=0.01, dz=0.0)
    laser.set_rotation(roll=0.0, pitch=0.0, yaw=0.0)
    
    fps = 0
    fps_last = 0
    fps_timer = time.time()

    # ===== 2. 主循环 =====
    while True:
        # 获取图像
        ret, frame = cam.read()
        if not ret:
            print("无法获取图像")
            break

        fps += 1
        if time.time() - fps_timer >= 1.0:
            fps_last = fps
            fps = 0
            fps_timer = time.time()
        print(f"FPS: {fps_last}")
        
        # 检测靶子
        binary, board = detector.detect(frame)
        result = detector.draw_boards(frame)
        
        # ===== 关键：在这里应用激光补偿 =====
        if board is not None and pnp.position is not None:
            # 获取相机位姿
            camera_pos = pnp.position
            rvec = pnp.rvec
            
            # 计算激光笔目标位姿
            laser_pos, laser_rot = laser.compensate(camera_pos, rvec)
            
            # 显示激光目标
            cv2.putText(result, f"Laser Target: ({laser_pos[0]:.2f}, {laser_pos[1]:.2f})", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # TODO: 发送 laser_pos 给云台控制系统
            # servo.set_target(laser_pos)
        
        # 显示
        cv2.imshow("Result", result)
        cv2.imshow("Binary", binary)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()