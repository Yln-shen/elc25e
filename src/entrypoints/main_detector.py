# src/entrypoints/main_detector.py
import cv2
import time
from src.vision.camera import Camera
from src.vision.detector import Detector
from src.vision.pnp import PNPSolver
from src.control.laser import LaserCompensator
from src.utils.decorators import measure_fps

@measure_fps  # ← 装饰器自动统计帧率
def run_loop(cam, detector, pnp, laser):
    """单帧处理逻辑，返回 (继续运行标志, 处理后的图像)"""
    ret, frame = cam.read()
    if not ret:
        return False, None
    
    binary, board = detector.detect(frame)
    result = detector.draw_boards(frame)
    
    # 在画面上显示 FPS
    cv2.putText(result, f"FPS: {run_loop.fps:.1f}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    if board is not None and pnp.position is not None:
        camera_pos = pnp.position
        rvec = pnp.rvec
        laser_pos, laser_rot = laser.compensate(camera_pos, rvec)
        cv2.putText(result, f"Laser Target: ({laser_pos[0]:.2f}, {laser_pos[1]:.2f})", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
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

    # ===== 2. 主循环 =====
    while True:
        running, _ = run_loop(cam, detector, pnp, laser)
        if not running:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()