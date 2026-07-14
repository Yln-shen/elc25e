# src/entrypoints/main_detector.py
import cv2
import time
import numpy as np
from datetime import datetime
from src.vision.camera import Camera
from src.vision.detector import Detector
from src.vision.pnp import PNPSolver
from src.vision.tracker import Tracker
from src.control.laser import LaserCompensator
from src.utils.decorators import measure_fps


# ===== 全局状态跟踪 =====
_last_status = None
_last_yaw = None
_last_pitch = None


def print_status_change(status, yaw=None, pitch=None, dist=None, tvec=None):
    """在状态变化时打印详细信息"""
    global _last_status, _last_yaw, _last_pitch
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if status != _last_status:
        print("=" * 50)
        print(f"[{timestamp}] 状态变化: {_last_status} → {status}")
        _last_status = status
    
    if status == "TRACKING" and yaw is not None and pitch is not None:
        if _last_yaw is None or abs(yaw - _last_yaw) > 0.1 or abs(pitch - _last_pitch) > 0.1:
            print("=" * 50)
            print(f"[{timestamp}] 云台指令 (激光补偿+滤波后)")
            print(f"  Yaw:   {yaw:.2f}°")
            print(f"  Pitch: {pitch:.2f}°")
            if dist is not None:
                print(f"  Dist:  {dist:.3f} m")
            if tvec is not None:
                print(f"  tvec:  ({tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f}) m")
            print("=" * 50)
            _last_yaw = yaw
            _last_pitch = pitch


@measure_fps
def run_loop(cam, detector, pnp, laser, tracker):
    ret, frame = cam.read()
    if not ret:
        return False, None
    
    binary, board = detector.detect(frame)
    result = detector.draw_boards(frame)
    
    h, w = result.shape[:2]
    
    # ===== 1. 显示 FPS =====
    cv2.putText(result, f"FPS: {run_loop.fps:.1f}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # ===== 2. PnP 解算 =====
    laser_pos = None
    tvec_raw = None
    
    if board is not None and len(board.points) == 4:
        image_pts = np.array(board.points, dtype=np.float32)
        pnp_result = pnp.solve(image_pts)
        
        if pnp_result['success']:
            tvec_raw = pnp_result['tvec']
            rvec_raw = pnp_result['rvec']
            
            # ===== 3. 激光补偿 =====
            # laser_pos = 靶心在激光笔坐标系下的位置
            # 这个位置就是激光笔应该瞄准的方向！
            laser_pos, _ = laser.compensate(tvec_raw, rvec_raw)
    
    # ===== 4. 滤波器对激光补偿后的位置进行平滑 =====
    filtered_xyz = tracker.track(laser_pos)
    
    # ===== 5. 从滤波后的位置计算云台角度 =====
    yaw_filt = None
    pitch_filt = None
    dist_filt = None
    
    if filtered_xyz is not None:
        yaw_filt, pitch_filt = tracker.get_yaw_pitch()
        x_f, y_f, z_f = filtered_xyz
        dist_filt = np.sqrt(x_f**2 + y_f**2 + z_f**2)
        
        # ===== 6. 显示滤波后的结果 =====
        # cv2.putText(result, f"Filt Y/P: ({yaw_filt:.2f}, {pitch_filt:.2f}) deg", 
        #            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # cv2.putText(result, f"Filt XYZ: ({x_f:.3f}, {y_f:.3f}, {z_f:.3f}) m", 
        #            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # ===== 7. 发送角度到云台 =====
    if yaw_filt is not None and pitch_filt is not None:
        # laser_pos 已经包含了激光补偿
        # yaw_filt/pitch_filt 是从滤波后的 laser_pos 计算的
        # 所以云台转过去，激光笔就能打在靶心上！
        # TODO: servo.set_target(yaw_filt, pitch_filt)
        
        cv2.putText(result, f"SEND: ({yaw_filt:.2f}, {pitch_filt:.2f}) deg", 
                   (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # ===== 8. 状态变化时打印 =====
    if tracker.if_find:
        print_status_change("TRACKING", yaw_filt, pitch_filt, dist_filt, tvec_raw)
    else:
        print_status_change("LOST")
    
    # ===== 9. 绘制Tracker调试信息 =====
    result = tracker.draw_debug(result)
    
    cv2.imshow("Binary", binary)
    cv2.imshow("Result", result)
    
    return True, result


def main():
    try:
        cam = Camera(index=3, width=640, height=480, fps=120)
    except Exception as e:
        print(f"摄像头初始化失败: {e}，尝试默认摄像头...")
        cam = Camera(index=1)
    
    pnp = PNPSolver(target_width=0.262, target_height=0.174)
    detector = Detector(rectangle_min_area=1000, rectangle_max_area=50000, pnp_solver=pnp)
    laser = LaserCompensator(params_file="laser_params.json")
    
    tracker = Tracker(
        use_kf=True,
        frame_add=35,
        qx=4.0, qy=4.0, qz=3.0, r=0.01
    )
    tracker.set_projection_params(
        K=np.array([
            [581.516, 0.0, 385.208],
            [0.0, 582.147, 181.477],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32),
        img_width=640,
        img_height=480
    )
    
    print("\n" + "=" * 60)
    print("按 'q' 退出")
    
    while True:
        running, _ = run_loop(cam, detector, pnp, laser, tracker)
        if not running:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.cam.release()
    cv2.destroyAllWindows()
    print("\n程序已退出")


if __name__ == "__main__":
    main()