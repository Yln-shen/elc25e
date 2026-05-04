import cv2
import numpy as np
import time
import sys
from model import Detector, Laser, Tracker, Camera, Serial,EmmMotor,SysParams


def main():
    # ========== 初始化配置 ==========
    # 摄像头
    cam = Camera(index=2)
    
    # 激光
    laser = Laser(width_deviation=0, height_deviation=50)
    
    # 检测器
    detector = Detector(
        rectangle_max_area=60000,
        rectangle_min_area=1000,
        laser=laser
    )
    
    # 跟踪器
    tracker = Tracker(
        vfov=100,
        img_width=640,
        use_kf=True,
        frame_add=30
    )
    
    # ========== 串口和电机初始化（移到循环外）==========
    SERIAL_PORT_PITCH = '/dev/ttyACM0' 
    SERIAL_PORT_YAW = '/dev/ttyACM1'
    BAUDRATE = 115200
    MOTOR_ID_PITCH = 1
    MOTOR_ID_YAW = 2
    
    motor_pitch = EmmMotor(
        port=SERIAL_PORT_PITCH,
        baudrate=BAUDRATE,
        timeout=1,
        motor_id=MOTOR_ID_PITCH,
        pulse=3200
    )
    
    motor_yaw = EmmMotor(
        port=SERIAL_PORT_YAW,
        baudrate=BAUDRATE,
        timeout=1,
        motor_id=MOTOR_ID_YAW,
        pulse=12800
    )
    
    # 版本验证和使能（只执行一次）
    ver_data = motor_pitch.emm_v5_read_sys_params(s=SysParams.S_VER)
    ver_data = motor_yaw.emm_v5_read_sys_params(s=SysParams.S_VER)
    
    motor_pitch.emm_v5_en_control(state=True)
    motor_yaw.emm_v5_en_control(state=True)
    
    # 帧计数
    fps = 0
    fps_last = 0
    fps_timer = time.time()
    last_print_lines = 1
    
    print("按 'q' 退出")
    
    # ========== 主循环 ==========
    try:
        while True:
            # 1. 获取图像
            ret, frame = cam.read()
            if not ret:
                print("无法获取图像")
                break
            
            # 2. 计算FPS
            fps += 1
            if time.time() - fps_timer >= 1.0:
                fps_last = fps
                fps = 0
                fps_timer = time.time()
            
            # 3-5. 检测和跟踪
            board = detector.detect(frame)
            binary = detector.process(frame)
            result = detector.draw_boards(frame, show_coords=True)
            
            laser_center = detector.laser_center
            if laser_center is not None:
                yaw, pitch = tracker.track(laser_center)
            else:
                yaw, pitch = tracker.track(None)
            
            # 6. 控制电机（只在需要时发送指令）
            if tracker.if_find and abs(yaw) > 0.01 or abs(pitch) > 0.01:
                motor_pitch.emm_v5_move_to_angle(
                    angle_deg=pitch, vel_rpm=500, acc=100, abs_mode=False
                )
                motor_yaw.emm_v5_move_to_angle(
                    angle_deg=yaw, vel_rpm=500, acc=100, abs_mode=False
                )
                # 不要在这里用 time.sleep(2)！
            
            # 7-9. 显示和输出（保持原有逻辑）
            # ... 显示代码 ...
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    finally:
        # 清理资源
        motor_pitch.emm_v5_stop_now()
        motor_pitch.emm_v5_en_control(state=False)
        motor_yaw.emm_v5_stop_now()
        motor_yaw.emm_v5_en_control(state=False)
        cam.cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
