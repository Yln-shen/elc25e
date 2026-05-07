import cv2
import numpy as np
import time
import sys
from model import Detector, Laser, Tracker, Camera, EmmMotor, SysParams, GPIO, PID



def main():
    # ========== 初始化配置 ==========
    try:
        cam = Camera(index=0)
    except Exception as e:
        print(f"摄像头初始化失败: {e}，尝试默认摄像头...")
        cam = Camera(index=0)

    laser = Laser(
        width_deviation=15,
        height_deviation=5
    )
    
    #检测矩形
    detector = Detector(
        rectangle_max_area=130000,
        rectangle_min_area=10000,
        laser=laser
    )

    tracker = Tracker(
        vfov=100,
        img_width=640,
        use_kf=False,
        frame_add=5
    )

    gpio = GPIO(
        chip_path='/dev/gpiochip4', 
        line_offset=18, 
        consumer='laser'
    )

    # ===========================================
    # 串口和电机初始化
    # ===========================================
    SERIAL_PORT_PITCH = '/dev/ttyACM0'
    SERIAL_PORT_YAW   = '/dev/ttyS4'
    BAUDRATE = 115200
    MOTOR_ID_PITCH = 1
    MOTOR_ID_YAW = 2

    print(f"正在尝试连接串口: {SERIAL_PORT_PITCH} ...")
    print(f"正在尝试连接串口: {SERIAL_PORT_YAW} ...")

    motor_pitch = EmmMotor(
        port=SERIAL_PORT_PITCH,
        baudrate=BAUDRATE,
        timeout=1,
        motor_id=MOTOR_ID_PITCH,
        pulse=3200
    )
    print("串口1连接成功！")

    motor_yaw = EmmMotor(
        port=SERIAL_PORT_YAW,
        baudrate=BAUDRATE,
        timeout=1,
        motor_id=MOTOR_ID_YAW,
        pulse=12800
    )
    print("串口2连接成功！")

    # 测试通信
    print("\n--- 测试: 读取系统版本 ---")
    ver_data = motor_pitch.emm_v5_read_sys_params(s=SysParams.S_VER)
    ver_data = motor_yaw.emm_v5_read_sys_params(s=SysParams.S_VER)
    if ver_data and len(ver_data) >= 4:
        print(f"版本响应数据 (Hex): {ver_data.hex()}")
    else:
        print("读取版本失败或无响应，请检查接线和ID。")
        sys.exit(1)

    # 使能电机
    motor_pitch.emm_v5_en_control(state=True)
    motor_yaw.emm_v5_en_control(state=True)
    # ===========================================

    fps = 0
    fps_last = 0
    fps_timer = time.time()
    last_print_lines = 1

    # ===== PD控制变量 =====
    Kp = 0.08   # 比例系数
    Kd = 0.04   # 微分系数
    last_pitch_err = 0.0
    last_yaw_err = 0.0

    print("按 'q' 退出")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("无法获取图像")
                break

            fps += 1
            if time.time() - fps_timer >= 1.0:
                fps_last = fps
                fps = 0
                fps_timer = time.time()

            binary, board = detector.detect(frame)
            result = detector.draw_boards(frame, show_coords=True)
            laser_center = detector.laser_center

            if laser_center is not None:
                yaw, pitch = tracker.track(laser_center)

                # GPIO 状态防抖，只在状态改变时才操作
                if not hasattr(main, 'gpio_state'):
                    main.gpio_state = False

                target_near = (abs(yaw) < 2 and abs(pitch) < 2)
                if target_near != main.gpio_state:
                    if target_near:
                        gpio.on()
                    else:
                        gpio.off()
                    main.gpio_state = target_near
                    
                if tracker.if_find:
                    # ===== PD控制 =====
                    # Pitch
                    pitch_err = pitch
                    pitch_diff = pitch_err - last_pitch_err
                    pitch_cmd = Kp * pitch_err + Kd * pitch_diff
                    last_pitch_err = pitch_err
                    
                    # Yaw
                    yaw_err = yaw
                    yaw_diff = yaw_err - last_yaw_err
                    pid_yaw = PID(Kp=0.04, Kd=0)
                    yaw_cmd = pid_yaw.compute(yaw_err, yaw_diff)
                    
                    # 最小角度阈值
                    min_angle = 0.3

                                        # 主循环中，只在必要时才读取电机状态
                    if fps % 10 == 0:  # 每30帧（约1秒）读取一次
                        ver_data = motor_pitch.emm_v5_read_sys_params(
                            s=SysParams.S_CPOS, 
                            timeout=0.003  # 3ms超时，更快
                        )
                    
                    if abs(pitch) > min_angle:
                        motor_pitch.emm_v5_move_to_angle(
                            angle_deg=pitch_cmd,
                            vel_rpm=2, 
                            acc=0, 
                            abs_mode=False
                        )
                    
                    if abs(yaw) > min_angle:
                        motor_yaw.emm_v5_move_to_angle(
                            angle_deg=yaw_cmd,
                            vel_rpm=2, 
                            acc=0, 
                            abs_mode=False
                        )
                else:
                    motor_pitch.emm_v5_stop_now()
                    motor_yaw.emm_v5_stop_now()
                    last_pitch_err = 0.0
                    last_yaw_err = 0.0
            else:
                yaw, pitch = 0.0, 0.0
                yaw_cmd, pitch_cmd = 0.0, 0.0
                last_pitch_err = 0.0
                last_yaw_err = 0.0
                
                motor_pitch.emm_v5_vel_control(dir=0, vel=0, acc=0)
                motor_yaw.emm_v5_vel_control(dir=0, vel=0, acc=0)
                time.sleep(0.01)
                motor_pitch.emm_v5_stop_now()
                motor_yaw.emm_v5_stop_now()
                time.sleep(0.01)
                
                tracker.if_find = False
                tracker.predict = False
                tracker.lost = 0
                tracker.kf_position = None

            # ========== 显示部分 ==========
            sys.stdout.write(f"\033[{last_print_lines}A")
            sys.stdout.write("\033[J")
            
            if laser_center is not None:
                print(f"板子坐标: ({laser_center[0]:>7.1f}, {laser_center[1]:>7.1f})")
                print(f"偏航: {yaw:>6.1f}° cmd: {yaw_cmd:>6.1f}°  俯仰: {pitch:>6.1f}° cmd: {pitch_cmd:>6.1f}°  FPS: {fps_last}")
                last_print_lines = 2
            else:
                print(f"无目标，电机已停止  FPS: {fps_last}")
                last_print_lines = 2 if tracker.kf_position is not None else 1

            if tracker.kf_position is not None:
                kf_x = int(detector.laser_pixel[0] + tracker.kf_position[0])
                kf_y = int(detector.laser_pixel[1] + tracker.kf_position[1])
                print(f"DEBUG: kf_pos=({tracker.kf_position[0]:.1f}, {tracker.kf_position[1]:.1f})")
                result = tracker.draw_kf(result, detector.laser_pixel)
                last_print_lines += 1
            else:
                print(f"DEBUG: kf_position is None")

            sys.stdout.flush()

            cv2.putText(result, f"FPS: {fps_last}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if laser_center is not None:
                status_text = "Track: OK"
                status_color = (0, 255, 0)
            else:
                status_text = "Track: LOST"
                status_color = (0, 0, 255)
            
            cv2.putText(result, status_text, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            if laser_center is not None:
                cv2.putText(result, f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow('Mask', binary)
            cv2.imshow('Detection', result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n\n退出程序")
                break
            elif key == ord('s'):
                last_print_lines = 1

    except KeyboardInterrupt:
        print("\n\n程序被中断")

    finally:
        motor_pitch.emm_v5_en_control(state=False)
        motor_yaw.emm_v5_en_control(state=False)
        cam.cam.release()
        gpio.off()
        gpio.release()
        cv2.destroyAllWindows()
        print("资源已释放")

if __name__ == "__main__":
    main()