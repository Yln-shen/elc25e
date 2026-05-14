# main_motor.py - 完整优化版（防重启 + 滑条初始化修正 + 滑条值保存）

import cv2
import numpy as np
import time
import sys
import os
import gc
from model import Detector, Tracker, Camera, EmmMotor, SysParams, GPIO, PID


def main():
    # ========== 初始化配置 ==========
    try:
        cam = Camera(index=0)
        cam.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception as e:
        print(f"摄像头初始化失败: {e}")
        return

    detector = Detector(
        rectangle_max_area=1300000,
        rectangle_min_area=1000,
        use_pnp=True
    )

    tracker = Tracker(
        vfov=51.6,
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

    motor_pitch = None
    motor_yaw = None

    try:
        motor_pitch = EmmMotor(
            port=SERIAL_PORT_PITCH,
            baudrate=BAUDRATE,
            timeout=1,
            motor_id=MOTOR_ID_PITCH,
            pulse=3200
        )
        print("串口1连接成功！")
    except Exception as e:
        print(f"串口1连接失败: {e}")
        gpio.release()
        cam.cam.release()
        return

    try:
        motor_yaw = EmmMotor(
            port=SERIAL_PORT_YAW,
            baudrate=BAUDRATE,
            timeout=1,
            motor_id=MOTOR_ID_YAW,
            pulse=12800
        )
        print("串口2连接成功！")
    except Exception as e:
        print(f"串口2连接失败: {e}")
        motor_pitch.emm_v5_en_control(state=False)
        gpio.release()
        cam.cam.release()
        return

    # 测试通信
    print("\n--- 测试: 读取系统版本 ---")

    try:
        ver_data = motor_pitch.emm_v5_read_sys_params(s=SysParams.S_VER, timeout=0.5)
        ver_data = motor_yaw.emm_v5_read_sys_params(s=SysParams.S_VER, timeout=0.5)
        if ver_data and len(ver_data) >= 4:
            print(f"版本响应数据 (Hex): {ver_data.hex()}")
        else:
            print("读取版本失败或无响应")
    except Exception as e:
        print(f"通信测试失败: {e}")

    # 使能电机
    motor_pitch.emm_v5_en_control(state=True)
    motor_yaw.emm_v5_en_control(state=True)
    # ===========================================

    fps = 0
    fps_last = 0
    fps_timer = time.time()
    last_print_lines = 1
    frame_count = 0

    # ===== PD控制变量 =====
    Kp_pitch = 0.05
    Kd_pitch = 0.1
    Kp_yaw = 0.1
    Kd_yaw = 0.1
    last_pitch_err = 0.0
    last_yaw_err = 0.0

    val_pitch = 2
    val_yaw = 6

    # ===== 读取上次保存的滑条值 =====
    OFFSET_FILE = "slider_offset.txt"
    if os.path.exists(OFFSET_FILE):
        try:
            with open(OFFSET_FILE, 'r') as f:
                saved = f.read().strip().split(',')
                YAW_OFFSET_INIT = float(saved[0])
                PITCH_OFFSET_INIT = float(saved[1])
            print(f"读取上次滑条值: YAW={YAW_OFFSET_INIT:.2f}°, PITCH={PITCH_OFFSET_INIT:.2f}°")
        except:
            YAW_OFFSET_INIT = -0.5
            PITCH_OFFSET_INIT = 0.6
            print("读取滑条文件失败，使用默认值")
    else:
        YAW_OFFSET_INIT = -0.5
        PITCH_OFFSET_INIT = 0.6
        print("首次运行，使用默认滑条值")

    # ===== 滑条初始化 =====
    OFFSET_RANGE = 5.0        # ±5度
    SLIDER_MAX = 200

    yaw_slider_init = int((YAW_OFFSET_INIT + OFFSET_RANGE) / (2 * OFFSET_RANGE) * SLIDER_MAX)
    pitch_slider_init = int((PITCH_OFFSET_INIT + OFFSET_RANGE) / (2 * OFFSET_RANGE) * SLIDER_MAX)

    cv2.namedWindow('Control', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Control', 400, 150)
    cv2.createTrackbar('YAW_OFFSET', 'Control', yaw_slider_init, SLIDER_MAX, lambda x: None)
    cv2.createTrackbar('PITCH_OFFSET', 'Control', pitch_slider_init, SLIDER_MAX, lambda x: None)

    print(f"滑条初始: YAW={YAW_OFFSET_INIT:.2f}° (滑条:{yaw_slider_init}), PITCH={PITCH_OFFSET_INIT:.2f}° (滑条:{pitch_slider_init})")
    print(f"滑条范围: ±{OFFSET_RANGE}°")
    print("按 'q' 退出")

    error_count = 0
    max_errors = 10

    # 初始化滑条变量，避免退出时未定义
    YAW_OFFSET = YAW_OFFSET_INIT
    PITCH_OFFSET = PITCH_OFFSET_INIT

    try:
        while True:
            try:
                ret, frame = cam.read()
                if not ret:
                    print("无法获取图像，等待...")
                    time.sleep(0.1)
                    error_count += 1
                    if error_count > max_errors:
                        print("连续读取失败，退出")
                        break
                    continue

                error_count = 0
                frame_count += 1

                fps += 1
                if time.time() - fps_timer >= 1.0:
                    fps_last = fps
                    fps = 0
                    fps_timer = time.time()

                # 检测
                try:
                    binary, board = detector.detect(frame)
                    result = detector.draw_boards(frame, show_coords=True)
                    camera_center_offset = detector.camera_center_offset
                except Exception as e:
                    print(f"检测失败: {e}")
                    continue

                # 读取滑条
                YAW_OFFSET = (cv2.getTrackbarPos('YAW_OFFSET', 'Control') / SLIDER_MAX) * (2 * OFFSET_RANGE) - OFFSET_RANGE
                PITCH_OFFSET = (cv2.getTrackbarPos('PITCH_OFFSET', 'Control') / SLIDER_MAX) * (2 * OFFSET_RANGE) - OFFSET_RANGE

                # 优先使用PNP
                use_pnp = (detector.pnp.yaw is not None and detector.pnp.pitch is not None)

                # 优先使用PNP
                use_pnp = (detector.pnp.yaw is not None and detector.pnp.pitch is not None)

                if use_pnp:
                    yaw = detector.pnp.yaw - YAW_OFFSET
                    pitch = detector.pnp.pitch - PITCH_OFFSET
                    distance = detector.pnp.distance
                    center_error = detector.pnp.center_error
                    if camera_center_offset is not None:
                        pixel_yaw, pixel_pitch = tracker.pixel_to_yaw_pitch(camera_center_offset)
                    else:
                        pixel_yaw, pixel_pitch = 0.0, 0.0
                elif camera_center_offset is not None:
                    yaw, pitch = tracker.track(camera_center_offset)
                    distance = None
                    center_error = None
                    pixel_yaw, pixel_pitch = yaw, pitch
                else:
                    yaw, pitch = 0.0, 0.0
                    distance = None
                    center_error = None
                    pixel_yaw, pixel_pitch = 0.0, 0.0

                # GPIO和电机控制
                if use_pnp or camera_center_offset is not None:
                    if not hasattr(main, 'gpio_state'):
                        main.gpio_state = False

                    target_near = (abs(yaw) < 0.5 and abs(pitch) < 0.5)
                    if target_near != main.gpio_state:
                        try:
                            if target_near:
                                gpio.on()
                            else:
                                gpio.off()
                            main.gpio_state = target_near
                        except Exception as e:
                            print(f"GPIO操作失败: {e}")

                    if use_pnp or tracker.if_find:
                        min_angle = 0.3

                        # ===== Pitch =====
                        pitch_abs = abs(pitch)
                        if pitch_abs > min_angle:
                            if pitch_abs < 2.0:
                                pitch_cmd = 0.3 if pitch > 0 else -0.3
                            else:
                                pitch_cmd = Kp_pitch * pitch + Kd_pitch * (pitch - last_pitch_err)
                                pitch_cmd = max(-5.0, min(5.0, pitch_cmd))
                            last_pitch_err = pitch
                            try:
                                motor_pitch.emm_v5_move_to_angle(
                                    angle_deg=pitch_cmd, vel_rpm=val_pitch, acc=0, abs_mode=False)
                            except Exception as e:
                                print(f"Pitch电机命令失败: {e}")
                        else:
                            pitch_cmd = 0.0

                        # ===== Yaw =====
                        yaw_abs = abs(yaw)
                        if yaw_abs > min_angle:
                            if yaw_abs < 2.0:
                                yaw_cmd = 0.3 if yaw > 0 else -0.3
                            else:
                                yaw_cmd = Kp_yaw * yaw + Kd_yaw * (yaw - last_yaw_err)
                                yaw_cmd = max(-5.0, min(5.0, yaw_cmd))
                            last_yaw_err = yaw
                            try:
                                motor_yaw.emm_v5_move_to_angle(
                                    angle_deg=yaw_cmd, vel_rpm=val_yaw, acc=0, abs_mode=False)
                            except Exception as e:
                                print(f"Yaw电机命令失败: {e}")
                        else:
                            yaw_cmd = 0.0

                        if frame_count % 10 == 0:
                            try:
                                ver_data = motor_pitch.emm_v5_read_sys_params(s=SysParams.S_CPOS, timeout=0.1)
                            except:
                                pass
                    else:
                        yaw_cmd, pitch_cmd = 0.0, 0.0
                        try:
                            motor_pitch.emm_v5_stop_now()
                            motor_yaw.emm_v5_stop_now()
                        except:
                            pass
                        last_pitch_err = 0.0
                        last_yaw_err = 0.0
                else:
                    yaw, pitch = 0.0, 0.0
                    yaw_cmd, pitch_cmd = 0.0, 0.0
                    distance = None
                    center_error = None
                    last_pitch_err = 0.0
                    last_yaw_err = 0.0
                    try:
                        motor_pitch.emm_v5_vel_control(dir=0, vel=0, acc=0)
                        motor_yaw.emm_v5_vel_control(dir=0, vel=0, acc=0)
                        time.sleep(0.01)
                        motor_pitch.emm_v5_stop_now()
                        motor_yaw.emm_v5_stop_now()
                    except:
                        pass
                    tracker.if_find = False
                    tracker.predict = False
                    tracker.lost = 0
                    tracker.kf_position = None

                    # ===== 终端显示 =====
                    if frame_count % 5 == 0:
                        sys.stdout.write(f"\033[2A")  # ← 先固定上移2行
                        sys.stdout.write("\033[J")     # ← 清除下面所有内容

                        if use_pnp or camera_center_offset is not None:
                            if use_pnp:
                                cx = camera_center_offset[0] if camera_center_offset is not None else 0
                                cy = camera_center_offset[1] if camera_center_offset is not None else 0
                                print(f"偏移:({cx:.0f},{cy:.0f})px 距离:{distance:.2f}m  "
                                    f"Y:{yaw:.1f}° P:{pitch:.1f}°  "
                                    f"滑条:Y={YAW_OFFSET:.1f}° P={PITCH_OFFSET:.1f}°  "
                                    f"FPS:{fps_last}")
                                yaw_remain = abs(yaw)
                                pitch_remain = abs(pitch)
                                if yaw_remain < 0.5 and pitch_remain < 0.5:
                                    print("✓ 已对准!")
                                else:
                                    print(f"距目标: Y={yaw_remain:.1f}° P={pitch_remain:.1f}°  |  "
                                        f"电机cmd: Y={yaw_cmd:.2f}° P={pitch_cmd:.2f}°")
                                last_print_lines = 2
                            else:
                                print(f"像素模式  Y:{yaw:.1f}° P:{pitch:.1f}°  FPS:{fps_last}")
                                last_print_lines = 1
                        elif detector.relative_board is not None:
                            print(f"检测到靶子但PNP未就绪  FPS:{fps_last}")
                            last_print_lines = 1
                        else:
                            print(f"无目标  FPS:{fps_last}")
                            last_print_lines = 1

                        sys.stdout.flush()

                # ===== 图像显示 =====
                cv2.putText(result, f"FPS: {fps_last}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if use_pnp:
                    cv2.putText(result, "PNP", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    if distance is not None:
                        cv2.putText(result, f"Dist:{distance:.2f}m", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                cv2.imshow('Detection', result)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    print("\n退出程序")
                    break

                # 定期释放内存
                if frame_count % 100 == 0:
                    gc.collect()

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"循环异常: {e}")
                error_count += 1
                if error_count > max_errors:
                    print("连续异常过多，退出")
                    break
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n程序被中断")

    # ===== 保存滑条值 =====
    try:
        with open(OFFSET_FILE, 'w') as f:
            f.write(f"{YAW_OFFSET:.2f},{PITCH_OFFSET:.2f}")
        print(f"滑条值已保存: YAW={YAW_OFFSET:.2f}°, PITCH={PITCH_OFFSET:.2f}°")
    except Exception as e:
        print(f"保存滑条值失败: {e}")

    finally:
        print("正在释放资源...")
        try:
            motor_pitch.emm_v5_en_control(state=False)
        except:
            pass
        try:
            motor_yaw.emm_v5_en_control(state=False)
        except:
            pass
        try:
            cam.cam.release()
        except:
            pass
        try:
            gpio.off()
        except:
            pass
        try:
            gpio.release()
        except:
            pass
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        print("资源已释放")


if __name__ == "__main__":
    main()