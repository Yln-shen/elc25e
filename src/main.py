import cv2
import numpy as np
import time
import sys
from model import Detector, Laser, Tracker, Camera, Serial

def main():
    # ========== 初始化配置 ==========
    # 摄像头
    try:
        cam = Camera(index=0)
    except Exception as e:
        print(f"摄像头初始化失败: {e}，尝试默认摄像头...")
        cam = Camera(index=1)

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

    # 串口通信
    serial_port = Serial(
        port='/dev/ttyACM0',
        baudrate=115200,
        timeout=1,
        write_timeout=1
    )

    # 帧计数
    fps = 0
    fps_last = 0
    fps_timer = time.time()
    
    # 用于两行刷新
    last_print_lines = 1  # 记录上次打印了几行

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

            # 3. 检测板子
            board = detector.detect(frame)

            # 4. 获取掩膜和绘制结果
            binary = detector.process(frame)
            result = detector.draw_boards(frame, show_coords=True)

            # 5. 跟踪板子并计算偏航/俯仰角
            laser_center = detector.laser_center
            
            if laser_center is not None:
                yaw, pitch = tracker.track(laser_center)
            else:
                yaw, pitch = tracker.track(None)

            # 6. 清除之前的输出 + 终端输出
            # 光标上移 last_print_lines 行，清除旧内容
            sys.stdout.write(f"\033[{last_print_lines}A")  # 上移
            sys.stdout.write("\033[J")  # 清除到屏幕底部
            
            if tracker.if_find and laser_center is not None:
                # 有跟踪目标且有激光中心坐标
                serial_port.send_data(yaw=yaw, pitch=pitch)
                print(f"板子坐标: ({laser_center[0]:>7.1f}, {laser_center[1]:>7.1f})")
                if abs(yaw) > 0.01 or abs(pitch) > 0.01:
                    print(f"偏航: {yaw:>6.1f}°  俯仰: {pitch:>6.1f}°  FPS: {fps_last}")
                else:
                    print(f"FPS: {fps_last}")
                last_print_lines = 2

            
            elif tracker.if_find and laser_center is None:
                #有跟踪目标、无激光中心坐标
                print(f"预测:偏航: {yaw:>6.1f}°  俯仰: {pitch:>6.1f}°  FPS: {fps_last}")
                last_print_lines = 1
            
            # ========== 改动1：DEBUG信息改为用清除+print，而不是\r ==========
            if tracker.kf_position is not None:
                kf_x = int(detector.laser_pixel[0] + tracker.kf_position[0])
                kf_y = int(detector.laser_pixel[1] + tracker.kf_position[1])
                # 原来: print(f"DEBUG: ...", end="\r")  ← 会冲突
                # 改为和上面一样的清除逻辑，但追加行数
                sys.stdout.write(f"\033[1A")  # 上移1行（覆盖之前的DEBUG行）
                sys.stdout.write("\033[J")   # 清除

                #如果 kf_pos 值很大，说明预测偏移大，可能目标移动快或检测不稳定
                # 对比 pixel 和 laser_pixel，可以看出卡尔曼滤波器对位置的修正幅度
                #当 laser_center 为 None 时（丢失检测），纯靠 kf_pos 预测来维持跟踪
                print(f"DEBUG: kf_pos=({tracker.kf_position[0]:.1f}, {tracker.kf_position[1]:.1f}), " #偏移量
                    f"pixel=({kf_x}, {kf_y}), "            #卡尔曼滤波后的板子中心坐标
                    f"laser_pixel={detector.laser_pixel}")
                last_print_lines += 1  # 多加了一行DEBUG
                
                result = tracker.draw_kf(result, detector.laser_pixel)
            # ========== 改动1结束 ==========
            else:
                # ========== 改动2：同样处理 ==========
                sys.stdout.write(f"\033[1A")  # 上移1行
                sys.stdout.write("\033[J")   # 清除
                print(f"DEBUG: kf_position is None")
                last_print_lines += 1
                # ========== 改动2结束 ==========
            
            sys.stdout.flush()  # 立即刷新输出

            # 7. 显示FPS和状态
            cv2.putText(result, f"FPS: {fps_last}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示跟踪状态
            if tracker.if_find:
                status_text = "Track: OK" if laser_center is not None else "Track: PREDICT"
                status_color = (0, 255, 0) if laser_center is not None else (0, 255, 255)
            else:
                status_text = "Track: LOST"
                status_color = (0, 0, 255)
            
            cv2.putText(result, status_text, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            # 显示角度信息
            if tracker.if_find:
                angle_text = f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}"
                cv2.putText(result, angle_text, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # 8. 显示图像
            cv2.imshow('Mask', binary)
            cv2.imshow('Detection', result)

            # 9. 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n\n退出程序")
                break
            elif key == ord('s'):
                last_print_lines = 1  # 重置行数

    except KeyboardInterrupt:
        print("\n\n程序被中断")

    finally:
        cam.cam.release()
        serial_port.close()
        cv2.destroyAllWindows()
        print("资源已释放")

if __name__ == "__main__":
    main()
