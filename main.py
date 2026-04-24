import cv2
import numpy as np
import time
from detector import Detector,Laser # 导入detector.py
from tracker import Tracker # 导入tracker.py
from camera import Camera # 导入camera.py
from ser import Serial # 导入ser.py
#from Kalman import KalmanFilter # 导入Kalman.py

def main():
    # ========== 初始化配置 ==========
    # 摄像头
    try:
        cam = Camera(index=0)
    except Exception as e:
        print(f"摄像头初始化失败: {e}，尝试默认摄像头...")
        cam = Camera(index=0)

    # 激光
    laser = Laser(width_deviation=0, height_deviation=50)

    # 检测器
    detector = Detector(
        rectangle_max_area=60000,
        rectangle_min_area=1000,
        kernel=(5, 5),
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
    capture_count = 0

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
            closing = detector.process(frame)
            result = detector.draw_boards(frame, show_coords=True)

            # 5. 跟踪板子并计算偏航/俯仰角
            laser_center = detector.laser_center  # 可能为 None
            
            # 修复：只有检测到板子且有激光中心时才跟踪
            if laser_center is not None:
                yaw, pitch = tracker.track(laser_center)
            else:
                yaw, pitch = tracker.track(None)

            # 6. 发送角度到串口 + 终端输出
            if tracker.if_find and laser_center is not None:
                # 有跟踪目标且有激光中心坐标
                serial_port.send_data(yaw=yaw, pitch=pitch)
                # 终端输出
                if abs(yaw) > 0.01 or abs(pitch) > 0.01:
                    print(f"\r板子坐标: ({laser_center[0]:>7.1f}, {laser_center[1]:>7.1f})  "
                          f"偏航: {yaw:>6.1f}°  俯仰: {pitch:>6.1f}°  FPS: {fps_last}", end="")
                else:
                    print(f"\r板子坐标: ({laser_center[0]:>7.1f}, {laser_center[1]:>7.1f})  "
                          f"已对准中心  FPS: {fps_last}", end="")
            
            elif tracker.if_find and laser_center is None:
                # 卡尔曼预测状态，没有实际检测到
                print(f"\r预测跟踪中...  偏航: {yaw:>6.1f}°  俯仰: {pitch:>6.1f}°  FPS: {fps_display}", end="")
            
            elif board is not None and board.laser_center is not None:
                # 检测到板子但跟踪未就绪
                print(f"\r板子坐标: ({laser_center[0]:>7.1f}, {laser_center[1]:>7.1f})  "
                      f"等待跟踪...  FPS: {fps_last}", end="")
            
            else:
                # 未检测到板子
                print(f"\r未检测到板子  FPS: {fps_last}", end="")

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
            cv2.imshow('Mask', closing)
            cv2.imshow('Detection', result)

            # 9. 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n\n退出程序")
                break
            elif key == ord('s'):
                capture_count += 1
                filename = f"capture_{capture_count:04d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"\n已保存截图: {filename}")

    except KeyboardInterrupt:
        print("\n\n程序被中断")

    finally:
        # 释放资源
        cam.cam.release()
        serial_port.close()
        cv2.destroyAllWindows()
        print("资源已释放")
if __name__ == "__main__":
    main()

