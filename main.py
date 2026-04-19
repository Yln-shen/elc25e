import cv2
import numpy as np
import time
from detector import Detector # 导入detector.py
from tracker import Tracker # 导入tracker.py
from camera import Camera # 导入camera.py
from ser import Serial # 导入ser.py
from Kalman import KalmanFilter # 导入Kalman.py

def main():
    """
    主函数：只做调度
    
    数据流：
        1. 读帧
        2. 生成掩膜
        3. 检测板子 → 存入 detector.current_board
        4. 读取 laser_center
        5. 转换为角度
        6. 发送到串口
        7. 显示
    """
    # ========== 初始化 ==========
    # 黑色HSV范围（需要根据实际环境调整）
    black_range = ([0, 0, 0], [180, 255, 70])
    
    # 创建检测器
    detector = Detector(black_range, 3000, (5,5))
    
    # 设置激光偏移（需要标定！）
    # 假设激光在图像中心右侧5像素，下侧10像素
    detector.laser.width_deviation = 0    # 激光偏右5像素
    detector.laser.height_deviation = 50   # 激光偏下10像素
    
    # 创建角度计算器
    angle_tracker = Tracker(vfov=100, img_width=640)
    
    # 初始化摄像头
    try:
        cam = Camera(index=0)
    except Exception as e:
        print(f"摄像头初始化失败: {e}，尝试默认摄像头...")
        cam = Camera(index=0)
    
    # 串口（根据实际模块取消注释）
    import ser
    ser_port = ser.Serial(port='/dev/ttyUSB0', baudrate=115200)
    # ser_port = None
    
    print("开始检测...")
    print("  坐标系：原点 = 激光笔中心")
    print("  X正方向：向右，Y正方向：向下")
    print("  按 'q' 退出，按 's' 保存")
    
    frame_count = 0
    last_send_time = 0
    send_interval = 0.05  # 20Hz
    
    # ========== 主循环 ==========
    while True:
        # 1. 获取图像
        ret, frame = cam.read()
        if not ret:
            print("无法获取图像")
            break
        
        frame_count += 1
        
        # 2. 生成掩膜
        # mask = detector.process(frame)
        
        # # 3. 检测板子（结果存入 detector.current_board）
        # detector.update_board(mask, frame)
        detector.detect(frame)
        # 4. 获取以激光笔为原点的板子坐标
        laser_center = detector.get_laser_center()
        
        # 5. 计算角度并发送
        current_time = time.time()
        if laser_center is not None and (current_time - last_send_time) >= send_interval:
            yaw, pitch = angle_tracker.pixel_to_yaw_pitch(laser_center)
            
            if ser_port:
                try:
                    # ser_port.send_data(yaw, pitch)
                    print(f"[串口] Yaw={yaw:.2f}°, Pitch={pitch:.2f}°")
                except Exception as e:
                    print(f"发送失败: {e}")
            else:
                print(f"[模拟] Yaw={yaw:.2f}°, Pitch={pitch:.2f}°")
                print(f"       板子坐标: ({laser_center[0]:.1f}, {laser_center[1]:.1f}) 像素")
            
            last_send_time = current_time
        
        # 6. 绘制并显示
        result = detector.draw_boards(frame, show_coords=True)
        
        # 在画面上叠加角度信息
        if laser_center is not None:
            yaw, pitch = angle_tracker.pixel_to_yaw_pitch(laser_center)
            cv2.putText(result, f"Yaw: {yaw:.2f} deg", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(result, f"Pitch: {pitch:.2f} deg", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 显示窗口
        closing = detector.process(frame)
        cv2.imshow('Mask', closing)
        cv2.imshow('Detection', result)
        
        # 7. 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"已保存: {filename}")
    
    # ========== 清理 ==========
    if ser_port:
        ser_port.close()
    cam.release()
    cv2.destroyAllWindows()
    print("程序结束")


if __name__ == "__main__":
    main()

