import cv2
import numpy as np
import time
from detector import Detector, Laser
from tracker import Tracker
from camera import Camera
from ser import Serial
import matplotlib.pyplot as plt
from collections import deque

def main():
    """
    主函数：添加卡尔曼滤波效果可视化
    """
    # ========== 初始化 ==========
    black_range = ([0, 0, 0], [180, 255, 70])
    laser = Laser(width_deviation=0, height_deviation=50)
    detector = Detector(black_range, 3000, (5,5), laser=laser)
    
    # 创建两个 tracker：一个使用 KF，一个不使用
    tracker_with_kf = Tracker(vfov=100, img_width=640, use_kf=True, frame_add=35)
    tracker_without_kf = Tracker(vfov=100, img_width=640, use_kf=False, frame_add=35)
    
    # 初始化摄像头
    try:
        cam = Camera(index=2)
    except Exception as e:
        print(f"摄像头初始化失败: {e}，尝试默认摄像头...")
        cam = Camera(index=0)
    
    ser_port = Serial(port='/dev/ttyUSB0', baudrate=115200)
    
    print("开始检测...")
    print("  按 'q' 退出，按 's' 保存，按 'p' 显示/隐藏图表")
    
    # ========== 数据记录 ==========
    data_length = 200  # 保存最近200个数据点
    timestamps = deque(maxlen=data_length)
    raw_yaw = deque(maxlen=data_length)
    raw_pitch = deque(maxlen=data_length)
    kf_yaw = deque(maxlen=data_length)
    kf_pitch = deque(maxlen=data_length)
    
    frame_count = 0
    last_send_time = 0
    send_interval = 0.05
    start_time = time.time()
    show_plot = False  # 是否显示图表
    
    # 设置 matplotlib 交互模式
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # ========== 主循环 ==========
    while True:
        # 1. 获取图像
        ret, frame = cam.read()
        if not ret:
            print("无法获取图像")
            break
        
        frame_count += 1
        current_time = time.time() - start_time
        
        # 2. 检测板子
        board = detector.detect(frame)
        
        # 3. 获取原始坐标
        laser_center = detector.laser_center
        
        # 4. 分别用有KF和无KF的tracker处理
        if laser_center is not None:
            # 无KF的原始角度
            raw_yaw_val, raw_pitch_val = tracker_without_kf.pixel_to_yaw_pitch(laser_center)
            # 有KF的滤波角度
            kf_yaw_val, kf_pitch_val = tracker_with_kf.track(laser_center)
        else:
            # 目标丢失时的处理
            raw_yaw_val, raw_pitch_val = 0.0, 0.0
            kf_yaw_val, kf_pitch_val = tracker_with_kf.track(None)
        
        # 5. 记录数据
        timestamps.append(current_time)
        raw_yaw.append(raw_yaw_val)
        raw_pitch.append(raw_pitch_val)
        kf_yaw.append(kf_yaw_val)
        kf_pitch.append(kf_pitch_val)
        
        # 6. 发送数据（使用KF滤波后的角度）
        if laser_center is not None and (current_time - last_send_time) >= send_interval:
            if ser_port:
                try:
                    print(f"[串口] Yaw(KF)={kf_yaw_val:.2f}°, Pitch(KF)={kf_pitch_val:.2f}°")
                    print(f"       原始 Yaw={raw_yaw_val:.2f}°, Pitch={raw_pitch_val:.2f}°")
                except Exception as e:
                    print(f"发送失败: {e}")
            
            last_send_time = current_time
        
        # 7. 绘制图像
        result = detector.draw_boards(frame, show_coords=True)
        
        # 在图像上显示滤波效果对比
        cv2.putText(result, f"Raw  Yaw: {raw_yaw_val:6.2f} deg", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(result, f"KF   Yaw: {kf_yaw_val:6.2f} deg", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(result, f"Diff Yaw: {abs(kf_yaw_val - raw_yaw_val):6.2f} deg", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 显示KF状态
        if tracker_with_kf.predict:
            status_text = "KF: Predicting" if not laser_center else "KF: Tracking"
            cv2.putText(result, status_text, (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 8. 更新图表
        if show_plot and len(timestamps) > 10:
            # 更新Yaw图表
            ax1.clear()
            ax1.plot(list(timestamps), list(raw_yaw), 'b-', label='Raw Yaw', alpha=0.6, linewidth=1)
            ax1.plot(list(timestamps), list(kf_yaw), 'r-', label='KF Yaw', linewidth=2)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Yaw (deg)')
            ax1.set_title('Yaw Angle Comparison (Blue: Raw, Red: KF)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 更新Pitch图表
            ax2.clear()
            ax2.plot(list(timestamps), list(raw_pitch), 'b-', label='Raw Pitch', alpha=0.6, linewidth=1)
            ax2.plot(list(timestamps), list(kf_pitch), 'r-', label='KF Pitch', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Pitch (deg)')
            ax2.set_title('Pitch Angle Comparison (Blue: Raw, Red: KF)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
        
        # 9. 显示图像窗口
        closing = detector.process(frame)
        cv2.imshow('Mask', closing)
        cv2.imshow('Detection', result)
        
        # 10. 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"已保存: {filename}")
        elif key == ord('p'):
            show_plot = not show_plot
            if show_plot:
                print("图表显示已开启")
                plt.show()
            else:
                print("图表显示已关闭")
                plt.close()
    
    # ========== 清理 ==========
    if ser_port:
        ser_port.close()
    cam.cam.release()
    cv2.destroyAllWindows()
    plt.close('all')
    print("程序结束")


if __name__ == "__main__":
    main()