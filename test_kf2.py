import cv2
import numpy as np
import time
from detector import Detector, Laser
from tracker import Tracker
from camera import Camera
import ser
from Kalman import KalmanFilter
import matplotlib.pyplot as plt
from collections import deque
from mpl_toolkits.mplot3d import Axes3D

def draw_mini_chart(img, data_raw, data_kf, width=300, height=150, pos=(10, 200), title="Yaw"):
    """在图像上绘制小型实时曲线图"""
    if len(data_raw) < 2:
        return img
    
    # 创建小画布
    chart = np.zeros((height, width, 3), dtype=np.uint8)
    chart[:] = (240, 240, 240)  # 浅灰色背景
    
    # 归一化数据到图表高度
    data_list_raw = list(data_raw)
    data_list_kf = list(data_kf)
    
    all_data = data_list_raw + data_list_kf
    if max(all_data) - min(all_data) < 0.1:
        return img
    
    y_min, y_max = min(all_data), max(all_data)
    y_range = y_max - y_min
    y_padding = y_range * 0.15
    y_min -= y_padding
    y_max += y_padding
    y_range = y_max - y_min
    
    # 绘制网格线
    for i in range(5):
        y = int(height * i / 4)
        cv2.line(chart, (0, y), (width, y), (200, 200, 200), 1)
    
    # 绘制曲线
    x_step = width / max(len(data_list_raw) - 1, 1)
    
    for i in range(len(data_list_raw) - 1):
        # 原始数据（蓝色）
        y1 = height - int((data_list_raw[i] - y_min) / y_range * height)
        y2 = height - int((data_list_raw[i+1] - y_min) / y_range * height)
        x1, x2 = int(i * x_step), int((i+1) * x_step)
        cv2.line(chart, (x1, y1), (x2, y2), (255, 150, 0), 2)
        
        # KF数据（绿色）
        if i < len(data_list_kf) - 1:
            y1 = height - int((data_list_kf[i] - y_min) / y_range * height)
            y2 = height - int((data_list_kf[i+1] - y_min) / y_range * height)
            cv2.line(chart, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 添加标题和数值
    cv2.putText(chart, f"{title}", (5, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(chart, f"Raw: {data_list_raw[-1]:.2f}", (5, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)
    cv2.putText(chart, f"KF: {data_list_kf[-1]:.2f}", (5, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(chart, f"Diff: {abs(data_list_raw[-1]-data_list_kf[-1]):.2f}", (5, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # 将图表叠加到主图像上
    x, y = pos
    if y+height <= img.shape[0] and x+width <= img.shape[1]:
        img[y:y+height, x:x+width] = chart
        cv2.rectangle(img, (x, y), (x+width, y+height), (0, 0, 0), 2)
    
    return img

def draw_noise_meter(img, raw_val, kf_val, pos=(10, 360)):
    """绘制噪声水平指示器"""
    diff = abs(raw_val - kf_val)
    max_diff = 5.0  # 最大期望差值
    
    # 计算条的长度
    bar_width = 250
    bar_height = 15
    fill_width = min(int(diff / max_diff * bar_width), bar_width)
    
    x, y = pos
    
    # 背景条
    cv2.rectangle(img, (x, y), (x+bar_width, y+bar_height), (80, 80, 80), -1)
    cv2.rectangle(img, (x, y), (x+bar_width, y+bar_height), (0, 0, 0), 1)
    
    # 填充条（颜色根据差值变化）
    if diff < 1.0:
        color = (0, 255, 0)  # 绿色：噪声小
    elif diff < 3.0:
        color = (0, 255, 255)  # 黄色：噪声中等
    else:
        color = (0, 0, 255)  # 红色：噪声大
    
    cv2.rectangle(img, (x, y), (x+fill_width, y+bar_height), color, -1)
    
    # 文字
    cv2.putText(img, f"Noise Level: {diff:.2f} deg", (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 刻度标记
    for i in range(6):
        tick_x = x + int(bar_width * i / 5)
        cv2.line(img, (tick_x, y+bar_height), (tick_x, y+bar_height+5), (0, 0, 0), 1)
        cv2.putText(img, f"{i*1.0:.1f}", (tick_x-10, y+bar_height+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    return img

def draw_stats_panel(img, raw_yaw, raw_pitch, kf_yaw, kf_pitch, pos=(10, 400)):
    """绘制统计信息面板"""
    x, y = pos
    panel_width = 250
    panel_height = 120
    
    # 半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+panel_width, y+panel_height), (50, 50, 50), -1)
    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    cv2.rectangle(img, (x, y), (x+panel_width, y+panel_height), (0, 0, 0), 2)
    
    # 计算统计信息
    if len(raw_yaw) > 5:
        yaw_std_raw = np.std(list(raw_yaw))
        yaw_std_kf = np.std(list(kf_yaw))
        pitch_std_raw = np.std(list(raw_pitch))
        pitch_std_kf = np.std(list(kf_pitch))
        
        improvement_yaw = (1 - yaw_std_kf / (yaw_std_raw + 1e-6)) * 100
        improvement_pitch = (1 - pitch_std_kf / (pitch_std_raw + 1e-6)) * 100
        
        texts = [
            "=== Statistics ===",
            f"Yaw Std  Raw: {yaw_std_raw:.3f}",
            f"Yaw Std  KF:  {yaw_std_kf:.3f}",
            f"Improve: {improvement_yaw:.1f}%",
            f"Pitch Std Raw: {pitch_std_raw:.3f}",
            f"Pitch Std KF:  {pitch_std_kf:.3f}",
            f"Improve: {improvement_pitch:.1f}%"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(img, text, (x+10, y+20+i*15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    return img

def update_3d_trajectory(ax, raw_points, kf_points):
    """更新3D轨迹"""
    ax.clear()
    
    if len(raw_points) > 1:
        raw_points = np.array(raw_points)
        kf_points = np.array(kf_points)
        time_axis = np.arange(len(raw_points))
        
        # 绘制轨迹
        ax.plot(raw_points[:, 0], raw_points[:, 1], time_axis, 
                'b-', alpha=0.5, linewidth=1, label='Raw Trajectory')
        ax.plot(kf_points[:, 0], kf_points[:, 1], time_axis, 
                'r-', linewidth=2, label='KF Trajectory')
        
        # 当前点
        ax.scatter([raw_points[-1, 0]], [raw_points[-1, 1]], [time_axis[-1]], 
                   c='blue', s=50, marker='o', label='Current Raw')
        ax.scatter([kf_points[-1, 0]], [kf_points[-1, 1]], [time_axis[-1]], 
                   c='red', s=100, marker='*', label='Current KF')
    
    ax.set_xlabel('Yaw (deg)', fontsize=10)
    ax.set_ylabel('Pitch (deg)', fontsize=10)
    ax.set_zlabel('Time Steps', fontsize=10)
    ax.set_title('3D Trajectory Visualization', fontsize=12)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.draw()
    plt.pause(0.001)

def main():
    """
    主函数：增强版卡尔曼滤波效果可视化
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
    
    # 串口（根据实际情况注释/取消注释）
    try:
        ser_port = ser.Serial(port='/dev/ttyUSB0', baudrate=115200)
    except:
        print("串口初始化失败，使用模拟模式")
        ser_port = None
    
    print("\n" + "="*50)
    print("激光追踪系统 - 卡尔曼滤波可视化")
    print("="*50)
    print("控制键：")
    print("  'q' - 退出程序")
    print("  's' - 保存当前画面")
    print("  'p' - 显示/隐藏 2D 图表")
    print("  '3' - 显示/隐藏 3D 轨迹图")
    print("  'c' - 清空数据记录")
    print("="*50 + "\n")
    
    # ========== 数据记录 ==========
    data_length = 200  # 保存最近200个数据点
    timestamps = deque(maxlen=data_length)
    raw_yaw = deque(maxlen=data_length)
    raw_pitch = deque(maxlen=data_length)
    kf_yaw = deque(maxlen=data_length)
    kf_pitch = deque(maxlen=data_length)
    
    # 3D轨迹数据
    trajectory_raw = []
    trajectory_kf = []
    max_trajectory = 100
    
    frame_count = 0
    last_send_time = 0
    send_interval = 0.05
    start_time = time.time()
    show_plot = False
    show_3d = False
    
    # 设置 matplotlib 交互模式
    plt.ion()
    
    # 2D图表
    fig_2d, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 3D图表
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # FPS计算
    fps_counter = deque(maxlen=30)
    last_fps_time = time.time()
    
    # ========== 主循环 ==========
    while True:
        # 1. 获取图像
        ret, frame = cam.read()
        if not ret:
            print("无法获取图像")
            break
        
        # 计算FPS
        current_fps_time = time.time()
        fps = 1.0 / (current_fps_time - last_fps_time + 1e-6)
        fps_counter.append(fps)
        avg_fps = np.mean(fps_counter)
        last_fps_time = current_fps_time
        
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
        
        # 记录3D轨迹
        if laser_center is not None:
            trajectory_raw.append([raw_yaw_val, raw_pitch_val])
            trajectory_kf.append([kf_yaw_val, kf_pitch_val])
            if len(trajectory_raw) > max_trajectory:
                trajectory_raw.pop(0)
                trajectory_kf.pop(0)
        
        # 6. 发送数据（使用KF滤波后的角度）
        if laser_center is not None and (current_time - last_send_time) >= send_interval:
            if ser_port:
                try:
                    ser_port.send_data(kf_yaw_val, kf_pitch_val)
                    print(f"[串口] Yaw(KF)={kf_yaw_val:6.2f}°, Pitch(KF)={kf_pitch_val:6.2f}°  FPS:{avg_fps:.1f}")
                except Exception as e:
                    print(f"发送失败: {e}")
            else:
                if frame_count % 10 == 0:  # 每10帧打印一次
                    print(f"[模拟] Yaw(KF)={kf_yaw_val:6.2f}°, Pitch(KF)={kf_pitch_val:6.2f}°  FPS:{avg_fps:.1f}")
            
            last_send_time = current_time
        
        # 7. 绘制图像
        result = detector.draw_boards(frame, show_coords=True)
        
        # 在图像上显示FPS
        cv2.putText(result, f"FPS: {avg_fps:.1f}", (result.shape[1]-100, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 在图像上显示滤波效果对比
        cv2.putText(result, f"Raw  Yaw: {raw_yaw_val:6.2f} deg", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(result, f"KF   Yaw: {kf_yaw_val:6.2f} deg", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(result, f"Diff Yaw: {abs(kf_yaw_val - raw_yaw_val):6.2f} deg", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(result, f"Raw  Pitch: {raw_pitch_val:6.2f} deg", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(result, f"KF   Pitch: {kf_pitch_val:6.2f} deg", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(result, f"Diff Pitch: {abs(kf_pitch_val - raw_pitch_val):6.2f} deg", (10, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 显示KF状态
        if tracker_with_kf.predict:
            if not laser_center:
                status_text = "KF: PREDICTING"
                color = (0, 165, 255)
            else:
                status_text = "KF: TRACKING"
                color = (0, 255, 0)
            cv2.putText(result, status_text, (10, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 绘制增强可视化
        if len(raw_yaw) > 10:
            result = draw_mini_chart(result, raw_yaw, kf_yaw, 
                                     width=280, height=130, pos=(10, 220), title="Yaw")
            result = draw_mini_chart(result, raw_pitch, kf_pitch, 
                                     width=280, height=130, pos=(10, 360), title="Pitch")
        
        result = draw_noise_meter(result, raw_yaw_val, kf_yaw_val, pos=(300, 50))
        result = draw_noise_meter(result, raw_pitch_val, kf_pitch_val, pos=(300, 100))
        
        if len(raw_yaw) > 10:
            result = draw_stats_panel(result, raw_yaw, raw_pitch, kf_yaw, kf_pitch, pos=(300, 150))
        
        # 8. 更新2D图表
        if show_plot and len(timestamps) > 10:
            # 更新Yaw图表
            ax1.clear()
            ax1.plot(list(timestamps), list(raw_yaw), 'b-', label='Raw Yaw', alpha=0.5, linewidth=1)
            ax1.plot(list(timestamps), list(kf_yaw), 'r-', label='KF Yaw', linewidth=2)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Yaw (deg)')
            ax1.set_title('Yaw Angle Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 更新Pitch图表
            ax2.clear()
            ax2.plot(list(timestamps), list(raw_pitch), 'b-', label='Raw Pitch', alpha=0.5, linewidth=1)
            ax2.plot(list(timestamps), list(kf_pitch), 'r-', label='KF Pitch', linewidth=2)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Pitch (deg)')
            ax2.set_title('Pitch Angle Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
        
        # 9. 更新3D轨迹
        if show_3d and len(trajectory_raw) > 1:
            update_3d_trajectory(ax_3d, trajectory_raw, trajectory_kf)
        
        # 10. 显示图像窗口
        closing = detector.process(frame)
        cv2.imshow('Mask', closing)
        cv2.imshow('Detection', result)
        
        # 11. 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{frame_count:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"已保存: {filename}")
        elif key == ord('p'):
            show_plot = not show_plot
            if show_plot:
                print("2D图表显示已开启")
                plt.show()
            else:
                print("2D图表显示已关闭")
                plt.close(fig_2d)
                fig_2d, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        elif key == ord('3'):
            show_3d = not show_3d
            if show_3d:
                print("3D轨迹显示已开启")
                plt.show()
            else:
                print("3D轨迹显示已关闭")
                plt.close(fig_3d)
                fig_3d = plt.figure(figsize=(10, 8))
                ax_3d = fig_3d.add_subplot(111, projection='3d')
        elif key == ord('c'):
            # 清空数据
            timestamps.clear()
            raw_yaw.clear()
            raw_pitch.clear()
            kf_yaw.clear()
            kf_pitch.clear()
            trajectory_raw.clear()
            trajectory_kf.clear()
            print("数据已清空")
    
    # ========== 清理 ==========
    print("\n正在退出程序...")
    
    # 保存最终数据
    if len(timestamps) > 0:
        import pandas as pd
        data = {
            'timestamp': list(timestamps),
            'raw_yaw': list(raw_yaw),
            'raw_pitch': list(raw_pitch),
            'kf_yaw': list(kf_yaw),
            'kf_pitch': list(kf_pitch)
        }
        df = pd.DataFrame(data)
        filename = f"kalman_data_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"数据已保存到 {filename}")
        
        # 打印统计信息
        print("\n" + "="*50)
        print("卡尔曼滤波效果统计")
        print("="*50)
        yaw_std_raw = np.std(list(raw_yaw))
        yaw_std_kf = np.std(list(kf_yaw))
        pitch_std_raw = np.std(list(raw_pitch))
        pitch_std_kf = np.std(list(kf_pitch))
        
        print(f"Yaw 标准差 - 原始: {yaw_std_raw:.3f}°, KF: {yaw_std_kf:.3f}°")
        print(f"Yaw 平滑度提升: {(1 - yaw_std_kf/(yaw_std_raw+1e-6))*100:.1f}%")
        print(f"Pitch 标准差 - 原始: {pitch_std_raw:.3f}°, KF: {pitch_std_kf:.3f}°")
        print(f"Pitch 平滑度提升: {(1 - pitch_std_kf/(pitch_std_raw+1e-6))*100:.1f}%")
        print("="*50)
    
    if ser_port:
        ser_port.close()
    cam.release()
    cv2.destroyAllWindows()
    plt.close('all')
    print("程序结束")


if __name__ == "__main__":
    main()