import cv2
import numpy as np
import camera # 导入你之前的 camera.py
import math
import ser 
import time

RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180

class Laser:
    def __init__(self, lower_laser, upper_laser, laser_max_area):
        self.lower_laser = np.array(lower_laser)
        self.upper_laser = np.array(upper_laser)
        self.area = laser_max_area 

class Rectangle:
    def __init__(self, black_lower, black_upper, rectangle_min_area):
        self.black_lower = np.array(black_lower)
        self.black_upper = np.array(black_upper)
        self.rectangle_min_area = rectangle_min_area

class Board:
    def __init__(self):
        self.points = []  # 存储四个角点（原始图像坐标）
        self.center = None  # 中心点坐标（原始图像坐标）
        self.relative_points = []  # 相对于图像中心点的坐标
        self.relative_center = None  # 相对于图像中心的中心点坐标

class Detector:
    def __init__(self, laser_config, rectangle_config, kernel):
        self.laser = laser_config
        self.rectangle = rectangle_config
        self.kernel = np.ones(kernel, np.uint8)
        self.boards = []  # 存储检测到的板子
        self.frame_center = None  # 存储图像中心点

    def process(self, frame):
        """处理图像，生成掩膜"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 激光掩膜
        laser_mask = cv2.inRange(hsv, self.laser.lower_laser, self.laser.upper_laser)
        # 矩形（黑色区域）掩膜
        rectangle_mask = cv2.inRange(hsv, self.rectangle.black_lower, self.rectangle.black_upper)
        # 合并掩膜（使用 OR 操作）
        mask = cv2.bitwise_or(laser_mask, rectangle_mask)

        # 开运算，先腐蚀再膨胀
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        # 闭运算，先膨胀再腐蚀
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)

        return mask, closing

    def tf_point(self, point, frame):
        """
        转换坐标原点，让原点变成图像中心位置
        
        Args:
            point: 原始坐标 (x, y)，基于左上角原点
            frame: 图像数组，用于获取尺寸信息
            
        Returns:
            tuple: 转换后的坐标 (x', y')，基于图像中心原点
        """
        if frame is None:
            raise ValueError("No frame available for coordinate transformation")
        
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError("point must be a tuple/list of (x, y)")
        
        height, width = frame.shape[:2]
        # 存储图像中心点供后续使用
        self.frame_center = (width / 2, height / 2)
        center_x = point[0] - width / 2
        center_y = point[1] - height / 2
        return (center_x, center_y)
    
    def tf_points_batch(self, points, frame):
        """批量转换多个坐标点"""
        height, width = frame.shape[:2]
        self.frame_center = (width / 2, height / 2)
        return [(p[0] - width / 2, p[1] - height / 2) for p in points]
    
    def find_board(self, binary, frame=None):
        """查找四边形板子，如果提供frame则同时计算相对坐标"""
        boards = []
        board_contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        for contour in board_contours:
            area = cv2.contourArea(contour)
            if area > self.rectangle.rectangle_min_area:
                # 逼近多边形，获取四边形
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # 筛选四边形
                if len(approx) == 4:
                    # 获取四个角点
                    points = approx.reshape(4, 2)
                    
                    # 按左上、左下、右下、右上排序
                    sum_xy = points.sum(axis=1)
                    diff_xy = points[:, 0] - points[:, 1]
                    sorted_points = [
                        points[np.argmin(sum_xy)],  # 左上：x+y 最小
                        points[np.argmax(diff_xy)],  # 左下：x-y 最大
                        points[np.argmax(sum_xy)],  # 右下：x+y 最大
                        points[np.argmin(diff_xy)]   # 右上：x-y 最小
                    ]
                    
                    # 计算长宽比（A4纸比例筛选）
                    # 计算宽度和高度
                    width = np.linalg.norm(sorted_points[0] - sorted_points[1])  # 左边宽度
                    height = np.linalg.norm(sorted_points[1] - sorted_points[2])  # 底部高度
                    
                    # 计算长宽比（确保比值 >= 1）
                    ratio = max(width, height) / min(width, height)
                    
                    # A4纸比例大约是1.414，允许一定误差（1.2-1.6）
                    if 1.2 <= ratio <= 1.6:
                        # 创建 Board 对象
                        board = Board()
                        board.points = [tuple(pt) for pt in sorted_points]
                        
                        # 计算原始中心点
                        center_x = int(sum(p[0] for p in board.points) / 4)
                        center_y = int(sum(p[1] for p in board.points) / 4)
                        board.center = (center_x, center_y)
                        
                        # 如果提供了frame，计算相对于图像中心的坐标
                        if frame is not None:
                            # 转换四个角点
                            board.relative_points = self.tf_points_batch(board.points, frame)
                            # 转换中心点
                            board.relative_center = self.tf_point(board.center, frame)
                        
                        boards.append(board)
                        
                        # 打印信息
                        if frame is not None:
                            print(f"检测到板子:")
                            print(f"  原始中心: ({center_x}, {center_y})")
                            print(f"  相对中心: ({board.relative_center[0]:.1f}, {board.relative_center[1]:.1f})")
        
        self.boards = boards
        if boards:
            center = (boards[0].relative_center[0], boards[0].relative_center[1])
            return boards, center
        return boards, None
    
    def draw_boards(self, frame, show_relative=False):
        """在图像上绘制检测到的板子，可选择显示相对坐标"""
        result = frame.copy()
        
        for board in self.boards:
            # 绘制四边形边框
            pts = np.array(board.points, dtype=np.int32)
            cv2.polylines(result, [pts], True, (0, 255, 0), 2)
            
            # 绘制中心点
            cv2.circle(result, board.center, 5, (255, 0, 0), -1)
            
            # 显示坐标信息
            if show_relative and hasattr(board, 'relative_center') and board.relative_center:
                # 显示相对坐标（基于图像中心）
                cv2.putText(result, 
                           f"Rel: ({board.relative_center[0]:.1f}, {board.relative_center[1]:.1f})", 
                           (board.center[0] - 80, board.center[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # 显示原始坐标
                cv2.putText(result, 
                           f"Abs: {board.center}", 
                           (board.center[0] - 80, board.center[1] + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            else:
                # 只显示原始坐标
                cv2.putText(result, f"Center: {board.center}", 
                           (board.center[0] - 50, board.center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 显示检测到的板子数量
        cv2.putText(result, f'Boards: {len(self.boards)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制图像中心点（坐标系原点）
        if hasattr(self, 'frame_center') and self.frame_center:
            center_pixel = (int(self.frame_center[0]), int(self.frame_center[1]))
            cv2.circle(result, center_pixel, 8, (0, 0, 255), -1)
            cv2.putText(result, "Origin (0,0)", 
                       (center_pixel[0] + 10, center_pixel[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)   
        return result
    
    def get_relative_coordinates(self):
        """获取所有板子的相对坐标"""
        coordinates = []
        for board in self.boards:
            if hasattr(board, 'relative_center') and board.relative_center:
                coordinates.append({
                    'center': board.relative_center,
                    'corners': board.relative_points
                })
        return coordinates
    
class tracker:
    def __init__(self, vfov=100, img_width=640):
        self.vfov = vfov
        self.img_width = img_width

    def pixel_to_yaw_pitch(self, center):
        """将像素坐标转换为偏航角和俯仰角"""
        if center is None:
            return 0.0, 0.0
            
        vfov_radians = self.vfov * DEG2RAD
        focal_pixel_distance = (self.img_width / 2) / math.tan(vfov_radians / 2)
        if focal_pixel_distance == 0:
            focal_pixel_distance = 0.000_000_1
        yaw = math.atan(center[0] / focal_pixel_distance) * RAD2DEG
        pitch = math.atan(center[1] / focal_pixel_distance) * RAD2DEG
        return yaw, pitch

def main():
    """主函数"""
    # 配置参数
    # 激光颜色范围（HSV）
    laser_config = Laser([100, 50, 200], [160, 255, 255], 50)
    # 黑色矩形范围（HSV）
    rectangle_config = Rectangle([0, 0, 0], [180, 255, 70], 3000)

    # 初始化检测器
    detector = Detector(laser_config, rectangle_config, (5,5))

    # 初始化串口 - 修改这里，使用你之前定义的Serial类
    import ser  # 假设你的串口模块文件名为serial_module.py
    try:
        ser = ser.Serial(port='/dev/ttyACM0', baudrate=115200)
        print("串口初始化成功")
    except Exception as e:
        print(f"串口初始化失败: {e}")
        ser = None
    
    # 初始化角度转换器
    angle_tracker = tracker(vfov=100, img_width=640)
    
    # 初始化摄像头
    try:
        cam = camera.Camera(index=2)
    except Exception as e:
        print(f"摄像头初始化失败: {e}")
        print("尝试使用默认摄像头...")
        cam = camera.Camera(index=0)

    print("开始检测...")
    print("按 'q' 退出")
    print("按 's' 保存当前帧")
    print("按 'c' 清除所有板子记录")
    
    frame_count = 0
    last_send_time = 0
    send_interval = 0.05  # 发送间隔50ms，20Hz
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("无法获取图像帧")
            break
        
        frame_count += 1
        
        # 处理图像
        mask, closing = detector.process(frame)
        
        # 查找板子，传入frame以便计算相对坐标
        boards, center = detector.find_board(closing, frame)
        
        # 计算并发送角度
        current_time = time.time()
        if center is not None and (current_time - last_send_time) >= send_interval:
            # 计算yaw和pitch
            yaw, pitch = angle_tracker.pixel_to_yaw_pitch(center)
            
            # 通过串口发送
            if ser:
                try:
                    ser.send_data(yaw, pitch)
                    print(f"发送角度: Yaw={yaw:.2f}, Pitch={pitch:.2f}")
                except Exception as e:
                    print(f"发送失败: {e}")
            
            last_send_time = current_time
        
        # 绘制检测结果（设置show_relative=True显示相对坐标）
        result = detector.draw_boards(frame, show_relative=True)
        
        # 显示角度信息（可选）
        if center is not None:
            yaw, pitch = angle_tracker.pixel_to_yaw_pitch(center)
            cv2.putText(result, f"Yaw: {yaw:.2f} deg", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(result, f"Pitch: {pitch:.2f} deg", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 显示图像
        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Closing', closing)
        cv2.imshow('Board Detection', result)
        
        # 每30帧打印一次相对坐标信息
        if frame_count % 30 == 0 and boards:
            print(f"\n--- 第{frame_count}帧检测结果 ---")
            rel_coords = detector.get_relative_coordinates()
            for i, coord in enumerate(rel_coords):
                print(f"板子{i+1} 相对中心: ({coord['center'][0]:.1f}, {coord['center'][1]:.1f})")
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("退出程序")
            break
        elif key == ord('s'):
            # 保存当前帧
            filename = f"capture_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"已保存图像: {filename}")
            
            # 保存检测结果信息
            if boards:
                with open(f"detection_{frame_count}.txt", 'w') as f:
                    f.write(f"帧序号: {frame_count}\n")
                    f.write(f"检测到的板子数量: {len(boards)}\n")
                    for i, board in enumerate(boards):
                        f.write(f"\n板子{i+1}:\n")
                        f.write(f"  原始中心坐标: {board.center}\n")
                        if hasattr(board, 'relative_center'):
                            f.write(f"  相对中心坐标: {board.relative_center}\n")
                        f.write(f"  四个角点坐标:\n")
                        for j, point in enumerate(board.points):
                            f.write(f"    角点{j+1}: {point}\n")
                print(f"已保存检测信息: detection_{frame_count}.txt")
        
        elif key == ord('c'):
            # 清除板子记录
            detector.boards = []
            print("已清除所有板子记录")

    # 释放资源
    if ser:
        ser.close()
    cam.release()
    cv2.destroyAllWindows()
    print("程序结束")

if __name__ == "__main__":
    main()