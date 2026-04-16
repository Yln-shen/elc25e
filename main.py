import cv2
import numpy as np
import camera # 导入你之前的 camera.py
import math
import ser 
import time

RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180

class laser:
    def __init__(self, width_deviation=0, height_deviation=5):
        self.width_deviation = width_deviation   # 水平方向偏差（像素）
        self.height_deviation = height_deviation # 垂直方向偏差（像素）
    
class Board:
    def __init__(self): 
        self.points = []  # 存储四个角点（原始图像坐标）[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        self.center = None  # 中心点坐标（原始图像坐标）(x, y)
        # self.relative_points = []  # 相对于图像中心点的坐标
        # self.relative_center = None  # 相对于图像中心的中心点坐标
        self.laser_center = None  # 相对于激光中心的坐标(转换后)

class Detector:
    def __init__(self, color, rectangle_min_area, kernel):
        self.black_lower = np.array(color[0])
        self.black_upper = np.array(color[1])

        self.rectangle_min_area = rectangle_min_area

        # self.rectangle = rectangle_config
        self.kernel = np.ones(kernel, np.uint8)

        self.current_board = None  # 存储当前检测到的板子
        # self.boards = []  # 存储检测到的板子
        self.frame_center = None  # 存储图像中心点
        self.laser = laser()  # 修改：在构造函数中明确初始化laser对象

    def process(self, frame):
        """处理图像，生成掩膜"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 矩形（黑色区域）掩膜
        mask = cv2.inRange(hsv, self.black_lower, self.black_upper)

        # 开运算，先腐蚀再膨胀
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        # 闭运算，先膨胀再腐蚀
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)

        return closing
    
    def find_board(self, closing):
        """查找四边形,并创建板子"""
        rectangle_contours = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        for contour in rectangle_contours:
            area = cv2.contourArea(contour)
            if area < self.rectangle_min_area:
                continue
            # 逼近多边形
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            # 筛选四边形
            if len(approx) != 4:
                continue
            # 获取四个角点
            points = approx.reshape(4, 2)
            # 按左上、左下、右下、右上排序
            sum_xy = points.sum(axis=1) #x+y
            diff_xy = points[:, 0] - points[:, 1] #x-y
            sorted_points = [
                points[np.argmin(sum_xy)],  # 左上：x+y 最小
                points[np.argmax(diff_xy)],  # 左下：x-y 最大           
                points[np.argmax(sum_xy)],  # 右下：x+y 最大
                points[np.argmin(diff_xy)]   # 右上：x-y 最小
            ]

            # 计算宽度（左上到左下）
            width = np.linalg.norm(sorted_points[0] - sorted_points[1])
            # 计算高度（左下到右下）
            height = np.linalg.norm(sorted_points[1] - sorted_points[2])
            # 长宽比 = 长边 / 短边
            ratio = max(width, height) / min(width, height)
            
            # 板子应该是矩形，长宽比在1.2~1.6之间
            # 如果太接近1（正方形）或太大（细长条），就不是我们要的
            if not (1.2 <= ratio <= 1.6):
                continue

            board = Board()
            board.points = [tuple(pt) for pt in sorted_points]  # 转成元组列表
            
            # ========== 第7步：计算中心点（像素坐标）==========
            # 中心点 = 四个角点的平均值
            cx = int(sum(p[0] for p in board.points) / 4)
            cy = int(sum(p[1] for p in board.points) / 4)
            board.center = (cx, cy)
            return board
        return None

    def tf_point(self,board,frame):    
        # ========== 第8步：转换坐标 ==========# ========== 第8步：转换坐标（以激光笔中心为原点）==========
        # 8.1 获取图像中心坐标
        if board is not None:
            h, w = frame.shape[:2]
            image_center_x = w / 2
            image_center_y = h / 2
            self.frame_center = (image_center_x, image_center_y)
            # 8.2 激光笔中心在图像中的像素坐标（需要标定）
            #     width_deviation: 激光光斑相对于图像中心的水平偏移（像素）
            #     height_deviation: 激光光斑相对于图像中心的垂直偏移（像素）
            laser_pixel_x = image_center_x + self.laser.width_deviation
            laser_pixel_y = image_center_y + self.laser.height_deviation

            # 8.3 计算板子中心相对于激光中心的坐标
            #     原点 = 激光笔中心
            #     X轴正方向：向右
            #     Y轴正方向：向下（图像坐标系）
            laser_center_x = board.center[0] - laser_pixel_x
            laser_center_y = board.center[1] - laser_pixel_y

            # 8.4 存入 board
            board.laser_center = (laser_center_x, laser_center_y)
            
            # ========== 第9步：存到 current_board，然后返回 ==========
            self.current_board = board 
        else:
            self.current_board = None
    # self.current_board = None

    def detect(self, frame):
        closing = self.process(frame)
        board = self.find_board(closing)
        self.tf_point(board,frame)
        return board
    
    def get_laser_center(self):
        """获取以激光笔为原点的板子坐标"""
        if self.current_board is None:
            return None
        return self.current_board.laser_center
    
    def draw_boards(self, frame, show_coords=True):
        """
        在图像上绘制检测结果，只显示坐标转换后的坐标系
        
        坐标系说明：
            - 原点 = 激光笔中心
            - X轴正方向：向右
            - Y轴正方向：向下
        """
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # 1. 计算图像中心和激光笔中心在像素坐标系中的位置
        image_center_pixel = (int(w/2), int(h/2))
        laser_pixel_x = w/2 + self.laser.width_deviation
        laser_pixel_y = h/2 + self.laser.height_deviation
        laser_pixel = (int(laser_pixel_x), int(laser_pixel_y))
        
        # 2. 绘制坐标轴（以激光笔为原点）
        axis_length = 50  # 坐标轴长度（像素）
        # X轴（红色）
        cv2.arrowedLine(result, laser_pixel, 
                        (laser_pixel[0] + axis_length, laser_pixel[1]), 
                        (0, 0, 255), 2, tipLength=0.1)
        cv2.putText(result, "X", (laser_pixel[0] + axis_length + 5, laser_pixel[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Y轴（绿色）
        cv2.arrowedLine(result, laser_pixel, 
                        (laser_pixel[0], laser_pixel[1] + axis_length), 
                        (0, 255, 0), 2, tipLength=0.1)
        cv2.putText(result, "Y", (laser_pixel[0] + 5, laser_pixel[1] + axis_length + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 3. 绘制图像中心（绿色点）- 不显示坐标
        cv2.circle(result, image_center_pixel, 5, (0, 255, 0), -1)
        cv2.putText(result, "Image Center", (image_center_pixel[0] + 10, image_center_pixel[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # 4. 绘制激光笔中心（红色点，原点）
        cv2.circle(result, laser_pixel, 8, (0, 0, 255), -1)
        cv2.putText(result, "Laser (Origin)", (laser_pixel[0] + 10, laser_pixel[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # 激光笔在转换后坐标系中的坐标（总是原点）
        cv2.putText(result, "Laser Coord: (0, 0)", (laser_pixel[0] + 10, laser_pixel[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 5. 如果没有检测到板子
        if self.current_board is None:
            cv2.putText(result, "No Board", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result
        
        board = self.current_board
        
        # 6. 绘制四边形边框
        pts = np.array(board.points, dtype=np.int32)
        cv2.polylines(result, [pts], True, (255, 0, 0), 2)
        
        # 7. 绘制板子中心点（蓝色点）
        cv2.circle(result, board.center, 6, (255, 0, 0), -1)
        cv2.putText(result, "Board Center", (board.center[0] + 10, board.center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # 8. 显示板子中心在激光坐标系中的坐标（只显示转换后的坐标）
        if show_coords and board.laser_center:
            # 在板子中心旁边显示坐标
            coord_text = f"({board.laser_center[0]:.1f}, {board.laser_center[1]:.1f})"
            cv2.putText(result, coord_text, (board.center[0] - 60, board.center[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 在图像左下角显示详细信息
            info_text = f"Board Coord: ({board.laser_center[0]:.1f}, {board.laser_center[1]:.1f})"
            cv2.putText(result, info_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 9. 显示状态信息
        cv2.putText(result, "Board Found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result

class Tracker:
    def __init__(self, vfov=100, img_width=640):
        self.vfov = vfov
        self.img_width = img_width

    def pixel_to_yaw_pitch(self, laser_center):
        """将像素坐标转换为偏航角和俯仰角"""
        if laser_center is None:
            return 0.0, 0.0
            
        vfov_rad = self.vfov * DEG2RAD
        
        # 计算焦距（像素单位）
        # 焦距 = 图像半宽 / tan(半视场角)
        focal = (self.img_width / 2) / math.tan(vfov_rad / 2)
        
        if abs(focal) < 1e-6:
            return 0.0, 0.0
        
        # 计算角度
        yaw = math.atan(laser_center[0] / focal) * RAD2DEG
        pitch = math.atan(laser_center[1] / focal) * RAD2DEG
        
        return yaw, pitch
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
        cam = camera.Camera(index=0)
    except Exception as e:
        print(f"摄像头初始化失败: {e}，尝试默认摄像头...")
        cam = camera.Camera(index=0)
    
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

