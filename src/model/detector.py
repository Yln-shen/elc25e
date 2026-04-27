import cv2
import numpy as np
import math

RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180

class Laser:
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
    def __init__(self, rectangle_min_area, rectangle_max_area, laser=None):
        #self.black_lower = np.array(color[0])
        #self.black_upper = np.array(color[1])

        self.rectangle_max_area = rectangle_max_area
        self.rectangle_min_area = rectangle_min_area

        # self.rectangle = rectangle_config
        #self.kernel = np.ones(kernel, np.uint8)

        self.relative_board = None  # 存储当前检测到的板子
        self.boards = []
          # 存储检测到的板子
        self.frame_center = None  # 存储图像中心点坐标
        self.laser_pixel = None  # 存储激光笔中心点坐标
        self.board_center = None  # 存储板子中心点坐标
        self.laser_center = None  # 存储板子中心相对于激光中心的坐标

        self.laser = laser  # 修改：在构造函数中明确初始化laser对象

    # def process(self, frame):
    #     """处理图像，生成掩膜"""
    #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     # 矩形（黑色区域）掩膜
    #     mask = cv2.inRange(hsv, self.black_lower, self.black_upper)

    #     # 开运算，先腐蚀再膨胀
    #     opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
    #     # 闭运算，先膨胀再腐蚀
    #     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)

    #     return closing

    def process(self, frame):
        """处理图像，生成掩膜（使用 Otsu 二值化）"""
        # 1. 转为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Otsu 反二值化（因为你的板子是黑色边框/黑色区域）
        #    THRESH_BINARY_INV: 暗的部分变白，亮的部分变黑
        #    THRESH_OTSU: 自动计算最佳阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. 形态学操作（去噪、填洞）
        # opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)
        
        return binary

    
    def find_boards(self, binary):
        """查找四边形,并创建板子"""
        boards = []
        rectangle_contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        for contour in rectangle_contours:
            #筛选面积
            area = cv2.contourArea(contour)
            if area < self.rectangle_min_area or area > self.rectangle_max_area:
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
            if not (1.1 <= ratio <= 1.6):
                continue

            board = Board()
            board.points = [tuple(pt) for pt in sorted_points]  # 转成元组列表
            board.area = area
            # ========== 第7步：计算中心点（像素坐标）==========
            # 中心点 = 四个角点的平均值
            cx = int(sum(p[0] for p in board.points) / 4)
            cy = int(sum(p[1] for p in board.points) / 4)
            board.center = (cx, cy)
            #self.board_center = board.center
            boards.append(board)
        return boards
    
    def select_board(self,boards):
        """得到当前帧里一个正确的板子"""
        if len(boards) == 0:
            board = None
        else:
            board = max(boards, key=lambda board: board.area)
        return board

    def tf_point(self, board, frame):    
        """转换坐标（以激光笔中心为原点）"""
        # 获取图像尺寸和中心坐标
        h, w = frame.shape[:2]
        image_center_x = w / 2
        image_center_y = h / 2
        self.frame_center = (int(image_center_x), int(image_center_y))
        
        # 激光笔中心在图像中的像素坐标
        laser_pixel_x = int(image_center_x + self.laser.width_deviation)
        laser_pixel_y = int(image_center_y + self.laser.height_deviation)
        self.laser_pixel = (laser_pixel_x, laser_pixel_y)
        
        if board is not None:
            self.board_center = board.center
            # 计算板子中心相对于激光中心的坐标
            laser_center_x = board.center[0] - laser_pixel_x
            laser_center_y = board.center[1] - laser_pixel_y
            # 存入 board
            board.laser_center = (laser_center_x, laser_center_y)
            self.laser_center = board.laser_center
        else:
            self.board_center = None
            self.laser_center = None

    def detect(self, frame):
        """主检测函数"""
        closing = self.process(frame)
        boards = self.find_boards(closing)
        board = self.select_board(boards)
        self.tf_point(board, frame)
        self.relative_board = board
        return board
    
    def draw_boards(self, frame, show_coords=True):
        """
        在图像上绘制检测结果，只显示坐标转换后的坐标系
        
        坐标系说明：
            - 原点 = 激光笔中心
            - X轴正方向：向右
            - Y轴正方向：向下
        """
        result = frame.copy()
      
        #绘制图像中心（绿色点）- 不显示坐标
        cv2.circle(result, self.frame_center, 5, (0, 0, 255), -1)
        cv2.putText(result, "Frame Center", (self.frame_center[0] + 10, self.frame_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        #绘制激光笔中心（紫色点）
        cv2.circle(result, self.laser_pixel, 8, (125, 43, 46), -1)
        cv2.putText(result, "Laser Center", (self.laser_pixel[0] + 10, self.laser_pixel[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 43, 46), 1)
        
        if self.board_center is not None:
        #绘制板子中心（蓝色点）
            cv2.circle(result, self.board_center, 8, (0, 255, 0), -1)
            cv2.putText(result, "Board center", (self.board_center[0] + 10, self.board_center[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
        #如果没有检测到板子
        if self.relative_board is None:
            cv2.putText(result, "No Board", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result
        
        board = self.relative_board
        
        #绘制四边形边框
        pts = np.array(board.points, dtype=np.int32)
        cv2.polylines(result, [pts], True, (255, 0, 0), 2)
        
        #显示板子中心在激光坐标系中的坐标（只显示转换后的坐标）
        if show_coords and board.laser_center:
            # 在板子中心旁边显示坐标
            coord_text = f"({self.laser_center[0]:.1f}, {self.laser_center[1]:.1f})"
            cv2.putText(result, coord_text, (board.center[0] - 60, board.center[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
        # 9. 显示状态信息
        cv2.putText(result, "Board Found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result
    
if __name__ == '__main__':
    from camera import Camera  # 导入camera.py
    from ser import Serial
    from tracker import Tracker
    import time
    
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
        frame_add=35
    )

    # 串口通信
    serial_port = Serial(
        port='/dev/ttyACM0',
        baudrate=115200,
        timeout=1,
        write_timeout=1
    )

    # 帧计数
    fps_counter = 0
    fps_display = 0
    fps_timer = time.time()
    capture_count = 0

    print("按 'q' 退出，按 's' 保存截图")
    print("-" * 40)

    # ========== 主循环 ==========
    try:
        while True:
            # 1. 获取图像
            ret, frame = cam.read()
            if not ret:
                print("无法获取图像")
                break

            # 2. 计算FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
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
                          f"偏航: {yaw:>6.1f}°  俯仰: {pitch:>6.1f}°  FPS: {fps_display}", end="")
                else:
                    print(f"\r板子坐标: ({laser_center[0]:>7.1f}, {laser_center[1]:>7.1f})  "
                          f"已对准中心  FPS: {fps_display}", end="")
            
            elif tracker.if_find and laser_center is None:
                # 卡尔曼预测状态，没有实际检测到
                print(f"\r预测跟踪中...  偏航: {yaw:>6.1f}°  俯仰: {pitch:>6.1f}°  FPS: {fps_display}", end="")
            
            elif board is not None and board.laser_center is not None:
                # 检测到板子但跟踪未就绪
                print(f"\r板子坐标: ({laser_center[0]:>7.1f}, {laser_center[1]:>7.1f})  "
                      f"等待跟踪...  FPS: {fps_display}", end="")
            
            else:
                # 未检测到板子
                print(f"\r未检测到板子  FPS: {fps_display}", end="")

            # 7. 显示FPS和状态
            cv2.putText(result, f"FPS: {fps_display}", (10, 60),
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