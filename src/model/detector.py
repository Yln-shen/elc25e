# detector.py - 只改变量名，保留所有注释
import cv2
import numpy as np
import math
from .pnp import PNPSolver

RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180

class Board:
    def __init__(self): 
        self.points = []  # 存储四个角点（原始图像坐标）[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        self.center = None  # 中心点坐标（原始图像坐标）(x, y)
        self.camera_center = None  # 相对于摄像头中心的坐标(转换后) ← 改

class Detector:
    def __init__(self, rectangle_min_area, rectangle_max_area,use_pnp=True):
        self.rectangle_max_area = rectangle_max_area
        self.rectangle_min_area = rectangle_min_area

        self.relative_board = None  # 存储当前检测到的板子
        self.boards = []  # 存储检测到的板子
        self.frame_center = None  # 存储图像中心点坐标
        self.board_center = None  # 存储板子中心点坐标
        self.camera_center_offset = None  # 存储板子中心相对于摄像头中心的坐标 ← 改

        self.use_pnp = use_pnp
        self.pnp = PNPSolver()

    def process(self, frame):
        """处理图像，生成掩膜（使用 Otsu 二值化）"""
        # 1. 转为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Otsu 反二值化（因为你的板子是黑色边框/黑色区域）
        #    THRESH_BINARY_INV: 暗的部分变白，亮的部分变黑
        #    THRESH_OTSU: 自动计算最佳阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
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
            #逼近多边形
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
                points[np.argmin(diff_xy)],  # 左下：x-y 最小 ← 之前排查角点排序时改的
                points[np.argmax(sum_xy)],  # 右下：x+y 最大
                points[np.argmax(diff_xy)]   # 右上：x-y 最大 ← 之前排查角点排序时改的
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
            # 计算中心点（像素坐标）
            # 中心点 = 四个角点的平均值
            cx = int(sum(p[0] for p in board.points) / 4)
            cy = int(sum(p[1] for p in board.points) / 4)
            board.center = (cx, cy)
            boards.append(board)
        return boards

    def select_board(self, boards):
        """得到当前帧里一个正确的板子"""
        if len(boards) == 0:
            board = None
        else:
            board = max(boards, key=lambda board: board.area)
        return board

    def tf_point(self, board, frame):    
        """转换坐标（以摄像头中心为原点）"""
        # 获取图像尺寸和中心坐标
        h, w = frame.shape[:2]
        image_center_x = w / 2
        image_center_y = h / 2
        self.frame_center = (int(image_center_x), int(image_center_y))
        
        if board is not None:
            self.board_center = board.center
            # 计算板子中心相对于摄像头中心的坐标 ← 改
            camera_center_x = board.center[0] - image_center_x   # ← 改
            camera_center_y = board.center[1] - image_center_y   # ← 改
            # 存入 board
            board.camera_center = (camera_center_x, camera_center_y)  # ← 改
            self.camera_center_offset = board.camera_center      # ← 改
        else:
            self.board_center = None
            self.camera_center_offset = None  # ← 改

    def detect(self, frame):
        """主检测函数"""
        binary = self.process(frame)
        boards = self.find_boards(binary)
        board = self.select_board(boards)
        self.tf_point(board, frame)
        self.relative_board = board

        # PNP调用
        if self.use_pnp and board is not None and len(board.points) == 4:
            print("正在调用PNP, 角点数:", len(board.points)) 
            self.pnp.solve(board.points)
        else:
            print("PNP跳过, board:", board is not None, "角点数:", len(board.points) if board else 0) 
            self.pnp.position = None
            self.pnp.yaw = None
            self.pnp.pitch = None
            self.pnp.distance = None

        return binary, board

    def draw_boards(self, frame, show_coords=True):
        """
        在图像上绘制检测结果
        
        坐标系说明：
            - 原点 = 摄像头中心
            - X轴正方向：向右
            - Y轴正方向：向下
        """
        result = frame.copy()
      
        #绘制图像中心（红色点）
        cv2.circle(result, self.frame_center, 5, (0, 0, 255), -1)
        cv2.putText(result, "Frame Center", (self.frame_center[0] + 10, self.frame_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        if self.board_center is not None:
        #绘制板子中心（绿色点）
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
        
        #显示板子中心在摄像头坐标系中的坐标 ← 改
        if show_coords and board.camera_center:  # ← 改
            # 在板子中心旁边显示坐标
            coord_text = f"({board.camera_center[0]:.1f}, {board.camera_center[1]:.1f})"  # ← 改
            cv2.putText(result, coord_text, (board.center[0] - 60, board.center[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
        # 显示状态信息
        cv2.putText(result, "Board Found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result