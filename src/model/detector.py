# detector.py - 添加PNP中心显示
import cv2
import numpy as np
import math
from .pnp import PNPSolver

RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180

class Board:
    def __init__(self): 
        self.points = []
        self.center = None
        self.camera_center = None

class Detector:
    def __init__(self, rectangle_min_area, rectangle_max_area, use_pnp=True):
        self.rectangle_max_area = rectangle_max_area
        self.rectangle_min_area = rectangle_min_area
        self.relative_board = None
        self.boards = []
        self.frame_center = None
        self.board_center = None
        self.camera_center_offset = None
        self.use_pnp = use_pnp
        self.pnp = PNPSolver()

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)
        return binary

    def order_points(self, pts):
        """排序角点：左上→右上→右下→左下"""
        s = pts.sum(axis=1)           # x+y
        tl = pts[np.argmin(s)]        # 左上
        br = pts[np.argmax(s)]        # 右下
        
        # np.diff(axis=1)计算的是 y-x
        d = np.diff(pts, axis=1).flatten()
        tr = pts[np.argmin(d)]        # 右上：y-x最小
        bl = pts[np.argmax(d)]        # 左下：y-x最大
        
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def find_boards(self, binary):
        boards = []
        contours_result = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rectangle_contours = contours_result[0] if len(contours_result) == 2 else contours_result[1]
        
        for contour in rectangle_contours:
            area = cv2.contourArea(contour)
            if area < self.rectangle_min_area or area > self.rectangle_max_area:
                continue
            
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) != 4:
                continue
            
            if not cv2.isContourConvex(approx):
                continue
            
            points = approx.reshape(4, 2).astype(np.float32)
            rect = self.order_points(points)
            
            if len(np.unique(rect, axis=0)) < 4:
                continue
            
            w1 = np.linalg.norm(rect[0] - rect[1])
            w2 = np.linalg.norm(rect[3] - rect[2])
            h1 = np.linalg.norm(rect[0] - rect[3])
            h2 = np.linalg.norm(rect[1] - rect[2])
            
            w = (w1 + w2) / 2
            h = (h1 + h2) / 2
            
            if w == 0 or h == 0:
                continue
                
            ratio = max(w, h) / min(w, h)
            
            if not (1.1 <= ratio <= 1.8):
                continue
            
            board = Board()
            board.points = [tuple(pt) for pt in rect]
            board.area = area
            board.center = (int(np.mean(rect[:, 0])), int(np.mean(rect[:, 1])))
            boards.append(board)
        
        return boards

    def select_board(self, boards):
        return max(boards, key=lambda b: b.area) if boards else None

    def tf_point(self, board, frame):    
        h, w = frame.shape[:2]
        self.frame_center = (w // 2, h // 2)
        
        if board is not None:
            self.board_center = board.center
            cx = board.center[0] - w / 2
            cy = board.center[1] - h / 2
            board.camera_center = (cx, cy)
            self.camera_center_offset = board.camera_center
        else:
            self.board_center = None
            self.camera_center_offset = None

    def detect(self, frame):
        binary = self.process(frame)
        boards = self.find_boards(binary)
        board = self.select_board(boards)
        self.tf_point(board, frame)
        self.relative_board = board

        if self.use_pnp and board is not None and len(board.points) == 4:
            self.pnp.solve(board.points)
        else:
            self.pnp.position = None
            self.pnp.yaw = None
            self.pnp.pitch = None
            self.pnp.distance = None
            self.pnp.center_projected = None
            self.pnp.center_error = None

        return binary, board

    def draw_boards(self, frame, show_coords=True):
        result = frame.copy()
      
        # ===== 绘制帧中心（红色） =====
        cv2.circle(result, self.frame_center, 5, (0, 0, 255), -1)
        cv2.putText(result, "FC", (self.frame_center[0] + 10, self.frame_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        # 十字线
        cv2.line(result, (self.frame_center[0]-20, self.frame_center[1]), 
                 (self.frame_center[0]+20, self.frame_center[1]), (0, 0, 255), 1)
        cv2.line(result, (self.frame_center[0], self.frame_center[1]-20), 
                 (self.frame_center[0], self.frame_center[1]+20), (0, 0, 255), 1)
            
        if self.relative_board is None:
            cv2.putText(result, "No Board", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result
        
        board = self.relative_board
        
        # ===== 绘制四边形边框（蓝色）=====
        pts = np.array(board.points, dtype=np.int32)
        cv2.polylines(result, [pts], True, (255, 0, 0), 2)
        
        # ===== 绘制角点标签 =====
        labels = ['0:TL', '1:TR', '2:BR', '3:BL']
        colors = [(0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255)]
        
        for i, (pt, label, color) in enumerate(zip(board.points, labels, colors)):
            pt_int = (int(pt[0]), int(pt[1]))
            cv2.circle(result, pt_int, 6, color, -1)
            cv2.putText(result, label, (pt_int[0]+8, pt_int[1]-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 2)
        
        # ===== 绘制检测到的板子中心（绿色实心）=====
        if self.board_center is not None:
            cv2.circle(result, self.board_center, 10, (0, 255, 0), -1)
            cv2.putText(result, "DET", (self.board_center[0] + 15, self.board_center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # ===== 【关键】绘制PNP投影中心（蓝色十字）=====
        if self.pnp.center_projected is not None:
            cp = self.pnp.center_projected
            cp_int = (int(cp[0]), int(cp[1]))
            
            # 蓝色十字标记
            cv2.drawMarker(result, cp_int, (255, 0, 0), 
                          markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(result, "PNP", (cp_int[0] + 15, cp_int[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # ===== 连接检测中心和PNP中心（白色线）=====
            if self.board_center is not None:
                cv2.line(result, self.board_center, cp_int, (255, 255, 255), 1)
                
                # 显示偏差
                cx_screen = (self.board_center[0] + cp_int[0]) // 2
                cy_screen = (self.board_center[1] + cp_int[1]) // 2
                cv2.putText(result, f"err:{self.pnp.center_error:.1f}px", 
                           (cx_screen - 40, cy_screen - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # ===== 显示坐标信息 =====
        if show_coords and board.camera_center:
            coord_text = f"({board.camera_center[0]:.1f}, {board.camera_center[1]:.1f})"
            cv2.putText(result, coord_text, (board.center[0] - 60, board.center[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
        cv2.putText(result, "Board Found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result