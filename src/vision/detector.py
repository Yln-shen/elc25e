# detector.py - 添加PNP中心显示
import cv2
import numpy as np
import math
import time
from src.vision.pnp import PNPSolver
from src.vision.camera import Camera

RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180

class Board:
    def __init__(self): 
        self.points = []
        self.center = None
        self.camera_center = None

class Detector:
    def __init__(self, rectangle_min_area, rectangle_max_area, pnp_solver=None):
        self.rectangle_max_area = rectangle_max_area
        self.rectangle_min_area = rectangle_min_area
        self.relative_board = None
        self.boards = []
        self.frame_center = None
        self.board_center = None
        self.camera_center_offset = None
        self.pnp = pnp_solver if pnp_solver else PNPSolver()

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
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
        board_contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # 首先尝试寻找内轮廓
        inner_contours = []
        for i, contour in enumerate(board_contours):
            if hierarchy[0][i][3] != -1:  # 有父轮廓的轮廓（内轮廓）
                inner_contours.append((i, contour))
        
        # 如果没有内轮廓，则使用外轮廓（无父轮廓的轮廓）
        target_contours = inner_contours if inner_contours else [
            (i, c) for i, c in enumerate(board_contours) if hierarchy[0][i][3] == -1
        ]
        
        # ===== 注意：这里要有 for 循环，缩进要对！=====
        for i, contour in target_contours:
            area = cv2.contourArea(contour)  # ← 这行之前漏了！
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
            
            if len(np.unique(rect, axis=0)) != 4:
                continue
            
            w1 = np.linalg.norm(rect[0] - rect[1])  # 上边
            w2 = np.linalg.norm(rect[3] - rect[2])  # 下边
            h1 = np.linalg.norm(rect[0] - rect[3])  # 左边
            h2 = np.linalg.norm(rect[1] - rect[2])  # 右边
            
            w = (w1 + w2) / 2  # 宽度（水平方向）
            h = (h1 + h2) / 2  # 高度（垂直方向）
            
            if w == 0 or h == 0:
                continue
            
            # 板子应该是横向的，宽度 > 高度
            if w < h:
                continue
                
            # 宽高比检查
            aspect_ratio = w / h
            if not (1.0 <= aspect_ratio <= 1.7):
                continue
            
            board = Board()
            board.points = [tuple(pt) for pt in rect]
            board.area = area
            board.center = (int(np.mean(rect[:, 0])), int(np.mean(rect[:, 1])))
            boards.append(board)
        
        return boards
    def select_board(self, boards):
        """选择中心离画面中心最近的板子"""
        if not boards:
            return None
        
        if self.frame_center is None:
            # 如果画面中心还没设置，回退到选择面积最小的
            return min(boards, key=lambda b: b.area)
        
        # 计算每个板子中心到画面中心的距离，选最近的
        return min(boards, key=lambda b: 
            (b.center[0] - self.frame_center[0])**2 + (b.center[1] - self.frame_center[1])**2
        )
    
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

        if self.pnp and board is not None:
            self.pnp.solve(board.points)
        else:
            # 清空 PNP 结果
            self.pnp.position = None
            self.pnp.distance = None
            self.pnp.yaw = None
            self.pnp.pitch = None
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
    
if __name__ == "__main__":
    # 测试代码
    try:
        cam = Camera(index=3, width=640, height=480, fps=120)
    except Exception as e:
        print(f"摄像头初始化失败: {e}，尝试默认摄像头...")
        cam = Camera(index=1)

    pnp = PNPSolver()
    detector = Detector(rectangle_min_area=100, rectangle_max_area=500000, pnp_solver=pnp)
    
    fps = 0
    fps_last = 0
    fps_timer = time.time()

    while True:
        ret, frame = cam.read()
        if not ret:
            print("无法获取图像")
            break

        fps += 1
        if time.time() - fps_timer >= 1.0:
            fps_last = fps
            fps = 0
            fps_timer = time.time()
        print(f"FPS: {fps_last}")
        
        binary, board = detector.detect(frame)
        result = detector.draw_boards(frame)
        
        cv2.imshow("Result", result)
        cv2.imshow("Binary", binary)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cam.cam.release()
    cv2.destroyAllWindows()