# detector.py - 添加PNP中心显示
import cv2
import numpy as np
import math
import time
from src.vision.pnp import PNPSolver
from src.vision.camera import Camera

RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180


def angle_between_lines(line1, line2):
    """
    计算两条线段之间的夹角（度）
    
    参数:
        line1: ((x1,y1), (x2,y2)) 线段1的两个端点
        line2: ((x1,y1), (x2,y2)) 线段2的两个端点
    
    返回:
        夹角（度），范围 0-90°
    """
    p1, p2 = line1
    p3, p4 = line2
    
    # 计算方向向量
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (p4[0] - p3[0], p4[1] - p3[1])
    
    # 计算向量长度
    len1 = np.sqrt(v1[0]**2 + v1[1]**2)
    len2 = np.sqrt(v2[0]**2 + v2[1]**2)
    
    if len1 < 1e-10 or len2 < 1e-10:
        return 0.0
    
    # 计算夹角余弦值
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    cos_angle = dot / (len1 * len2)
    
    # 限制范围，避免数值误差
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    # 弧度转角度，取锐角（0-90°）
    angle_rad = math.acos(abs(cos_angle))
    return angle_rad * RAD2DEG


class Board:
    def __init__(self): 
        self.points = []
        self.center = None
        self.parallel_score = None  # 平行度评分（越小越平行）
        self.parallel_angles = None  # 存储对边夹角


class Detector:
    def __init__(self, rectangle_min_area, rectangle_max_area, pnp_solver=None):
        self.rectangle_max_area = rectangle_max_area
        self.rectangle_min_area = rectangle_min_area
        self.relative_board = None
        self.boards = []
        self.frame_center = None
        self.board_center = None
        self.camera_center = None
        self.pnp = pnp_solver if pnp_solver else PNPSolver()

    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def order_points(self, pts):
        """排序角点：左上→右上→右下→左下"""
        s = pts.sum(axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        
        d = np.diff(pts, axis=1).flatten()
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def compute_parallel_score(self, board):
        """
        计算矩形的整体平行度评分
        
        评分规则：
            - 对边角度差越小，平行度越高
            - 四条边的整体平行度 = 上边与下边夹角 + 左边与右边夹角
            - 评分越小表示越平行
        
        返回:
            parallel_score: float，越小越平行
            angles: (top_bottom_angle, left_right_angle)
        """
        points = board.points  # [tl, tr, br, bl]
        
        # 提取四条边
        top_line = (points[0], points[1])     # 上边：tl -> tr
        right_line = (points[1], points[2])   # 右边：tr -> br
        bottom_line = (points[3], points[2])  # 下边：bl -> br
        left_line = (points[0], points[3])    # 左边：tl -> bl
        
        # 计算对边夹角
        top_bottom_angle = angle_between_lines(top_line, bottom_line)
        left_right_angle = angle_between_lines(left_line, right_line)
        
        # 整体平行度 = 两条对边的夹角之和
        # 理论上完美矩形：top_bottom_angle=0, left_right_angle=0
        parallel_score = top_bottom_angle + left_right_angle
        
        # 保存角度信息
        board.parallel_angles = (top_bottom_angle, left_right_angle)
        
        return parallel_score

    def find_boards(self, binary):
        boards = []
        board_contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # 首先尝试寻找内轮廓
        inner_contours = []
        for i, contour in enumerate(board_contours):
            if hierarchy[0][i][3] != -1:
                inner_contours.append((i, contour))
        
        # 如果没有内轮廓，则使用外轮廓
        target_contours = inner_contours if inner_contours else [
            (i, c) for i, c in enumerate(board_contours) if hierarchy[0][i][3] == -1
        ]
        
        for i, contour in target_contours:
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
            
            if len(np.unique(rect, axis=0)) != 4:
                continue
            
            w1 = np.linalg.norm(rect[0] - rect[1])
            w2 = np.linalg.norm(rect[3] - rect[2])
            h1 = np.linalg.norm(rect[0] - rect[3])
            h2 = np.linalg.norm(rect[1] - rect[2])
            
            w = (w1 + w2) / 2
            h = (h1 + h2) / 2
            
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
            board.center = (int(np.mean(rect[:, 0])), int(np.mean(rect[:, 1])))
            
            # ===== 计算平行度评分 =====
            board.parallel_score = self.compute_parallel_score(board)
            
            boards.append(board)
        
        return boards
    
    def select_board(self, boards):
        """
        选择矩形：优先选平行度最高的，再选最靠近画面中心的
        
        策略：
            1. 先按平行度评分排序（越小越平行）
            2. 从平行度最好的前N个中，选最靠近画面中心的
            3. 如果只有一个，直接选
        """
        if not boards:
            return None
        
        # ===== 策略1：先选平行度最高的 =====
        # 按平行度评分升序排序（越平行评分越小）
        sorted_boards = sorted(boards, key=lambda b: b.parallel_score)
        
        # 取平行度最好的前几个（如果总数>=3，取前3个；否则取全部）
        top_n = min(3, len(sorted_boards))
        candidates = sorted_boards[:top_n]
        
        # 如果有多个候选，再从候选中选最靠近画面中心的
        if len(candidates) > 1 and self.frame_center is not None:
            return min(candidates, key=lambda b: 
                (b.center[0] - self.frame_center[0])**2 + 
                (b.center[1] - self.frame_center[1])**2
            )
        else:
            # 只有一个候选，直接返回
            return candidates[0]
    
    def detect(self, frame):
        binary = self.process(frame)
        boards = self.find_boards(binary)
        board = self.select_board(boards)
        
        self.relative_board = board
        
        if board is not None and len(board.points) == 4:
            image_pts = np.array(board.points, dtype=np.float32)
            pnp_result = self.pnp.solve(image_pts)
            
            if pnp_result['success']:
                self.pnp.tvec = pnp_result['tvec']
                self.pnp.rvec = pnp_result['rvec']
                self.pnp.yaw = pnp_result['yaw']
                self.pnp.pitch = pnp_result['pitch']
                self.pnp.distance = pnp_result['distance']
        else:
            self.relative_board = None
        
        return binary, board

    def draw_boards(self, frame, show_coords=True):
        result = frame.copy()
        
        h, w = frame.shape[:2]
        self.frame_center = (w // 2, h // 2)
        
        cv2.drawMarker(result, self.frame_center, (0, 0, 255), 
                    markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        
        if not hasattr(self, 'relative_board') or self.relative_board is None:
            cv2.putText(result, "No Board", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result
        
        board = self.relative_board
        
        pts = np.array(board.points, dtype=np.int32)
        cv2.polylines(result, [pts], True, (255, 0, 0), 2)
        
        for pt in board.points:
            pt_int = (int(pt[0]), int(pt[1]))
            cv2.circle(result, pt_int, 6, (0, 255, 255), -1)
        
        if board.center is not None:
            cv2.circle(result, board.center, 10, (0, 255, 0), -1)

        
        if board.center is not None:
            self.camera_center = (self.frame_center[0] - board.center[0], 
                                self.frame_center[1] - board.center[1])
        
        if hasattr(self, 'pnp') and self.pnp is not None:
            if hasattr(self.pnp, 'center_projected') and self.pnp.center_projected is not None:
                cp = self.pnp.center_projected
                
                if isinstance(cp, np.ndarray):
                    if cp.ndim == 3:
                        cp = cp[0][0]
                    elif cp.ndim == 2:
                        cp = cp[0]
                    elif cp.ndim == 1:
                        cp = cp
                
                if len(cp) >= 2:
                    cp_int = (int(cp[0]), int(cp[1]))
                    
                    cv2.drawMarker(result, cp_int, (255, 0, 0), 
                                markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                    
                    coord_text = f"({cp[0]:.1f}, {cp[1]:.1f})"
                    cv2.putText(result, coord_text, (cp_int[0] + 15, cp_int[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                    
                    if board.center is not None:
                        cv2.line(result, board.center, cp_int, (255, 255, 255), 1)
                # else:
                #     cv2.putText(result, "Invalid PNP Center", (10, 120),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # else:
            #     cv2.putText(result, "No PNP Center", (10, 120),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # ===== 左上角显示信息 =====
        y_offset = 25
        line_height = 22
        font_scale = 0.5
        thickness = 1
        
        # if hasattr(self.pnp, 'tvec') and self.pnp.tvec is not None:
        #     tvec = self.pnp.tvec
        #     if isinstance(tvec, tuple):
        #         tvec = np.array(tvec)
        #     tvec_flat = tvec.flatten()
        #     tvec_text = f"tvec: ({tvec_flat[0]:.3f}, {tvec_flat[1]:.3f}, {tvec_flat[2]:.3f})"
        #     cv2.putText(result, tvec_text, (10, y_offset),
        #                 cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
        #     y_offset += line_height
        
        if hasattr(self.pnp, 'rvec') and self.pnp.rvec is not None:
            rvec = self.pnp.rvec
            if isinstance(rvec, tuple):
                rvec = np.array(rvec)
            rvec_flat = rvec.flatten()
            rvec_text = f"rvec: ({rvec_flat[0]:.3f}, {rvec_flat[1]:.3f}, {rvec_flat[2]:.3f})"
            cv2.putText(result, rvec_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
            y_offset += line_height
        
        # if hasattr(self.pnp, 'yaw') and self.pnp.yaw is not None:
        #     cv2.putText(result, f"yaw: {self.pnp.yaw:.2f} deg", (10, y_offset),
        #                 cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        #     y_offset += line_height
        
        # if hasattr(self.pnp, 'pitch') and self.pnp.pitch is not None:
        #     cv2.putText(result, f"pitch: {self.pnp.pitch:.2f} deg", (10, y_offset),
        #                 cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        #     y_offset += line_height
        
        # if hasattr(self.pnp, 'distance') and self.pnp.distance is not None:
        #     cv2.putText(result, f"distance: {self.pnp.distance:.3f} m", (10, y_offset),
        #                 cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        #     y_offset += line_height
        
        if board.center is not None and hasattr(self, 'camera_center'):
            offset_text = f"offset: ({self.camera_center[0]:.1f}, {self.camera_center[1]:.1f})"
            cv2.putText(result, offset_text, (10, y_offset + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
        
        return result


if __name__ == "__main__":
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

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 640, 480)
    cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Binary", 640, 480)

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
        
        cv2.putText(frame, f"FPS: {fps_last}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        binary, board = detector.detect(frame)
        result = detector.draw_boards(frame)
        
        cv2.imshow("Result", result)
        cv2.imshow("Binary", binary)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cam.cam.release()
    cv2.destroyAllWindows()