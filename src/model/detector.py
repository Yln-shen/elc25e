# detector.py - 完整版
import cv2
import numpy as np
import math
from .pnp import PNPSolver

RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180

class Board:
    def __init__(self): 
        self.points = []              # 四个角点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        self.center = None            # 中心点坐标 (x, y)
        self.camera_center = None     # 相对于摄像头中心的像素偏移 (x, y)

class Detector:
    def __init__(self, rectangle_min_area, rectangle_max_area):
        self.rectangle_max_area = rectangle_max_area
        self.rectangle_min_area = rectangle_min_area

        self.relative_board = None
        self.boards = []
        self.frame_center = None
        self.board_center = None
        self.camera_center_offset = None  # 板子相对摄像头中心的像素偏移

        self.pnp = PNPSolver(target_width=0.2, target_height=0.15)

    def process(self, frame):
        """处理图像，生成掩膜（使用 Otsu 二值化）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def find_boards(self, binary):
        """查找四边形,并创建板子"""
        boards = []
        rectangle_contours = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        
        for contour in rectangle_contours:
            area = cv2.contourArea(contour)
            if area < self.rectangle_min_area or area > self.rectangle_max_area:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            points = approx.reshape(4, 2)
            # 按左上、左下、右下、右上排序
            sum_xy = points.sum(axis=1)
            diff_xy = points[:, 0] - points[:, 1]
            sorted_points = [
                points[np.argmin(sum_xy)],   # 左上
                points[np.argmin(diff_xy)],  # 左下
                points[np.argmax(sum_xy)],   # 右下
                points[np.argmax(diff_xy)]   # 右上
            ]

            width = np.linalg.norm(sorted_points[0] - sorted_points[1])
            height = np.linalg.norm(sorted_points[1] - sorted_points[2])
            ratio = max(width, height) / min(width, height)
            
            if not (1.1 <= ratio <= 1.6):
                continue

            board = Board()
            board.points = [tuple(pt) for pt in sorted_points]
            board.area = area
            cx = int(sum(p[0] for p in board.points) / 4)
            cy = int(sum(p[1] for p in board.points) / 4)
            board.center = (cx, cy)
            boards.append(board)
        return boards

    def select_board(self, boards):
        """得到当前帧里一个正确的板子"""
        if len(boards) == 0:
            return None
        return max(boards, key=lambda board: board.area)

    def tf_point(self, board, frame):    
        """转换坐标（以摄像头中心为原点）"""
        h, w = frame.shape[:2]
        image_center_x = w / 2
        image_center_y = h / 2
        self.frame_center = (int(image_center_x), int(image_center_y))
        
        if board is not None:
            self.board_center = board.center
            camera_center_x = board.center[0] - image_center_x
            camera_center_y = board.center[1] - image_center_y
            board.camera_center = (camera_center_x, camera_center_y)
            self.camera_center_offset = board.camera_center
        else:
            self.board_center = None
            self.camera_center_offset = None

    def detect(self, frame):
        """主检测函数"""
        binary = self.process(frame)
        boards = self.find_boards(binary)
        board = self.select_board(boards)
        self.tf_point(board, frame)
        self.relative_board = board

        # PNP调用
        if board is not None and len(board.points) == 4:
            self.pnp.solve(board.points)
        else:
            self.pnp.position = None
            self.pnp.yaw = None
            self.pnp.pitch = None
            self.pnp.distance = None

        return binary, board

    def draw_boards(self, frame, show_coords=True):
        """在图像上绘制检测结果"""
        result = frame.copy()
      
        # 绘制图像中心（红色点）
        cv2.circle(result, self.frame_center, 5, (0, 0, 255), -1)
        cv2.putText(result, "Frame Center", (self.frame_center[0] + 10, self.frame_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        if self.board_center is not None:
            # 绘制板子中心（绿色点）
            cv2.circle(result, self.board_center, 8, (0, 255, 0), -1)
            cv2.putText(result, "Board center", (self.board_center[0] + 10, self.board_center[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        if self.relative_board is None:
            cv2.putText(result, "No Board", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result
        
        board = self.relative_board
        
        # 绘制四边形边框
        pts = np.array(board.points, dtype=np.int32)
        cv2.polylines(result, [pts], True, (255, 0, 0), 2)
        
        # 显示板子中心相对摄像头中心的像素偏移
        if show_coords and board.camera_center:
            coord_text = f"({board.camera_center[0]:.1f}, {board.camera_center[1]:.1f})"
            cv2.putText(result, coord_text, (board.center[0] - 60, board.center[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 显示PNP结果
        if self.pnp.yaw is not None:
            cv2.putText(result, f"PNP Yaw: {self.pnp.yaw:.1f}  Pitch: {self.pnp.pitch:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(result, f"Dist: {self.pnp.distance:.2f}m", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(result, "Board Found (PNP)", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(result, "Board Found", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result


if __name__ == '__main__':
    from camera import Camera
    from ser import Serial
    from tracker import Tracker
    import time
    
    try:
        cam = Camera(index=0)
    except Exception as e:
        print(f"摄像头初始化失败: {e}")
        cam = Camera(index=0)

    detector = Detector(
        rectangle_max_area=60000,
        rectangle_min_area=1000,
    )

    tracker = Tracker(
        vfov=100,
        img_width=640,
        use_kf=True,
        frame_add=35
    )

    serial_port = Serial(
        port='/dev/ttyACM0',
        baudrate=115200,
        timeout=1,
        write_timeout=1
    )

    fps_counter = 0
    fps_display = 0
    fps_timer = time.time()
    capture_count = 0

    print("按 'q' 退出，按 's' 保存截图")
    print("-" * 40)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("无法获取图像")
                break

            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_timer = time.time()

            binary, board = detector.detect(frame)
            result = detector.draw_boards(frame, show_coords=True)

            camera_center_offset = detector.camera_center_offset

            if camera_center_offset is not None:
                yaw, pitch = tracker.track(camera_center_offset)
            else:
                yaw, pitch = tracker.track(None)

            if tracker.if_find and camera_center_offset is not None:
                serial_port.send_data(yaw=yaw, pitch=pitch)
                if abs(yaw) > 0.01 or abs(pitch) > 0.01:
                    print(f"\r板子坐标: ({camera_center_offset[0]:>7.1f}, {camera_center_offset[1]:>7.1f})  "
                          f"偏航: {yaw:>6.1f}°  俯仰: {pitch:>6.1f}°  FPS: {fps_display}", end="")
                else:
                    print(f"\r板子坐标: ({camera_center_offset[0]:>7.1f}, {camera_center_offset[1]:>7.1f})  "
                          f"已对准中心  FPS: {fps_display}", end="")
            elif tracker.if_find and camera_center_offset is None:
                print(f"\r预测跟踪中...  偏航: {yaw:>6.1f}°  俯仰: {pitch:>6.1f}°  FPS: {fps_display}", end="")
            elif board is not None and board.camera_center is not None:
                print(f"\r板子坐标: ({camera_center_offset[0]:>7.1f}, {camera_center_offset[1]:>7.1f})  "
                      f"等待跟踪...  FPS: {fps_display}", end="")
            else:
                print(f"\r未检测到板子  FPS: {fps_display}", end="")

            cv2.putText(result, f"FPS: {fps_display}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if tracker.if_find:
                status_text = "Track: OK" if camera_center_offset is not None else "Track: PREDICT"
                status_color = (0, 255, 0) if camera_center_offset is not None else (0, 255, 255)
            else:
                status_text = "Track: LOST"
                status_color = (0, 0, 255)
            
            cv2.putText(result, status_text, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            if tracker.if_find:
                cv2.putText(result, f"Yaw: {yaw:.1f}  Pitch: {pitch:.1f}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow('Mask', binary)
            cv2.imshow('Detection', result)

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
        cam.cam.release()
        serial_port.close()
        cv2.destroyAllWindows()
        print("资源已释放")