# src/vision/tracker.py
import math
import time
import cv2
import numpy as np
from collections import deque
from .Kalman import AdaptiveEKF1D
from .pnp import PNPSolver
from .detector import Detector
from .camera import Camera

RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180


def time_diff(last_time=[None]):
    """计算两次调用之间的时间差（秒）"""
    current_time = time.time_ns()
    if last_time[0] is None:
        last_time[0] = current_time
        return 1e-9
    else:
        diff = current_time - last_time[0]
        last_time[0] = current_time
        return diff / 1e9


class Tracker:
    """
    3D 物理坐标跟踪器 (三个独立1D自适应EKF)
    
    输入: PnP 解算的 tvec (x, y, z) — 靶心在相机坐标系下的位置 (米)
    输出: 滤波后的 (x, y, z) 以及对应的 yaw, pitch 角度
    
    状态: 对 x, y, z 分别滤波
    模型: 匀速 (Constant Velocity)
    特点: 自适应Q，算力极低
    """
    def __init__(self, use_kf=True, frame_add=35,
                 qx=0.5, qy=0.5, qz=0.3, r=1.0):
        """
        参数:
            use_kf: 是否启用卡尔曼滤波
            frame_add: 丢失后继续预测的帧数
            qx/qy/qz: 各轴的过程噪声基值（越大越信任预测）
            r: 测量噪声（越大越信任预测）
        """
        self.use_kf = use_kf
        self.frame_add = frame_add
        
        # ===== 跟踪状态 =====
        self.lost = 0
        self.predict = False
        self.if_find = False
        
        # ===== 三个独立的1D自适应EKF =====
        self.kf_cx = AdaptiveEKF1D(Q_base=4, R=0.01)
        self.kf_cy = AdaptiveEKF1D(Q_base=4, R=0.01)
        self.kf_cz = AdaptiveEKF1D(Q_base=3, R=0.01)
        
        # ===== 数据存储 (3D物理坐标) =====
        self.position_filtered = None  # (x, y, z) 米
        self.position_raw = None       # (x, y, z) 米
        
        # ===== 轨迹存储 (用于绘制) =====
        self.trajectory = deque(maxlen=30)
        
        # ===== 预测状态 =====
        self.predicted_position = None
        
        # ===== 角度 =====
        self.yaw_filtered = 0.0
        self.pitch_filtered = 0.0
        
        # ===== 画面中心像素 (仅用于 debug 绘制) =====
        self.frame_center = (640 // 2, 480 // 2)
        
        # ===== 投影参数 =====
        self.K_proj = None
        self.use_pinhole_proj = False

    # ========== 核心：3D 物理坐标滤波 ==========
    def track(self, tvec):
        """
        3D 域跟踪：对 PnP 解算的 tvec (x, y, z) 进行滤波
        
        参数:
            tvec: (x, y, z) 元组 — 靶心在相机坐标系下的位置 (米)
                  或 None (丢失时)
        
        返回:
            (x, y, z): 滤波后的三维坐标 (米)，或 None
        """
        dt = time_diff()
        
        # ==========================================
        # 情况1：没有检测到目标
        # ==========================================
        if tvec is None:
            self.position_raw = None
            
            if self.use_kf:
                self.lost += 1
                
                if self.lost <= self.frame_add and self.predict:
                    # ----- 预测模式：只预测，不更新 -----
                    pred_x = self.kf_cx.predict(dt)
                    pred_y = self.kf_cy.predict(dt)
                    pred_z = self.kf_cz.predict(dt)
                    self.position_filtered = (pred_x, pred_y, pred_z)
                    self.predicted_position = self.position_filtered
                    self.if_find = True
                    self.trajectory.append(self.position_filtered)
                    return self.position_filtered
                else:
                    # ----- 完全丢失 -----
                    self.reset_kf()
                    self.lost = 0
                    self.predict = False
                    self.if_find = False
                    self.position_filtered = None
                    self.predicted_position = None
                    return None
            else:
                self.if_find = False
                self.position_filtered = None
                return None
        
        # ==========================================
        # 情况2：检测到目标，tvec = (x, y, z)
        # ==========================================
        else:
            self.predict = True
            self.if_find = True
            self.lost = 0
            self.predicted_position = None
            
            meas_x, meas_y, meas_z = tvec
            self.position_raw = (meas_x, meas_y, meas_z)
            
            if self.use_kf:
                # 首次检测：直接初始化三个滤波器
                if not self.kf_cx.is_initialized:
                    self.kf_cx.set_initial_state(meas_x)
                    self.kf_cy.set_initial_state(meas_y)
                    self.kf_cz.set_initial_state(meas_z)
                    self.position_filtered = (meas_x, meas_y, meas_z)
                else:
                    # 正常 EKF 流程: 先预测，再更新
                    self.kf_cx.predict(dt)
                    self.kf_cy.predict(dt)
                    self.kf_cz.predict(dt)
                    filt_x = self.kf_cx.update(meas_x)
                    filt_y = self.kf_cy.update(meas_y)
                    filt_z = self.kf_cz.update(meas_z)
                    self.position_filtered = (filt_x, filt_y, filt_z)
            else:
                self.position_filtered = (meas_x, meas_y, meas_z)
            
            if self.position_filtered is not None:
                self.trajectory.append(self.position_filtered)
            
            return self.position_filtered

    # ========== 从 3D 物理坐标直接算角度 ==========
    def _xyz_to_yaw_pitch(self, x, y, z):
        """
        从靶心在相机坐标系下的 (x, y, z) 计算云台角度
        
        yaw   = arctan(x / z)   — 水平偏转角
        pitch = arctan(-y / z)  — 垂直俯仰角 (y向上为正，pitch抬头为正)
        """
        if z is None or abs(z) < 1e-6:
            return 0.0, 0.0
        
        yaw = math.atan(x / z) * RAD2DEG
        pitch = math.atan(-y / z) * RAD2DEG
        
        return yaw, pitch
    
    def get_yaw_pitch(self):
        """从滤波后的 3D 坐标计算 (yaw, pitch)"""
        if self.position_filtered is None:
            return 0.0, 0.0
        
        x, y, z = self.position_filtered
        yaw, pitch = self._xyz_to_yaw_pitch(x, y, z)
        self.yaw_filtered = yaw
        self.pitch_filtered = pitch
        return yaw, pitch
    
    def get_raw_yaw_pitch(self):
        """从原始 PnP 测量值计算 (yaw, pitch)（用于对比）"""
        if self.position_raw is None:
            return 0.0, 0.0
        
        x, y, z = self.position_raw
        return self._xyz_to_yaw_pitch(x, y, z)
    
    def reset_kf(self):
        """重置所有滤波器"""
        self.kf_cx.reset()
        self.kf_cy.reset()
        self.kf_cz.reset()
        self.position_filtered = None
        self.predicted_position = None
        self.trajectory.clear()

    # ========== Debug 绘制 (把 3D 坐标投影回 2D 屏幕) ==========
    def set_projection_params(self, K, img_width, img_height):
        """
        设置投影参数，用于把 3D 物理坐标投影回 2D 像素以绘制调试信息。
        
        参数:
            K: 相机内参矩阵 (3x3)
            img_width: 图像宽度
            img_height: 图像高度
        """
        self.K_proj = K
        self.img_width = img_width
        self.img_height = img_height
        self.use_pinhole_proj = True
    
    def _project_3d_to_2d(self, x, y, z):
        """
        把相机坐标系下的 (x, y, z) 投影到像素坐标。
        """
        if z is None or abs(z) < 1e-6:
            return None
        
        if self.use_pinhole_proj and self.K_proj is not None:
            # 使用标定好的内参矩阵
            pt_3d = np.array([[x], [y], [z]], dtype=np.float32)
            pt_2d = self.K_proj @ pt_3d
            u = pt_2d[0, 0] / pt_2d[2, 0]
            v = pt_2d[1, 0] / pt_2d[2, 0]
            return (int(u), int(v))
        else:
            # 简单小孔模型，建议调用 set_projection_params()
            f = self.frame_center[0]
            u = int(self.frame_center[0] + f * x / z)
            v = int(self.frame_center[1] + f * y / z)
            return (u, v)
    
    def draw_debug(self, frame):
        """
        在图像上绘制调试信息（精简版）
        保留：历史轨迹、滤波后位置、预测位置、信息面板
        删除：原始观测值（RAW点）
        """
        result = frame.copy()
        h, w = result.shape[:2]
        
        # ===== 1. 绘制历史轨迹 (橙色) =====
        if len(self.trajectory) > 1:
            pts = []
            for pos in self.trajectory:
                if pos is None:
                    continue
                pt = self._project_3d_to_2d(pos[0], pos[1], pos[2])
                if pt is not None and 0 <= pt[0] < w and 0 <= pt[1] < h:
                    pts.append(pt)
            
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    cv2.line(result, pts[i-1], pts[i], (0, 165, 255), 2)
        
        # ===== 2. 绘制滤波后的位置 (青色) =====
        # if self.position_filtered is not None:
        #     filt_pt = self._project_3d_to_2d(*self.position_filtered)
        #     if filt_pt is not None and 0 <= filt_pt[0] < w and 0 <= filt_pt[1] < h:
        #         cv2.circle(result, filt_pt, 10, (0, 255, 255), -1)
        #         cv2.putText(result, "FILT", (filt_pt[0] + 12, filt_pt[1] - 5),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
        
        # ===== 3. 绘制预测位置 (粉色叉号) =====
        if self.predicted_position is not None:
            pred_pt = self._project_3d_to_2d(*self.predicted_position)
            if pred_pt is not None and 0 <= pred_pt[0] < w and 0 <= pred_pt[1] < h:
                cv2.drawMarker(result, pred_pt, (255, 0, 255),
                            markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
                cv2.putText(result, "PRED", (pred_pt[0] + 12, pred_pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        # ===== 4. 删除RAW点（已注释掉） =====
        # RAW点不再显示在画面上
        
        # ===== 5. 信息面板（删除多余信息，只保留必要的） =====


        
        # 只显示距离
        # if self.position_filtered is not None:
        #     x, y, z = self.position_filtered
        #     dist = math.sqrt(x*x + y*y + z*z)
        #     cv2.putText(result, f"Dist: {dist:.3f} m", 
        #             (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        #     y_offset += line_h
        
        # 删除RAW Y/P对比（已移到终端打印）
        # 删除Residual（已删除）
        
        return result


# ============================================================================
# 测试主程序 - 集成 detector.py + tracker.py
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tracker 测试程序')
    parser.add_argument('--qx', type=float, default=0.5, help='X轴过程噪声 (默认0.5)')
    parser.add_argument('--qy', type=float, default=0.5, help='Y轴过程噪声 (默认0.5)')
    parser.add_argument('--qz', type=float, default=0.3, help='Z轴过程噪声 (默认0.3)')
    parser.add_argument('--r', type=float, default=1.0, help='测量噪声 (默认1.0)')
    parser.add_argument('--no-kf', action='store_true', help='禁用卡尔曼滤波')
    parser.add_argument('--frame-add', type=int, default=35, help='丢失后预测帧数 (默认35)')
    parser.add_argument('--camera-index', type=int, default=3, help='摄像头索引 (默认3)')
    parser.add_argument('--save-video', action='store_true', help='保存视频')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Tracker 测试程序")
    print(f"参数: Qx={args.qx}, Qy={args.qy}, Qz={args.qz}, R={args.r}")
    print(f"卡尔曼滤波: {'关闭' if args.no_kf else '开启'}")
    print(f"预测帧数: {args.frame_add}")
    print("=" * 60)
    
    # ===== 1. 初始化摄像头 =====
    try:
        cam = Camera(index=args.camera_index, width=640, height=480, fps=120)
        print(f"摄像头 {args.camera_index} 初始化成功")
    except Exception as e:
        print(f"摄像头 {args.camera_index} 初始化失败: {e}，尝试默认摄像头...")
        cam = Camera(index=1)
    
    # ===== 2. 初始化 Detector 和 PNP =====
    pnp = PNPSolver()
    detector = Detector(rectangle_min_area=100, rectangle_max_area=500000, pnp_solver=pnp)
    
    # ===== 3. 初始化 Tracker =====
    tracker = Tracker(
        use_kf=not args.no_kf,
        frame_add=args.frame_add,
        qx=args.qx,
        qy=args.qy,
        qz=args.qz,
        r=args.r
    )
    
    # 设置投影参数
    tracker.set_projection_params(pnp.camera_matrix, 640, 480)
    
    # # ===== 4. 视频保存 =====
    # video_writer = None
    # if args.save_video:
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     video_writer = cv2.VideoWriter('tracker_test.avi', fourcc, 30.0, (640, 480))
    
    # ===== 5. 统计信息 =====
    fps = 0
    fps_last = 0
    fps_timer = time.time()
    frame_count = 0
    lost_count = 0
    predict_count = 0
    tracking_count = 0
    
    # ===== 6. 创建窗口 =====
    cv2.namedWindow("Tracker Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracker Test", 640, 480)
    cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Binary", 640, 480)
    
    print("按 'q' 退出")
    print("按 'r' 重置滤波器")
    print("按 'h' 显示/隐藏调试信息")
    print("-" * 60)
    
    show_debug = True
    debug_info_y = 460
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("无法获取图像")
            break
        
        frame_count += 1
        
        # ===== 7. 检测 =====
        binary, board = detector.detect(frame)
        
        # ===== 8. 提取 tvec =====
        tvec = None
        if board is not None:
            if hasattr(detector.pnp, 'tvec') and detector.pnp.tvec is not None:
                tvec = detector.pnp.tvec
                # 如果是 tuple，保持原样
                if isinstance(tvec, tuple):
                    pass
                elif isinstance(tvec, np.ndarray):
                    tvec = tuple(tvec.flatten())
        
        # ===== 9. 跟踪 =====
        filtered_pos = tracker.track(tvec)
        
        # 统计
        if filtered_pos is not None:
            if tracker.lost > 0 and tracker.lost <= tracker.frame_add:
                predict_count += 1
            else:
                tracking_count += 1
        else:
            lost_count += 1
        
        # ===== 10. 绘制 =====
        result = detector.draw_boards(frame)
        
        # 叠加 Tracker 调试信息
        if show_debug:
            result = tracker.draw_debug(result)
        else:
            # 只显示简单的状态
            status_text = "TRACKING" if tracker.if_find else "LOST"
            color = (0, 255, 0) if tracker.if_find else (0, 0, 255)
            cv2.putText(result, f"Status: {status_text}", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # ===== 11. FPS =====
        fps += 1
        if time.time() - fps_timer >= 1.0:
            fps_last = fps
            fps = 0
            fps_timer = time.time()
        cv2.putText(result, f"FPS: {fps_last}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ===== 12. 显示参数 =====
        if show_debug:
            param_text = f"Qx={args.qx} Qy={args.qy} Qz={args.qz} R={args.r}"
            cv2.putText(result, param_text, (10, debug_info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(result, f"Lost: {lost_count} Pred: {predict_count} Track: {tracking_count}",
                       (10, debug_info_y + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # ===== 13. 显示 =====
        cv2.imshow("Tracker Test", result)
        cv2.imshow("Binary", binary)
        
        # # ===== 14. 保存视频 =====
        # if video_writer is not None:
        #     video_writer.write(result)
        
        # ===== 15. 键盘控制 =====
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracker.reset_kf()
            print("滤波器已重置")
        elif key == ord('h'):
            show_debug = not show_debug
            print(f"调试信息: {'显示' if show_debug else '隐藏'}")
    
    # ===== 16. 统计报告 =====
    print("-" * 60)
    print("统计报告:")
    print(f"  总帧数: {frame_count}")
    print(f"  跟踪帧数: {tracking_count}")
    print(f"  预测帧数: {predict_count}")
    print(f"  丢失帧数: {lost_count}")
    if frame_count > 0:
        print(f"  跟踪率: {tracking_count/frame_count*100:.1f}%")
        print(f"  预测率: {predict_count/frame_count*100:.1f}%")
        print(f"  丢失率: {lost_count/frame_count*100:.1f}%")
    print("=" * 60)
    
    # ===== 17. 清理 =====
    # if video_writer is not None:
    #     video_writer.release()
    cam.cam.release()
    cv2.destroyAllWindows()