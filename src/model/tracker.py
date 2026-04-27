import math
import time
import cv2
from .Kalman import KalmanFilter

RAD2DEG = 180 / math.pi
DEG2RAD = math.pi / 180


def time_diff(last_time=[None]):
    """计算两次调用之间的时间差（秒）"""
    current_time = time.time_ns()

    if last_time[0] is None:
        last_time[0] = current_time
        return 1e-9  # 第一次返回1纳秒，避免除零
    else:
        diff = current_time - last_time[0]
        last_time[0] = current_time
        return diff / 1e9  # 纳秒转秒


class Tracker:
    def __init__(self, vfov=100, img_width=640, use_kf=True, frame_add=35):
        self.vfov = vfov
        self.img_width = img_width
        self.use_kf = use_kf
        self.frame_add = frame_add
        
        # ===== 跟踪状态变量 =====
        self.lost = 0          # 丢失帧计数
        self.predict = False   # 是否处于预测状态
        self.if_find = False   # 是否找到目标
        
        # ===== 卡尔曼滤波器（X和Y各一个） =====
        self.kf_cx = KalmanFilter()  # X坐标滤波器
        self.kf_cy = KalmanFilter()  # Y坐标滤波器
        
        # ===== 【新增1】KF位置存储 =====
        self.kf_position = None  # 存储滤波后的位置(x, y)

    def update_dt(self, dt):
        """
        更新KF的时间步长
        
        【重要】因为实际帧率可能变化，需要动态更新转移矩阵中的dt
        转移矩阵: [[1, dt],
                  [0, 1 ]]
        dt在[0,1]位置，表示"速度对位置的贡献"
        """
        self.kf_cx.kf.transitionMatrix[0, 1] = dt
        self.kf_cy.kf.transitionMatrix[0, 1] = dt
        self.kf_cx.dt = dt
        self.kf_cy.dt = dt

    def kf_predict(self):
        """执行KF预测"""
        self.kf_cx.predict()
        self.kf_cy.predict()

    def get_kf_state(self):
        """获取KF当前状态（位置值）"""
        return (self.kf_cx.get_state(), self.kf_cy.get_state())

    def reset_kf(self):
        """重置KF"""
        self.kf_cx.reset()
        self.kf_cy.reset()
        self.kf_position = None

    def kf_update(self, center):
        """用观测值更新KF"""
        self.kf_cx.update(center[0])
        self.kf_cy.update(center[1])

    def pixel_to_yaw_pitch(self, laser_center):
        """将像素坐标转换为偏航角和俯仰角"""
        if laser_center is None:
            return 0.0, 0.0
            
        vfov_rad = self.vfov * DEG2RAD
        focal = (self.img_width / 2) / math.tan(vfov_rad / 2)
        
        if abs(focal) < 1e-6:
            return 0.0, 0.0
        
        yaw = math.atan(laser_center[0] / focal) * RAD2DEG
        pitch = math.atan(laser_center[1] / focal) * RAD2DEG
        
        return yaw, pitch

    def track(self, laser_center):
        """
        跟踪目标，融合卡尔曼滤波
        
        三种情况：
        1. 首次检测到 → 初始化KF状态为检测值
        2. 持续检测到 → 正常KF滤波
        3. 丢失目标   → KF预测
        """
        dt = time_diff()  # 计算时间差

        # ==========================================
        # 情况1：没有检测到目标（laser_center为None）
        # ==========================================
        if laser_center is None:
            if self.use_kf:
                self.lost += 1  # 丢失计数+1
                
                # 判断：丢失帧数还在允许范围内，且之前有预测过
                if self.lost <= self.frame_add and self.predict:
                    # ----- 进入预测模式 -----
                    self.update_dt(dt)           # 更新时间步长
                    self.kf_predict()            # KF预测下一步位置
                    laser_center = self.get_kf_state()  # 获取预测位置
                    self.if_find = True          # 仍标记为"找到"
                    # 【新增2】保存预测位置，供draw_kf使用
                    self.kf_position = laser_center
                else:
                    # ----- 完全丢失 -----
                    self.reset_kf()              # 重置KF
                    self.lost = 0                # 丢失计数归零
                    self.predict = False         # 退出预测模式
                    self.if_find = False         # 标记为"丢失"
                    # 【新增3】清空位置
                    self.kf_position = None
                    return 0, 0
            else:
                # 不使用KF，直接返回
                self.if_find = False
                self.kf_position = None
                return 0, 0
        
        # ==========================================
        # 情况2：检测到目标（laser_center不为None）
        # ==========================================
        else:
            self.predict = True      # 标记：可以进行预测了
            self.if_find = True      # 标记：找到目标
            self.lost = 0            # 丢失计数归零
            
            if self.use_kf:
                # ===== 【新增4】首次检测，快速初始化 =====
                if not self.kf_cx.is_initialized:
                    """
                    首次检测到目标时：
                    - KF默认状态是(0,0)，离真实值很远
                    - 直接设置KF状态为当前检测值
                    - 跳过KF滤波，直接用原始检测值
                    - 下次检测时再启用KF
                    """
                    self.kf_cx.set_initial_state(laser_center[0])
                    self.kf_cy.set_initial_state(laser_center[1])
                    self.update_dt(dt)
                    # laser_center保持不变，使用原始检测值
                else:
                    # ===== 正常KF流程 =====
                    self.update_dt(dt)           # 更新时间步长
                    self.kf_update(laser_center) # 用检测值更新KF
                    self.kf_predict()            # KF预测
                    laser_center = self.get_kf_state()  # 获取滤波后的值
            
            # 【新增5】保存滤波后的位置，供draw_kf使用
            self.kf_position = laser_center
        
        # 转换为角度
        yaw, pitch = self.pixel_to_yaw_pitch(laser_center)
        return yaw, pitch

    # ===== 【新增6】绘制KF位置 =====
    def draw_kf(self, frame, laser_pixel):
        """
        在图像上绘制KF滤波后的位置（橙色标记）
        
        参数:
            frame: 原始图像
            laser_pixel: 激光笔在图像中的像素坐标 (x, y)
        
        注意:
            kf_position存储的是相对于激光中心的偏移量
            需要加上laser_pixel才能得到图像坐标
        """
        if self.kf_position is None:
            return frame
        
        result = frame.copy()
        
        # 转换：图像坐标 = 激光像素坐标 + KF相对偏移
        kf_x = int(laser_pixel[0] + self.kf_position[0])
        kf_y = int(laser_pixel[1] + self.kf_position[1])
        
        # 检查是否在图像范围内
        h, w = result.shape[:2]
        if kf_x < 0 or kf_x >= w or kf_y < 0 or kf_y >= h:
            return result
        
        # 橙色空心圆
        cv2.circle(result, (kf_x, kf_y), 12, (0, 165, 255), 2)  # 外圈
        # 标注
        cv2.putText(result, "KF", (kf_x + 15, kf_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        return result