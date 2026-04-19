import math
import time
from Kalman import KalmanFilter
from detector import Detector

RAD2DEG = 180 / math.pi #弧度转角度
DEG2RAD = math.pi / 180 #角度转弧度


def time_diff(last_time=[None]):
    """计算两次调用之间的时间差，单位为纳秒。"""
    current_time = time.time_ns()  # 获取当前时间（单位：纳秒）

    if last_time[0] is None:  # 如果是第一次调用，更新last_time
        last_time[0] = current_time
        return 1  # 防止除零错误，返回1纳秒

    else:  # 计算时间差
        diff = current_time - last_time[0]  # 计算时间差（单位：纳秒）
        last_time[0] = current_time  # 更新上次调用时间
        return diff / 1e9  # 返回时间差（秒）

class Tracker:
    def __init__(self, vfov=100, img_width=640,use_kf = True, frame_add = 35):
        self.vfov = vfov # 视场角
        self.img_width = img_width
        self.use_kf = use_kf  # 是否使用卡尔曼滤波
        self.frame_add = frame_add  # 补帧数
        self.lost = 0  # 丢失帧计数
        self.predict = False  # 是否处于预测状态
        self.if_find = False  # 是否找到目标
        # 初始化卡尔曼滤波器
        self.kf_cx = KalmanFilter()  # x 坐标滤波器
        self.kf_cy = KalmanFilter()  # y 坐标滤波器

    def update_dt(self, dt):
        """更新卡尔曼滤波器时间步长"""
        self.kf_cx.dt = dt
        self.kf_cy.dt = dt

    def kf_predict(self):
        """执行卡尔曼滤波预测"""
        self.kf_cx.predict()
        self.kf_cy.predict()

    def get_kf_state(self):
        """获取卡尔曼滤波器当前状态"""
        return (self.kf_cx.get_state(), self.kf_cy.get_state())

    def reset_kf(self):
        """重置卡尔曼滤波器"""
        self.kf_cx.reset()
        self.kf_cy.reset()

    def kf_update(self, center):
        """更新卡尔曼滤波器状态"""
        self.kf_cx.update(center[0])
        self.kf_cy.update(center[1])

    def pixel_to_yaw_pitch(self, laser_center):
        """将像素坐标转换为偏航角和俯仰角"""
        if laser_center is None:
            return 0.0, 0.0
            
        vfov_rad = self.vfov * DEG2RAD #将场视角转为弧度
        
        # 计算焦距（像素单位）
        # 焦距 = 图像半宽 / tan(半视场角)
        focal = (self.img_width / 2) / math.tan(vfov_rad / 2)
        
        if abs(focal) < 1e-6:
            return 0.0, 0.0
        
        # 计算角度
        yaw = math.atan(laser_center[0] / focal) * RAD2DEG
        pitch = math.atan(laser_center[1] / focal) * RAD2DEG
        
        return yaw, pitch
    
    def track(self,laser_center):
        """跟踪目标,融合卡尔曼滤波"""
        dt = time_diff()  # 计算时间差

        if laser_center is None:
            if self.use_kf:
                self.lost += 1
                if self.lost <= self.frame_add and self.predict:
                    self.update_dt(dt)  # 更新时间步长
                    self.kf_predict()  # 预测下一步
                    laser_center = self.get_kf_state()  # 获取预测的中心点
                    self.if_find = True
                else:
                    print("未检测到目标")
                    self.reset_kf()  # 重置滤波器
                    self.lost = 0
                    self.predict = False
                    self.if_find = False
                    return 0, 0
            else:
                print("未检测到目标")
                self.if_find = False
                return 0, 0
        else:
            # 检测到目标
            self.predict = True
            self.if_find = True
            self.lost = 0
            if self.use_kf:
                self.update_dt(dt)  # 更新时间步长
                self.kf_update(laser_center)  # 更新滤波器
                self.kf_predict()  # 预测下一步
                laser_center = self.get_kf_state()  # 获取预测的中心点
        yaw, pitch = self.pixel_to_yaw_pitch(laser_center)
        return yaw, pitch
        