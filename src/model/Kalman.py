import numpy as np
import cv2

class KalmanFilter:
    def __init__(self, R=0.5, Q=0.01):
        self.dt = 1/30  # 默认30fps的时间步长
        self.kf = cv2.KalmanFilter(2, 1)  # 状态维度=2(位置+速度), 观测维度=1(位置)
        
        # ===== 状态转移矩阵 A =====
        # [1, dt]  表示: 新位置 = 旧位置 + 速度*dt
        # [0, 1 ]  表示: 速度不变(匀速模型)
        self.kf.transitionMatrix = np.array([[1, self.dt],
                                             [0, 1]], np.float32)
        
        # ===== 观测矩阵 H =====
        # [1, 0] 表示: 我们只能观测到位置，观测不到速度
        self.kf.measurementMatrix = np.array([[1, 0]], np.float32)
        
        # ===== 过程噪声协方差 Q =====
        # Q越大 → 越不相信模型预测 → 越相信观测值 → 响应快但噪声大
        self.kf.processNoiseCov = np.eye(2, dtype=np.float32) * Q
        
        # ===== 观测噪声协方差 R =====
        # R越大 → 越不相信观测值 → 越平滑但响应慢
        self.kf.measurementNoiseCov = np.array([[R]], np.float32)

        # ===== 初始状态 x = [位置, 速度] = [0, 0] =====
        # 【问题】初始状态是(0,0)，离真实值很远
        self.kf.statePost = np.zeros((2, 1), np.float32)
        
        # ===== 初始协方差矩阵 P =====
        # 【问题】1000表示"非常不确定初始状态"
        # 导致KF需要很多帧才能从(0,0)收敛到真实值
        self.kf.errorCovPost = np.eye(2, dtype=np.float32) * 1000
        
        # ===== 【新增】初始化标志 =====
        self.is_initialized = False

    # ===== 【新增】设置初始状态方法 =====
    def set_initial_state(self, value):
        """
        将KF的状态直接设置为观测值，跳过从0收敛的过程
        
        参数:
            value: 第一次检测到的值（位置）
        
        原理:
            statePost = [[value],    ← 位置直接设为检测值
                         [0]]        ← 速度设为0（未知，先猜0）
        """
        self.kf.statePost = np.array([[value], [0]], np.float32)
        self.is_initialized = True  # 标记已初始化

    def predict(self):
        """预测下一步状态"""
        return self.kf.predict()

    def update(self, value):
        """用观测值更新状态"""
        measurement = np.array([[value]], np.float32)
        self.kf.correct(measurement)

    def reset(self):
        """重置KF状态"""
        self.kf.statePost = np.zeros((2, 1), np.float32)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32) * 100
        self.is_initialized = False  # 【新增】重置初始化标志

    def get_state(self):
        """获取当前状态（位置值）"""
        return self.kf.statePost[0, 0]
    
# 示例用法
if __name__ == "__main__":
    dt = 0.1  # 时间步长（秒）
    kf = KalmanFilter(R=0.5, Q=0.1)

    # 模拟传入的yaw数据
    measurements = [1.0, 1.2, 1.1, 1.1, 1.3, 1.0, 0.9, 1.1, 5.0, 1.0, 1.2, 1.1, 1.1, 2.3, 1.0, 0.9, 1.1]

    for yaw in measurements:
        kf.predict()  # 进行预测
        kf.update(yaw)  # 更新状态
        predicted_yaw = kf.get_state()  # 获取预测的yaw
        print(f"预测的 yaw: {predicted_yaw:.2f}")