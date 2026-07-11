# src/vision/Kalman.py
import numpy as np

class AdaptiveEKF1D:
    """
    1D 自适应扩展卡尔曼滤波器 (EKF)
    状态: [位置, 速度]  模型: 匀速 (Constant Velocity)
    特点: 根据观测残差自动调节 Q，快速响应突变，平滑跟踪匀速
    
    2x2矩阵求逆，计算量极低
    """
    def __init__(self, Q_base=0.5, R=1.0, dt=1/30.0):
        """
        Args:
            Q_base: 基础过程噪声 (像素/秒²)，越大响应越快
            R: 观测噪声 (像素)，取决于检测器精度
            dt: 初始时间步长 (秒)
        """
        self.dt = dt
        self.Q_base_value = Q_base
        self.R_value = R
        
        # 1. 状态向量 [位置, 速度]
        self.x = np.zeros((2, 1), dtype=np.float32)
        
        # 2. 状态协方差矩阵 P
        self.P = np.eye(2, dtype=np.float32) * 100.0
        
        # 3. 状态转移矩阵 F (匀速模型)
        self.F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
        
        # 4. 观测矩阵 H (只能看到位置)
        self.H = np.array([[1.0, 0.0]], dtype=np.float32)
        
        # 5. 观测噪声协方差 R
        self.R = np.array([[R]], dtype=np.float32)
        
        # 6. 基础过程噪声协方差 Q
        self._update_Q_base(Q_base)
        
        # 7. 状态标志
        self.is_initialized = False
        self._residual = 0.0
        self._last_measurement = 0.0
        
        # 8. 性能统计（可选）
        self.call_count = 0
        self.total_time = 0.0

    def _update_Q_base(self, Q_base):
        """更新基础过程噪声矩阵 (匀速模型)"""
        dt = self.dt
        # 速度噪声对位置的影响: Q[0,0] = q * dt²
        # 速度噪声对速度的影响: Q[1,1] = q * dt
        self.Q_base = np.array([
            [Q_base * (dt ** 2), 0.0],
            [0.0, Q_base * dt]
        ], dtype=np.float32)
        self.Q_base_value = Q_base

    def set_initial_state(self, value):
        """直接设置状态，跳过收敛期 (首次检测时调用)"""
        self.x = np.array([[value], [0.0]], dtype=np.float32)
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.is_initialized = True
        self._last_measurement = value

    def predict(self, dt=None):
        """
        预测步骤：根据匀速模型推演下一时刻状态
        
        Args:
            dt: 可选，更新时间步长 (如果帧率变化)
        
        Returns:
            float: 预测的位置
        """
        if dt is not None and dt != self.dt:
            self.dt = dt
            self.F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
            self._update_Q_base(self.Q_base_value)
        
        # 状态预测: x' = F * x
        self.x = self.F @ self.x
        # 协方差预测: P' = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q_base
        
        return self.x[0, 0]

    def update(self, measurement_value):
        """
        更新步骤：用观测值修正预测，并根据残差动态调整 Q
        
        Args:
            measurement_value: 观测值 (像素坐标)
        
        Returns:
            float: 滤波后的位置
        """
        self._last_measurement = measurement_value
        z = np.array([[measurement_value]], dtype=np.float32)
        
        # 1. 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 2. 计算观测残差 (创新)
        y = z - self.H @ self.x
        residual = y[0, 0]
        self._residual = residual
        
        # ===== 3. 自适应机制：根据残差动态调节 Q =====
        abs_residual = abs(residual)
        if abs_residual > 3.0 and self.is_initialized:
            # 突发运动/快速运动：增大 Q
            # 残差越大，Q 放大倍数越大 (最大15倍)
            scale = min(15.0, abs_residual / 2.0)
            Q_adaptive = self.Q_base * scale
        elif abs_residual > 1.0 and not self.is_initialized:
            # 初始化阶段：使用较大的 Q 快速收敛
            Q_adaptive = self.Q_base * 5.0
        else:
            # 正常跟踪：保持基础 Q (平滑)
            Q_adaptive = self.Q_base
        
        # 4. 使用自适应 Q 重新计算预测协方差
        P_pred = self.F @ self.P @ self.F.T + Q_adaptive
        
        # 5. 重新计算卡尔曼增益
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # 6. 状态更新: x = x + K * y
        self.x = self.x + K @ y
        
        # 7. 更新协方差: P = (I - K*H) * P_pred
        I = np.eye(2)
        self.P = (I - K @ self.H) @ P_pred
        
        self.is_initialized = True
        return self.x[0, 0]

    def get_state(self):
        """获取当前滤波后的位置"""
        return self.x[0, 0]
    
    def get_speed(self):
        """获取当前估计的速度 (像素/秒)"""
        return self.x[1, 0]
    
    def get_residual(self):
        """获取最近的观测残差，用于调试"""
        return self._residual
    
    def get_full_state(self):
        """获取完整状态 [位置, 速度]"""
        return (self.x[0, 0], self.x[1, 0])

    def reset(self):
        """重置滤波器"""
        self.x = np.zeros((2, 1), dtype=np.float32)
        self.P = np.eye(2, dtype=np.float32) * 100.0
        self.is_initialized = False
        self._residual = 0.0