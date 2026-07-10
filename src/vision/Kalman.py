# src/vision/Kalman.py
import numpy as np

class AdaptiveEKF3D:
    """
    3D 自适应扩展卡尔曼滤波器 (EKF)
    状态: [x, y, z, vx, vy, vz]  模型: 匀速 (Constant Velocity)
    特点: 根据观测残差自动调节 Q，快速响应突变
    """
    def __init__(self, Q_base=0.5, R=1.0, dt=1/30.0):
        """
        Args:
            Q_base: 基础过程噪声 (m/s²)
            R: 观测噪声 (m)
            dt: 初始时间步长 (秒)
        """
        self.dt = dt
        self.dim = 6  # 3 位置 + 3 速度
        
        # 1. 状态向量 [x, y, z, vx, vy, vz]
        self.x = np.zeros((6, 1), dtype=np.float32)
        
        # 2. 状态协方差矩阵 P
        self.P = np.eye(6, dtype=np.float32) * 10.0
        
        # 3. 状态转移矩阵 F
        self.F = np.eye(6, dtype=np.float32)
        self._update_F(dt)
        
        # 4. 观测矩阵 H（只观测位置）
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        
        # 5. 观测噪声协方差 R
        self.R = np.eye(3, dtype=np.float32) * R
        
        # 6. 过程噪声协方差 Q
        self.Q_base_value = Q_base
        self._update_Q_base(Q_base)
        
        self.is_initialized = False
        self._residual = np.zeros(3)

    def _update_F(self, dt):
        """更新状态转移矩阵"""
        self.F[0, 3] = dt  # x += vx * dt
        self.F[1, 4] = dt  # y += vy * dt
        self.F[2, 5] = dt  # z += vz * dt

    def _update_Q_base(self, Q_base):
        """更新基础过程噪声矩阵（匀速模型）"""
        dt = self.dt
        q = Q_base
        # 位置噪声: q * dt^2 / 2, 速度噪声: q * dt
        self.Q_base = np.eye(6, dtype=np.float32) * 0.01
        self.Q_base[0, 0] = q * dt
        self.Q_base[1, 1] = q * dt
        self.Q_base[2, 2] = q * dt
        self.Q_base[3, 3] = q * dt
        self.Q_base[4, 4] = q * dt
        self.Q_base[5, 5] = q * dt

    def set_initial_state(self, position):
        """设置初始状态（位置，速度设为0）"""
        self.x[0, 0] = position[0]
        self.x[1, 0] = position[1]
        self.x[2, 0] = position[2]
        self.is_initialized = True

    def predict(self, dt=None):
        """预测步骤"""
        if dt is not None and dt != self.dt:
            self.dt = dt
            self._update_F(dt)
            self._update_Q_base(self.Q_base_value)
        
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q_base
        return self.get_state()

    def update(self, measurement):
        """更新步骤（带自适应）"""
        z = np.array(measurement, dtype=np.float32).reshape(3, 1)
        
        # 1. 计算残差
        y = z - self.H @ self.x
        self._residual = y.flatten()
        
        # 2. 自适应 Q：残差大时增大 Q
        residual_norm = np.linalg.norm(self._residual)
        if residual_norm > 0.1:
            scale = min(15.0, residual_norm / 0.1)
            Q_adaptive = self.Q_base * scale
        else:
            Q_adaptive = self.Q_base
        
        # 3. 重新计算预测协方差
        P_pred = self.F @ self.P @ self.F.T + Q_adaptive
        
        # 4. 卡尔曼增益
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # 5. 状态更新
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ P_pred
        
        self.is_initialized = True
        return self.get_state()

    def get_state(self):
        """获取滤波后的位置 (x, y, z)"""
        return (self.x[0, 0], self.x[1, 0], self.x[2, 0])

    def get_full_state(self):
        """获取完整状态 (位置 + 速度)"""
        return self.x.flatten()

    def get_speed(self):
        """获取速度 (vx, vy, vz)"""
        return (self.x[3, 0], self.x[4, 0], self.x[5, 0])

    def reset(self):
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 100.0
        self.is_initialized = False