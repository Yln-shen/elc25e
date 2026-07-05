# pid.py
class PID:
    """
    增量式 PID 控制器
    """
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, output_min=-100.0, output_max=100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        
        # 历史数据存储
        self.reset()

    def reset(self):
        """
        重置控制器状态
        """
        self.prev_error = 0.0      # e(k-1)
        self.prev_prev_error = 0.0 # e(k-2)
        self.last_output = 0.0     # 上一次的输出值

    def compute(self, error, dt=0.01):
        """
        计算 PID 输出
        :param error: 当前误差
        :param dt: 时间间隔
        :return: 控制输出
        """
        # 1. 计算增量 Δu
        # 为了防止微分项在 dt 极小时爆炸，这里做保护
        derivative = (error - 2 * self.prev_error + self.prev_prev_error) / (dt * dt) if dt > 0 else 0
        delta_u = self.kp * (error - self.prev_error) + self.ki * error * dt + self.kd * derivative

        # 2. 计算最终输出 u(k) = u(k-1) + Δu
        output = self.last_output + delta_u

        # 3. 输出限幅
        output = max(self.output_min, min(self.output_max, output))

        # 4. 更新历史数据
        self.prev_prev_error = self.prev_error
        self.prev_error = error
        self.last_output = output

        return output

    def set_limits(self, output_min, output_max):
        """
        动态设置输出限幅
        """
        self.output_min = output_min
        self.output_max = output_max