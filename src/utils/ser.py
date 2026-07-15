import serial

class FireWaterIMU:
    def __init__(self, port='COM3', baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
    
    def connect(self):
        """连接串口"""
        self.ser = serial.Serial(self.port, self.baud, timeout=1)
        return self.ser.is_open
    
    def close(self):
        """关闭串口"""
        if self.ser:
            self.ser.close()
    
    def read_data(self):
        """读取并解析一帧数据，返回 (roll, pitch, yaw)"""
        try:
            line = self.ser.readline().decode('utf-8').strip()
            if not line:
                return None
            
            parts = line.split(',')
            if len(parts) == 3:
                self.roll, self.pitch, self.yaw = map(float, parts)
                return self.roll, self.pitch, self.yaw
            return None
        except Exception as e:
            print(f"解析错误: {e}")
            return None
    
    def get_roll(self):
        return self.roll
    
    def get_pitch(self):
        return self.pitch
    
    def get_yaw(self):
        return self.yaw


# ========== 使用示例 ==========
if __name__ == "__main__":
    imu = FireWaterIMU('COM3', 115200)
    
    if imu.connect():
        print("连接成功，开始读取数据...")
        try:
            while True:
                data = imu.read_data()
                if data:
                    roll, pitch, yaw = data
                    print(f"Roll: {roll:8.2f}°  Pitch: {pitch:8.2f}°  Yaw: {yaw:8.2f}°")
                    
                    # ===== 在这里添加你自己的处理逻辑 =====
                    # 直接用 roll, pitch, yaw 变量
                    
        except KeyboardInterrupt:
            print("\n停止接收")
        finally:
            imu.close()
    else:
        print("连接失败")