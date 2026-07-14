import serial as s
import struct
import time
import threading
from dataclasses import dataclass, field


# ===== 通信协议常量 =====
HEADER_1 = 0xAA
HEADER_2 = 0x55
TAIL_1 = 0x0D
TAIL_2 = 0x0A

CMD_GYRO = 0x02      # IMU 数据上报
FRAME_DATA_LEN = 36   # 9 × float32


@dataclass
class IMUData:
    """ICM-42688-P IMU 完整数据"""
    # 原始陀螺仪角速度 (dps)
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0
    # 原始加速度 (m/s^2 或 g，取决于 MCU 固件)
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0
    # MCU 解算的姿态角 (度)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    # 元数据
    timestamp: float = 0.0
    frame_count: int = 0


class SerialReceiver:
    """UART 接收端 — 独立线程持续接收 ICM-42688-P 数据"""

    def __init__(self, port='/dev/ttyACM0', baudrate=115200, timeout=0.02):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self._lock = threading.Lock()
        self._latest: IMUData | None = None
        self._frame_count = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._parse_errors = 0
        self._checksum_errors = 0

    # ===== 串口管理 =====

    def open_port(self):
        """打开串口（阻塞重试直到成功）"""
        while True:
            try:
                self.ser = s.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    bytesize=s.EIGHTBITS,
                    parity=s.PARITY_NONE,
                    stopbits=s.STOPBITS_ONE,
                    timeout=self.timeout,
                )
                print(f"[SerialReceiver] 串口已打开: {self.port} @ {self.baudrate}")
                return
            except Exception as e:
                print(f"[SerialReceiver] 打开串口失败 ({e})，1秒后重试...")
                time.sleep(1)

    def close(self):
        """关闭串口并停止接收线程"""
        self.stop()
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[SerialReceiver] 串口已关闭")

    # ===== 帧解析 =====

    @staticmethod
    def _parse_frame(payload: bytes) -> IMUData | None:
        """解析 36 字节 payload 为 IMUData"""
        if len(payload) < FRAME_DATA_LEN:
            return None
        try:
            gx, gy, gz, ax, ay, az, yaw, pitch, roll = struct.unpack(
                '<9f', payload[:FRAME_DATA_LEN]
            )
            return IMUData(
                gx=gx, gy=gy, gz=gz,
                ax=ax, ay=ay, az=az,
                yaw=yaw, pitch=pitch, roll=roll,
            )
        except struct.error:
            return None

    @staticmethod
    def _verify_checksum(cmd: int, length: int, data: bytes, checksum: int) -> bool:
        """验证 XOR 校验"""
        expected = cmd ^ length
        for b in data:
            expected ^= b
        return (expected & 0xFF) == (checksum & 0xFF)

    # ===== 后台接收线程 =====

    def _receive_loop(self):
        """后台线程：持续从串口读取并解析帧"""
        buf = bytearray()
        MIN_FRAME = 7       # header(2) + cmd(1) + len(1) + data(0) + chk(1) + tail(2)
        MAX_DATA_LEN = 64   # 合理的 payload 上限，防止垃圾 length 导致 OOM
        MAX_BUF = 4096      # 缓冲区上限，防止无限增长

        while self._running:
            try:
                if self.ser is None or not self.ser.is_open:
                    self.open_port()
                    buf.clear()

                # 读取可用字节
                waiting = self.ser.in_waiting or 1
                chunk = self.ser.read(waiting)
                if not chunk:
                    continue
                buf.extend(chunk)

                # 缓冲区上限保护
                if len(buf) > MAX_BUF:
                    buf = buf[-MAX_BUF // 2:]

                # 搜索帧头，至少需要 MIN_FRAME 字节才能进入解析
                while len(buf) >= MIN_FRAME:
                    idx = buf.find(bytes([HEADER_1, HEADER_2]))
                    if idx < 0:
                        # 未找到帧头：保留末尾 1 字节（可能是 HEADER_1 的残片）
                        buf = buf[-1:]
                        break

                    # 丢弃帧头前的杂散字节，此时 buf[0:2] == [0xAA, 0x55]
                    if idx > 0:
                        del buf[:idx]

                    # 需要至少读到 cmd + length
                    if len(buf) < 4:
                        break

                    cmd = buf[2]
                    length = buf[3]

                    # 校验 length 合法性
                    if not (1 <= length <= MAX_DATA_LEN):
                        # length 字段损坏，跳过这个假帧头，继续搜索
                        del buf[:2]
                        self._parse_errors += 1
                        continue

                    frame_total = 2 + 1 + 1 + length + 1 + 2  # header + cmd + len + data + chk + tail
                    if len(buf) < frame_total:
                        break  # 数据不完整，等下个 chunk

                    # 提取完整帧
                    frame = buf[:frame_total]
                    del buf[:frame_total]

                    # 校验帧尾
                    if frame[-2] != TAIL_1 or frame[-1] != TAIL_2:
                        self._parse_errors += 1
                        continue

                    # 提取字段并校验 checksum
                    frame_cmd = frame[2]
                    frame_len = frame[3]
                    frame_data = frame[4:4 + frame_len]
                    frame_chk = frame[4 + frame_len]

                    if not self._verify_checksum(frame_cmd, frame_len, frame_data, frame_chk):
                        self._checksum_errors += 1
                        continue

                    # 仅处理陀螺仪数据帧
                    if frame_cmd != CMD_GYRO or frame_len != FRAME_DATA_LEN:
                        continue

                    imu = self._parse_frame(frame_data)
                    if imu is None:
                        self._parse_errors += 1
                        continue

                    # 更新内部状态
                    imu.timestamp = time.time()
                    self._frame_count += 1
                    imu.frame_count = self._frame_count

                    with self._lock:
                        self._latest = imu

            except (s.SerialException, OSError) as e:
                print(f"[SerialReceiver] 串口错误: {e}，尝试重连...")
                try:
                    if self.ser and self.ser.is_open:
                        self.ser.close()
                except Exception:
                    pass
                self.ser = None
                time.sleep(0.5)

    # ===== 公开接口 =====

    def start(self):
        """启动后台接收线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()
        print("[SerialReceiver] 接收线程已启动")

    def stop(self):
        """停止后台接收线程"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("[SerialReceiver] 接收线程已停止")

    def get_data(self) -> IMUData | None:
        """获取最新一帧 IMU 数据（非阻塞，无新数据返回 None）"""
        with self._lock:
            return self._latest

    @property
    def has_data(self) -> bool:
        """是否有数据可用"""
        with self._lock:
            return self._latest is not None

    @property
    def stats(self) -> dict:
        """返回接收统计信息"""
        with self._lock:
            latest_ts = self._latest.timestamp if self._latest else 0
        return {
            'frame_count': self._frame_count,
            'parse_errors': self._parse_errors,
            'checksum_errors': self._checksum_errors,
            'latest_timestamp': latest_ts,
            'running': self._running,
        }


# ===== 便捷函数 =====

def create_receiver(port='/dev/ttyACM0', baudrate=115200) -> SerialReceiver:
    """工厂函数：创建并启动接收器"""
    rx = SerialReceiver(port=port, baudrate=baudrate)
    rx.start()
    return rx


# ===== 测试入口 =====

if __name__ == "__main__":
    rx = create_receiver()
    print("IMU 接收器已启动，按 Ctrl+C 退出...\n")
    try:
        while True:
            data = rx.get_data()
            if data:
                print(
                    f"\r[#{data.frame_count:05d}] "
                    f"Gyro: ({data.gx:+8.2f}, {data.gy:+8.2f}, {data.gz:+8.2f}) dps | "
                    f"Accel: ({data.ax:+8.3f}, {data.ay:+8.3f}, {data.az:+8.3f}) g | "
                    f"Attitude: Yaw={data.yaw:+8.2f} Pitch={data.pitch:+8.2f} Roll={data.roll:+8.2f} deg",
                    end='', flush=True
                )
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n\n正在退出...")
        print(f"统计: {rx.stats}")
    finally:
        rx.close()
