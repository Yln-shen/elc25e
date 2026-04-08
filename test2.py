import serial
import time

# 配置串口
ser = serial.Serial(
    port='/dev/ttyACM0',
    baudrate=115200,      # 默认波特率
    bytesize=8,
    parity='N',
    stopbits=1,
    timeout=1
)

# Modbus-RTU 读取版本号的命令 (从机地址0x01, 功能码0x04, 寄存器地址0x001F)
read_version_cmd = bytes.fromhex('01 04 00 1F 00 01 00 0C')

print("发送读取版本命令...")
ser.write(read_version_cmd)
time.sleep(0.2)

response = ser.read(20)
if response:
    print(f"收到响应: {response.hex()}")
    # 如果响应以 '01 04 02' 开头，说明通讯成功
else:
    print("未收到响应，请检查通讯模式配置")

ser.close()