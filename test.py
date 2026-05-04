import serial
import time

def test_motor():
    # 常见波特率列表
    baudrates = [9600, 19200, 38400, 57600, 115200]
    motor_ids = [1, 2, 3]
    
    port = '/dev/ttyACM0'
    
    for baudrate in baudrates:
        for motor_id in motor_ids:
            try:
                print(f"\n尝试 波特率={baudrate}, ID={motor_id}")
                ser = serial.Serial(port, baudrate, timeout=0.1)
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                
                # 读取版本号命令 (参考张大头协议)
                # 格式: [地址] [0x1F] [校验和]
                cmd = bytes([motor_id, 0x1F, (motor_id + 0x1F) & 0xFF])
                print(f"发送: {cmd.hex()}")
                ser.write(cmd)
                time.sleep(0.05)
                
                # 读取响应
                response = ser.read(20)
                if response:
                    print(f"✅ 响应: {response.hex()}")
                    print(f"成功！波特率={baudrate}, ID={motor_id}")
                    # 解析版本信息
                    if len(response) >= 4:
                        print(f"版本号: V{response[3]}.{response[2]}")
                    ser.close()
                    return True
                else:
                    print("❌ 无响应")
                    
                ser.close()
            except Exception as e:
                print(f"错误: {e}")
                continue
    
    print("\n所有组合都测试失败，请检查：")
    print("1. 串口助手的实际波特率设置")
    print("2. 电机ID（查看驱动器拨码开关）")
    print("3. 串口权限: sudo chmod 666 /dev/ttyACM0")
    return False

if __name__ == '__main__':
    test_motor()