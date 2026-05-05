
import gpiod
import time

class GPIO:
    def __init__(self, chip_path='/dev/gpiochip4', line=18, consumer='my-led'):
        self.chip = gpiod.Chip(chip_path)
        self.line_offset = line
        
        # 明确配置为输出模式
        settings = gpiod.LineSettings(
            direction=gpiod.line.Direction.OUTPUT
        )
        self.request = self.chip.request_lines(
            config={line: settings},
            consumer=consumer
        )
        print(f"GPIO {line} 初始化成功")
    
    def on(self):
        self.request.set_value(self.line_offset, gpiod.line.Value.ACTIVE)
    
    def off(self):
        self.request.set_value(self.line_offset, gpiod.line.Value.INACTIVE)
    
    def set(self, value):
        self.on() if value else self.off()
    
    def release(self):
        self.request.release()
        self.chip.close()
        print("GPIO 已释放")

if __name__ == "__main__":
    gpio = GPIO(chip_path='/dev/gpiochip4', line=18, consumer='my-led')
    print("高电平")
    gpio.on()
    time.sleep(2)
    print("低电平")
    gpio.off()
    time.sleep(1)
    gpio.release()