import gpiod
import time

# 1. 打开 GPIO 芯片
chip = gpiod.Chip('4') # '0' 对应 gpiochip0

# 2. 获取引脚 (以 line 17 为例)
line = chip.get_line(18)

# 3. 申请为输出模式，初始值为 0
line.request(consumer="my-led", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])

# 4. 让 LED 闪烁 5 次
while(1):
    line.set_value(1) # 高电平
    time.sleep(2)
    line.set_value(0) # 低电平
    time.sleep(1)