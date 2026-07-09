import time
import functools

def measure_fps(func):
    """装饰器：统计函数调用频率"""
    @functools.wraps(func)  # 保持被装饰函数的元信息（如函数名、文档字符串）不变
    def wrapper(*args, **kwargs):
        # 1. 统计逻辑
        wrapper.calls += 1  # 调用次数 +1
        # 计算距离上次调用经过的时间，并累加到总时间
        wrapper.total_time += time.time() - wrapper.last_time
        
        # 如果累计总时间 >= 1秒，则计算FPS并重置计数器
        if wrapper.total_time >= 1.0:
            wrapper.fps = wrapper.calls / wrapper.total_time  # 计算FPS
            wrapper.calls = 0        # 重置调用次数
            wrapper.total_time = 0   # 重置总时间
        # 更新最后一次调用的时间戳
        wrapper.last_time = time.time()
        
        # 2. 执行原始函数
        return func(*args, **kwargs)
    
    # 为 wrapper 函数初始化四个“属性”，用于存储统计数据
    wrapper.calls = 0
    wrapper.total_time = 0
    wrapper.fps = 0
    wrapper.last_time = time.time()
    return wrapper

