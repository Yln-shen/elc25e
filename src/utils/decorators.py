# src/utils/decorators.py
import time
import functools

def measure_fps(func):
    """装饰器：统计函数调用频率"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        wrapper.total_time += time.time() - wrapper.last_time
        if wrapper.total_time >= 1.0:
            wrapper.fps = wrapper.calls / wrapper.total_time
            wrapper.calls = 0
            wrapper.total_time = 0
        wrapper.last_time = time.time()
        return func(*args, **kwargs)
    
    wrapper.calls = 0
    wrapper.total_time = 0
    wrapper.fps = 0
    wrapper.last_time = time.time()
    return wrapper