import time
import sys
from datetime import datetime

try:
    import psutil
except ImportError:
    print("错误：未找到 'psutil' 库。")
    print("请先运行: pip3 install psutil")
    sys.exit(1)

# --- 配置 ---

# 1. 获取当前时间并格式化，用于文件名
#    格式： YYYY-MM-DD_HH-MM-SS (例如 2025-11-13_21-55-55)
current_time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# 2. 拼接日志文件名
LOG_FILE = f"docker_monitor_{current_time_str}.log"

# 采样间隔（秒）。
# 1秒会产生非常多的日志，建议使用 2秒 或 5秒。
# psutil.cpu_percent() 会阻塞这么久。
INTERVAL_SECONDS = 10

# --- 结束配置 ---

print(f"开始监控... 按 Ctrl+C 退出。")
# 打印出我们将要写入的动态文件名
print(f"日志将每 {INTERVAL_SECONDS} 秒追加到: {LOG_FILE}")
time.sleep(1)

try:
    # 使用 'a' 模式（追加模式）打开文件
    # buffering=1 表示“行缓冲”，确保每行都被立即写入
    with open(LOG_FILE, 'a', buffering=1, encoding='utf-8') as f:
        
        # 在日志中写入一个启动标记
        start_time_display = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"\n--- 监控会话开始: {start_time_display} (间隔: {INTERVAL_SECONDS}s) ---\n")
        
        while True:
            # 1. 获取 CPU 占用
            # 这个函数会阻塞 INTERVERAL_SECONDS 那么久，并返回这段时间的平均值
            cpu_percent = psutil.cpu_percent(interval=INTERVAL_SECONDS, percpu=False)
            
            # 2. 获取内存占用
            mem = psutil.virtual_memory()
            mem_used_gb = mem.used / (1024 ** 3)
            mem_total_gb = mem.total / (1024 ** 3)
            
            # 3. 获取当前时间戳
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 4. 格式化日志行
            log_line = (
                f"[{timestamp}] "
                f"CPU: {cpu_percent:5.1f}% | "
                f"内存: {mem.percent:5.1f}% "
                f"({mem_used_gb:.2f} GB / {mem_total_gb:.2f} GB)"
            )
            
            # 5. 打印到屏幕
            print(log_line)
            
            # 6. 写入到文件
            f.write(log_line + "\n")
            
except KeyboardInterrupt:
    # 当用户按下 Ctrl+C 时
    print(f"\n监控已停止。日志已保存到 {LOG_FILE}")
except Exception as e:
    print(f"\n发生错误: {e}")