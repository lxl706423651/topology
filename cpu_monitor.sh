#!/bin/bash

# --- 配置 ---
INTERVAL=60  # 间隔时间（秒），这里设为60秒
# 文件名以启动时间命名，例如: cpu_monitor_20231027_103001.log
LOG_FILE="./logs/cpu_monitor/cpu_monitor_$(date +%Y%m%d_%H%M%S).log"

# --- 初始化日志头 ---
# 格式: 时间, Load(1min), Load(5min), Load(15min), CPU_User%, CPU_Kernel%, CPU_Idle%
echo "Timestamp,Load_1,Load_5,Load_15,CPU_User,CPU_Kernel,CPU_Idle" > "$LOG_FILE"

echo "========================================================"
echo "开始监控... 日志文件: $LOG_FILE"
echo "按 Ctrl + C 停止监控"
echo "========================================================"

# --- 循环监控 ---
while true; do
    # 1. 获取当前时间
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

    # 2. 获取 Load Average (从 /proc/loadavg 读取更纯净，不需要处理 'load average:' 字符串)
    # /proc/loadavg 前三个字段分别是 1、5、15 分钟的负载
    LOAD_1=$(awk '{print $1}' /proc/loadavg)
    LOAD_5=$(awk '{print $2}' /proc/loadavg)
    LOAD_15=$(awk '{print $3}' /proc/loadavg)

    # 3. 获取 CPU 使用率
    # 使用 top -bn1 获取一次静态快照
    # grep "Cpu(s)" 提取 CPU 行
    # 下面的 awk 逻辑是为了适配大部分 Linux 发行版 top 的输出格式
    # 通常格式为: %Cpu(s): 10.5 us,  5.2 sy,  0.0 ni, 84.3 id...
    CPU_RAW=$(top -bn1 | grep "Cpu(s)")
    
    # 提取 User (us), System/Kernel (sy), Idle (id)
    # 逻辑：以 "us," 分割取前面部分，再取最后一个词，兼容性较好
    CPU_US=$(echo "$CPU_RAW" | awk -F'us,' '{print $1}' | awk '{print $NF}')
    CPU_SY=$(echo "$CPU_RAW" | awk -F'sy,' '{print $1}' | awk '{print $NF}')
    CPU_ID=$(echo "$CPU_RAW" | awk -F'id,' '{print $1}' | awk '{print $NF}')

    # 4. 写入文件 (CSV格式)
    echo "$TIMESTAMP,$LOAD_1,$LOAD_5,$LOAD_15,$CPU_US,$CPU_SY,$CPU_ID" >> "$LOG_FILE"

    # 5. 屏幕输出 (作为心跳包防止 SSH 断开)
    echo "[$TIMESTAMP] Load: $LOAD_1 | User: ${CPU_US}% | Kernel: ${CPU_SY}%"

    # 6. 等待
    sleep $INTERVAL
done