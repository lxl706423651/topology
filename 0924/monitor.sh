#!/bin/bash

# 1. 设置语言环境，防止因为系统语言不同导致 grep 抓取失败或小数点格式错误
export LC_ALL=C

# 2. 定义带时间戳的文件名 (.csv 格式方便 Excel 打开)
LOG_FILE="monitor_$(date +%Y%m%d_%H%M%S).csv"

# 3. 写入表头
echo "Timestamp,CPU_Usage_%,Mem_Usage_%" > "$LOG_FILE"

echo "开始监控... 数据写入: $LOG_FILE"
echo "按 Ctrl+C 停止"

while true; do
    # 获取当前时间
    TS=$(date "+%Y-%m-%d %H:%M:%S")
    
    # ---------------------------------------------------------
    # 获取内存使用率 (百分比)
    # free 输出的第2列是Total，第3列是Used
    # awk '{printf "%.2f", $3/$2 * 100}' 用于计算百分比并保留2位小数
    # ---------------------------------------------------------
    MEM_USAGE=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100}')
    
    # 获取 CPU 使用率 (使用 top)
    # 获取 idle (空闲) 值，然后用 100 减去它
    CPU_IDLE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print $1}')
    
    # 使用 awk 计算 100 - idle
    CPU_USAGE=$(awk -v idle="$CPU_IDLE" 'BEGIN {print 100 - idle}')
    
    # 写入文件
    echo "$TS,$CPU_USAGE,$MEM_USAGE" >> "$LOG_FILE"
    
    # (可选) 在终端实时打印，\r 用于覆盖当前行，实现动态刷新效果
    echo -ne "\r[$TS] CPU: ${CPU_USAGE}% | MEM: ${MEM_USAGE}% "
    
    sleep 20
done