#!/bin/bash

# 实验标识，用于命名输出文件
EXPERIMENT_NAME="4k_nodes_bgp_convergence"
LOG_FILE="${EXPERIMENT_NAME}_pidstat.log"

echo "开始收集所有 BIRD 进程的上下文切换数据..."
echo "Timestamp, PID, Container, %usr, %system, %CPU, cswch/s, nvcswch/s" > "$LOG_FILE"

# 找到宿主机上所有正在运行的 bird 进程的 PID
BIRD_PIDS=$(pgrep -f "bird -c")

if [ -z "$BIRD_PIDS" ]; then
    echo "未找到任何 BIRD 进程，请确保仿真器已启动。"
    exit 1
fi

# 将 PIDs 转换为 pidstat 可以接受的逗号分隔格式
PID_LIST=$(echo "$BIRD_PIDS" | tr '\n' ',' | sed 's/,$//')

# 每 1 秒采集一次数据，持续采集（可用 Ctrl+C 停止，或设置循环次数）
# -w: 报告上下文切换
# -u: 报告 CPU 利用率
# -h: 在一行中水平输出所有指标，方便后续 Python/Pandas 解析
echo "正在记录数据到 $LOG_FILE (按 Ctrl+C 停止)..."
pidstat -w -u -p "$PID_LIST" -h 1 | awk -v date="$(date +%s)" '
    # 跳过空行和表头行，提取我们需要的数据列
    /^[0-9]/ {
        # pidstat -h 的列顺序通常为: Time UID PID %usr %system %guest %wait %CPU CPU Command cswch/s nvcswch/s Command
        # 请根据你宿主机 pidstat 的实际输出版本微调列号($X)
        print date", "$3", "$4", "$5", "$8", "$10", "$11
    }
' >> "$LOG_FILE"
