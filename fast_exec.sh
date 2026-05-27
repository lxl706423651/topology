#!/bin/bash
# fast_exec_timed.sh
# 用法: ./fast_exec_timed.sh <容器名> <命令>

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <container_name> <command>"
    exit 1
fi

CONTAINER_ID=$1
CMD=${@:2} # 获取剩下的所有参数作为命令

# ==========================================
# 阶段 1: 获取 PID (Docker API 调用)
# ==========================================
# 获取开始时间 (纳秒)
start_t1=$(date +%s%N)

PID=$(docker inspect -f '{{.State.Pid}}' $CONTAINER_ID)

# 获取结束时间
end_t1=$(date +%s%N)
# 计算差值 (纳秒 / 1,000,000 = 毫秒)
duration1=$(( (end_t1 - start_t1) / 1000000 ))


# ==========================================
# 阶段 2: 飞雷神之术 (Namespace 切换)
# ==========================================
start_t2=$(date +%s%N)

# 执行命令
sudo nsenter -t $PID -n -m -u -i -p -- $CMD

end_t2=$(date +%s%N)
duration2=$(( (end_t2 - start_t2) / 1000000 ))

# ==========================================
# 结果输出
# ==========================================
# 注意：使用 >&2 将统计信息输出到 stderr，
# 这样如果你用管道处理脚本的输出结果，统计信息不会干扰数据流。
echo "---------------------------------------------------" >&2
echo "📊 Time Statistics:" >&2
echo "   1. Docker Inspect (Get PID) : ${duration1} ms" >&2
echo "   2. Nsenter Exec (Run Cmd)   : ${duration2} ms" >&2
echo "---------------------------------------------------" >&2
