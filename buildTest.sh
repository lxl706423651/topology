#!/bin/bash

# --- 记录开始时间 ---
start_time=$(date +%s)

# --- 配置 ---
COMPOSE_FILE="./output/docker-compose.yml"
PARALLEL_JOBS=12
BATCH_SIZE=50
LOG_FILE="build_debug.log"

# --- 清空旧日志 ---
> "$LOG_FILE"

# --- 执行 ---
echo "Fetching service names from $COMPOSE_FILE..."
SERVICES_LIST=$(docker compose -f "$COMPOSE_FILE" config --services)

if [ -z "$SERVICES_LIST" ]; then
    echo "ERROR: Could not get service list"
    exit 1
fi

echo "Starting parallel build..."
echo "  Parallel Jobs (-P): $PARALLEL_JOBS"
echo "  Batch Size (-n):    $BATCH_SIZE"

# 强制使用旧版构建器（保持你原有的设定）
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
# 定义一个 Wrapper 函数来记录单批次的时间
# 我们将其导出以便 xargs 调用
run_batch() {
    batch_id=$1
    shift
    echo "[Batch-$batch_id] START $(date +%T.%N)" >> build_timestamps.log
    
    # 捕获 docker compose 的输出，提取 Transferring context 的时间信息（如果 plain 模式显示的话）
    # 使用 time 命令统计这一批次的纯耗时
    /usr/bin/time -f "[Batch-$batch_id] SYSTEM_TIME: %E" \
        docker compose -f "$COMPOSE_FILE" build --progress=plain "$@" 2>> "$LOG_FILE"
    
    echo "[Batch-$batch_id] END   $(date +%T.%N)" >> build_timestamps.log
}
export -f run_batch
export COMPOSE_FILE
export LOG_FILE

# 准备批次并执行
# 我们用 awk 给每一批加一个 ID，方便追踪
echo "$SERVICES_LIST" | xargs -n $BATCH_SIZE | awk '{print NR, $0}' | \
    xargs -n $((BATCH_SIZE + 1)) -P $PARALLEL_JOBS bash -c 'run_batch "$@"' _

echo "-------------------------------------"
echo "All build batches finished."

# --- 计算并输出运行时间 ---
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "-------------------------------------"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"
echo "Check build_timestamps.log for concurrency analysis."
