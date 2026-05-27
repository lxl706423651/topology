#!/bin/bash

# --- 记录开始时间 ---
start_time=$(date +%s)  # 以秒为单位记录开始时间


# --- 配置 ---
COMPOSE_FILE="./output/docker-compose.yml"
PARALLEL_JOBS=16
BATCH_SIZE=50


# --- 执行 ---
echo "Fetching service names from $COMPOSE_FILE..."
SERVICES_LIST=$(docker compose -f "$COMPOSE_FILE" config --services)

if [ -z "$SERVICES_LIST" ]; then
    echo "ERROR: Could not get service list from $COMPOSE_FILE"
    exit 1
fi

echo "Starting parallel build..."
echo "  Parallel Jobs (-P): $PARALLEL_JOBS"
echo "  Batch Size (-n):    $BATCH_SIZE"

export DOCKER_BUILDKIT=0

echo "$SERVICES_LIST" | xargs -n $BATCH_SIZE -P $PARALLEL_JOBS \
    docker compose -f "$COMPOSE_FILE" up -d

echo "-------------------------------------"
echo "All build batches finished."


# --- 计算并输出运行时间 ---
end_time=$(date +%s)  # 以秒为单位记录结束时间
duration=$((end_time - start_time))  # 总秒数

# 转换为 时:分:秒 格式
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "-------------------------------------"
echo "Total runtime: ${hours}h ${minutes}m ${seconds}s"