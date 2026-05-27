#!/bin/bash

# --- 记录开始时间 ---
start_time=$(date +%s)  # 以秒为单位记录开始时间

# --- 核心配置 ---
COMPOSE_FILE="./output/docker-compose.yml"
PARALLEL_JOBS=4
BATCH_SIZE=50

# --- 日志配置（新增）---
# 定义日志目录
BASE_DIR="${EXP_LOG_DIR:-./logs}"

# 例如内存脚本可以这样写：
LOG_DIR="$BASE_DIR/dockerBuild"
# 创建日志目录（不存在则创建，-p 确保多级目录都能创建）
mkdir -p "$LOG_DIR"
# 生成带时间戳的日志文件名（格式：dockerUp_YYYYMMDD_HHMMSS.log）
LOG_FILE="${LOG_DIR}/dockerUp_$(date +%Y%m%d_%H%M%S).log"

# --- 日志写入函数（新增）---
# 功能：同时输出到控制台和日志文件
log_output() {
    local msg="$1"
    echo "$msg"  # 输出到控制台
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $msg" >> "$LOG_FILE"  # 带时间戳写入日志
}

# --- 初始化日志（新增）---
log_output "===== Docker Compose Up 批量启动脚本开始 ====="
log_output "脚本启动时间: $(date +'%Y-%m-%d %H:%M:%S')"
log_output "配置文件路径: $COMPOSE_FILE"
log_output "并行任务数: $PARALLEL_JOBS"
log_output "批次大小: $BATCH_SIZE"

# --- 执行 ---
log_output "Fetching service names from $COMPOSE_FILE..."
SERVICES_LIST=$(docker compose -f "$COMPOSE_FILE" config --services)

if [ -z "$SERVICES_LIST" ]; then
    error_msg="ERROR: Could not get service list from $COMPOSE_FILE"
    log_output "$error_msg"
    exit 1
fi

log_output "Starting parallel build..."
log_output "  Parallel Jobs (-P): $PARALLEL_JOBS"
log_output "  Batch Size (-n):    $BATCH_SIZE"

export DOCKER_BUILDKIT=0

# 执行批量启动，并将执行过程的输出也写入日志（新增）
log_output "开始执行 docker compose up -d 批量命令..."
echo "$SERVICES_LIST" | xargs -n $BATCH_SIZE -P $PARALLEL_JOBS \
    docker compose -f "$COMPOSE_FILE" up -d >> "$LOG_FILE" 2>&1

log_output "-------------------------------------"
log_output "All build batches finished."

# --- 计算并输出运行时间 ---
end_time=$(date +%s)  # 以秒为单位记录结束时间
duration=$((end_time - start_time))  # 总秒数

# 转换为 时:分:秒 格式
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

runtime_msg="Total runtime: ${hours}h ${minutes}m ${seconds}s"
log_output "-------------------------------------"
log_output "$runtime_msg"
log_output "===== Docker Compose Up 批量启动脚本结束 ====="
log_output "完整日志文件路径: $LOG_FILE"