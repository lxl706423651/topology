#!/bin/bash

# --- 配置区域 ---
start_time=$(date +%s)
COMPOSE_FILE="./output/docker-compose.yml"
# --- 日志配置（新增）---
# 定义日志目录
BASE_DIR="${EXP_LOG_DIR:-./logs}"
# 例如内存脚本可以这样写：
LOG_DIR="$BASE_DIR/dockerBuild"

# 创建日志目录（不存在则创建，-p 确保多级目录都能创建）
mkdir -p "$LOG_DIR"
# 生成带时间戳的日志文件名（格式：dockerBuild_YYYYMMDD_HHMMSS.log）
LOG_FILE="${LOG_DIR}/dockerBuild_$(date +%Y%m%d_%H%M%S).log"

# 清空之前的总日志
> "$LOG_FILE"

# --- 开启性能模式 ---
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1
# # 尝试提高文件句柄限制
# ulimit -n 65535 2>/dev/null || true

# --- 0. 检查 Context 大小 ---
echo "Checking build context size (target: ./output)..."
# 因为 context: . 在 output 目录下，所以我们需要检查 output 目录的大小
# 期望：几十 MB (如果还是几百 MB，说明 .dockerignore 没生效或漏了东西)
du -sh ./output 

echo "-------------------------------------------------"
echo "Phase 1: Cache Warmer (Standard)"
echo "-------------------------------------------------"
# 获取第一个服务名
FIRST_SERVICE=$(docker compose -f "$COMPOSE_FILE" config --services | head -n 1)

# 【修正点 1】: 将日志输出到 warmup.log，而不是直接扔进 /dev/null
# 这样如果失败了，我们可以看到原因
echo "Building first service: $FIRST_SERVICE ..."
docker compose -f "$COMPOSE_FILE" build "$FIRST_SERVICE" > warmup.log 2>&1

if [ $? -ne 0 ]; then 
    echo "❌ Warmup failed! Error log:"
    cat warmup.log
    rm warmup.log
    exit 1
fi
# 如果成功，把日志追加到总日志，并清理
cat warmup.log >> "$LOG_FILE"
rm warmup.log
echo "✅ Warmup successful."


# --- 核心调整 ---
BATCH_SIZE=50
PARALLEL_JOBS=16

echo "-------------------------------------------------"
echo "Phase 2: Golden Ratio Build (P=$PARALLEL_JOBS, B=$BATCH_SIZE)"
echo "-------------------------------------------------"

# 1. 生成批次列表
docker compose -f "$COMPOSE_FILE" config --services | xargs -n $BATCH_SIZE > batch_list.txt
TOTAL_BATCHES=$(wc -l < batch_list.txt)
echo "Total Batches to run: $TOTAL_BATCHES"

# 2. 执行构建
# 【修正点 2】: LOG_FILE 聚合策略
# 我们依然输出到独立文件 build_log_ID.txt 以防止并行写入导致乱码
# 但在最后我们会把它们合并
cat batch_list.txt | xargs -P $PARALLEL_JOBS -I {} sh -c '
    start=$(date +%s)
    
    # 生成唯一的 Batch ID
    BATCH_ID=$(echo "{}" | md5sum | cut -c 1-5)
    TEMP_LOG="build_log_${BATCH_ID}.txt"
    
    # 执行构建，日志写入临时文件
    docker compose -f ./output/docker-compose.yml build {} > "$TEMP_LOG" 2>&1
    ret=$?
    
    end=$(date +%s)
    dur=$((end - start))
    
    if [ $ret -eq 0 ]; then
        echo "✅ Batch $BATCH_ID done in ${dur}s"
    else
        echo "❌ Batch $BATCH_ID FAILED in ${dur}s (Check $TEMP_LOG)"
        # 失败时打印最后几行错误信息到屏幕
        tail -n 3 "$TEMP_LOG"
    fi
'

# --- 3. 日志聚合与清理 ---
echo "-------------------------------------------------"
echo "Aggregating logs to $LOG_FILE ..."

# 将所有临时日志合并到主日志文件
cat build_log_*.txt >> "$LOG_FILE" 2>/dev/null

# 删除临时文件
rm batch_list.txt
rm build_log_*.txt 2>/dev/null

# --- 结算 ---
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

runtime_msg="Total runtime: ${minutes}m ${seconds}s"
echo $runtime_msg  # 输出到控制台
echo $runtime_msg >> "$LOG_FILE"  # 写入日志文件

log_msg="Full log saved to: $LOG_FILE"
echo $log_msg  # 输出到控制台
echo $log_msg >> "$LOG_FILE"  # 写入日志文件