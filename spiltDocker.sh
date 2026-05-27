#!/bin/bash

# --- 配置 ---

# ！！！注意：Python 脚本现在默认输出到 'output/split_config'
# 请确保这个路径与您的 Python 脚本一致
CONFIG_DIR="output"

# 共享的网络文件
NETWORKS_FILE="$CONFIG_DIR/common-networks.yml"

# 您总共有多少个Batch文件
TOTAL_BATCHES=46

# 同时并行运行的“Batch”数量
# 既然 'depends_on' 错误已修复, 我们可以安全地
# 从 4 或 8 开始测试真正的构建性能。
MAX_PARALLEL_BATCHES=6

# --- 日志 ---
BUILD_LOG_DIR="build_logs_$(date +%F_%H%M%S)"
mkdir -p "$BUILD_LOG_DIR"
SUCCESS_LOG="$BUILD_LOG_DIR/summary_success.log"
FAIL_LOG="$BUILD_LOG_DIR/summary_fail.log"

# 清空历史日志
> "$SUCCESS_LOG"
> "$FAIL_LOG"

echo "================================================="
echo "Starting CONTROLLED Parallel Batch Build (v3 - depends_on removed)"
echo "  Config Dir: $CONFIG_DIR"
echo "  Max Parallel Batches: $MAX_PARALLEL_BATCHES"
echo "  Total Batches: $TOTAL_BATCHES"
echo "  Logs will be stored in: $BUILD_LOG_DIR"
echo "================================================="

# --- 核心功能 ---

build_single_batch() {
    local batch_num_padded=$(printf "%02d" $1)
    local batch_file="$CONFIG_DIR/services-batch-${batch_num_padded}.yml"
    local log_file="$BUILD_LOG_DIR/batch_${batch_num_padded}.log"

    if [ ! -f "$batch_file" ]; then
        echo "  SKIPPING: Batch $1 (File not found: $batch_file)"
        return
    fi

    echo "Building Batch ${batch_num_padded} (Services sequentially)..."

    # ！！！已更新：
    # 1. 将 --progress plain 移到 build 之前 (根据您的建议)
    # 2. 彻底移除了 --no-deps
    if docker compose -f "$NETWORKS_FILE" -f "$batch_file" --progress plain build > "$log_file" 2>&1; then
        echo "  ✅ SUCCESS: Batch ${batch_num_padded}"
        echo "Batch ${batch_num_padded}" >> "$SUCCESS_LOG"
    else
        echo "  ❌ FAILED: Batch ${batch_num_padded}. Check $log_file"
        echo "Batch ${batch_num_padded} (Log: $log_file)" >> "$FAIL_LOG"
    fi
}

# 导出函数和变量
export -f build_single_batch
export CONFIG_DIR NETWORKS_FILE BUILD_LOG_DIR SUCCESS_LOG FAIL_LOG
export DOCKER_BUILDKIT=0
# --- 执行 ---
START_TIME=$(date +%s)
echo "Build process started at: $(date)"

seq 1 $TOTAL_BATCHES | xargs -n 1 -P $MAX_PARALLEL_BATCHES bash -c 'build_single_batch "$@"' _

END_TIME=$(date +%s)

# --- 总结 ---
echo "-------------------------------------"
echo "Batch build summary:"
echo "  Success: $(wc -l < "$SUCCESS_LOG") batches"
echo "  Failed:  $(wc -l < "$FAIL_LOG") batches"
echo "Details in $BUILD_LOG_DIR"
echo "-------------------------------------"

DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo "Total execution time: ${MINUTES} minutes and ${SECONDS} seconds."
echo "================================================="