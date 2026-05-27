#!/bin/bash

# ==========================================
# 宿主机全局内存快照脚本 (带节点规模与精确时间戳)
# ==========================================

BASE_DIR="${EXP_LOG_DIR:-./logs}"
# 例如内存脚本可以这样写：
LOG_DIR="$BASE_DIR/memory"
mkdir -p "$LOG_DIR"

# 1. 计算实际容器数 (通过 output 目录下的子目录数)
if [ -d "output" ]; then
    # 统计目录数，包含 output 自身
    DIR_COUNT=$(find output -maxdepth 1 -type d | wc -l)
    # 减去 1 (即 output 目录本身)，得到真实的节点数
    CONTAINER_COUNT=$((DIR_COUNT - 1))
else
    echo "[!] 警告: 当前路径下未找到 output 目录，容器数将标记为 0"
    CONTAINER_COUNT=0
fi

# 2. 获取当前精确时间，用于文件命名和日志内容
CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")
# 文件名专用的时间格式 (去除空格和冒号，避免破坏文件系统命名规范)
FILE_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 3. 动态生成带有“节点数”和“时间戳”的日志文件名
# 效果类似: host_mem_1898_nodes_20260312_143000.csv
LOG_FILE="${LOG_DIR}/host_mem_${CONTAINER_COUNT}_nodes_${FILE_TIMESTAMP}.csv"

# 4. 写入 CSV 表头 (因为每次都是新文件，所以直接写入即可)
echo "Timestamp,Nodes,Used_Mem_MB,Slab_Mem_KB" > "$LOG_FILE"

# 5. 获取核心数据
# 获取系统已用物理内存 (单位: MB)
USED_MEM=$(free -m | awk '/^Mem:/ {print $3}')
# 获取内核 Slab 内存 (单位: KB)
SLAB_MEM=$(awk '/^Slab:/ {print $2}' /proc/meminfo)

# 6. 追加核心数据到刚才创建的文件中
echo "${CURRENT_TIME},${CONTAINER_COUNT},${USED_MEM},${SLAB_MEM}" >> "$LOG_FILE"

# 7. 终端完美回显
echo "[*] 快照已记录 -> 规模: ${CONTAINER_COUNT} 节点 | 时间: $CURRENT_TIME"
echo "    |_ 已用物理内存 : ${USED_MEM} MB"
echo "    |_ 内核 Slab 缓存: ${SLAB_MEM} KB"
echo "    |_ 独立快照文件 : $LOG_FILE"
