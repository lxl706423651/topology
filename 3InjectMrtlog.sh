#!/bin/bash

# 定义目标容器和配置文件路径
CONTAINER_NAME="as1269brd-r12-1.12.4.245"
CONFIG_FILE="/etc/bird/bird.conf"
MRT_LOG_DIR="/var/log/bird"

CURRENT_TS=$(date +"%Y%m%d_%H%M%S")
echo "开始为容器 $CONTAINER_NAME 配置 BGP MRT Dump..."

# 步骤 1: 确保容器内存在存放 MRT 文件的目录
docker exec "$CONTAINER_NAME" mkdir -p "$MRT_LOG_DIR"

# 步骤 2: 使用 sed 在配置文件的最顶端（第 1 行）插入 MRT 配置
# 注意：这里假设你使用的是 BIRD 2.x 语法。%s 会以 Unix 时间戳命名文件以防止覆盖
docker exec "$CONTAINER_NAME" sed -i '1i mrtdump "'$MRT_LOG_DIR'/bgp_routes-'$CURRENT_TS'.mrt";\nmrtdump protocols all;\n' "$CONFIG_FILE"
# 步骤 3: 优雅重载 BIRD 配置（极其重要）
# 使用 birdc configure 可以让 BIRD 重新读取配置文件，而无需重启容器或 BIRD 进程
# 这保证了你现有的网络拓扑和 BGP 会话不会因为重启而发生不必要的断开和振荡
docker exec "$CONTAINER_NAME" birdc configure

echo "配置注入完成！你可以通过 docker exec $CONTAINER_NAME ls $MRT_LOG_DIR 检查 MRT 文件是否开始生成。"
