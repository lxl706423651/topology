#!/bin/bash

# ==========================================
# MRT 路由表文件回收与 BIRD 配置还原脚本
# ==========================================

CONTAINER_NAME="as1269brd-r12-1.12.4.245"
BASE_DIR="${EXP_LOG_DIR:-./logs}"

# 例如内存脚本可以这样写：
HOST_DEST_DIR="$BASE_DIR/mrt"
mkdir -p "$HOST_DEST_DIR"
CONFIG_FILE="/etc/bird/bird.conf"

echo "[*] 开始处理容器: $CONTAINER_NAME"

# 1. 确保宿主机的目标文件夹存在
mkdir -p "$HOST_DEST_DIR"
echo "[*] 宿主机目标目录已准备: $HOST_DEST_DIR"

# 2. 查找并传输 *.mrt 文件
# 使用 sh -c 配合 ls 确保通配符在容器内被正确解析
FILES=$(docker exec "$CONTAINER_NAME" sh -c 'ls /var/log/bird/*.mrt 2>/dev/null' || true)

if [ -z "$FILES" ]; then
    echo "[!] 容器内未找到任何 .mrt 文件，跳过传输步骤。"
else
    echo "[*] 正在传输 MRT 文件到宿主机..."
    for f in $FILES; do
        # 去除可能存在的换行符或回车符
        f=$(echo "$f" | tr -d '\r\n')
        docker cp "$CONTAINER_NAME:$f" "$HOST_DEST_DIR/"
        echo "    |_ 已导出: $f"
    done
    
    # (可选) 传输完成后，删除容器内的 MRT 文件以释放空间
    docker exec "$CONTAINER_NAME" sh -c 'rm -f /var/log/bird/*.mrt'
    echo "[*] 容器内的遗留 MRT 文件已清理。"
fi

# 3. 还原 bird.conf 配置
echo "[*] 正在从 $CONFIG_FILE 中移除 mrtdump 配置行..."
# 使用 sed 的 '/pattern/d' 命令，直接删除所有包含 'mrtdump' 的行
docker exec "$CONTAINER_NAME" sed -i '/mrtdump/d' "$CONFIG_FILE"

# 4. 优雅重载 BIRD 进程
echo "[*] 正在热重载 BIRD 配置以应用更改..."
docker exec "$CONTAINER_NAME" birdc configure > /dev/null

echo "[*] 处理完成！你可以去 $HOST_DEST_DIR 目录下检查你的 MRT 文件了。"
