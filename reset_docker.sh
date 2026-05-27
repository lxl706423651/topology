#!/bin/bash

# 遇到错误立即停止
set -e

# 定义一些颜色输出，方便看进度
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO] $1${NC}"
}

err() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# 检查 dcbuild 是否可用，如果只是别名，脚本中可能无法直接使用
# 这里尝试启用别名扩展
shopt -s expand_aliases
# 尝试加载用户的 bashrc 以获取别名 (假设 dcbuild 定义在这里)
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# 再次检查 dcbuild，如果还是没有，定义一个临时的函数
if ! command -v dcbuild &> /dev/null; then
    log "'dcbuild' 命令未找到，尝试将其定义为 'docker compose build'..."
    dcbuild() {
        docker compose build
    }
fi

log "1. 停止 Docker 相关服务..."
sudo systemctl stop docker.socket
sudo systemctl stop docker
sudo systemctl stop containerd

log "2. 删除 Docker 数据和配置 (危险操作)..."
sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd
sudo rm -rf /etc/docker
sudo rm -rf /var/run/docker
sudo rm -rf /var/run/containerd

log "3. 重新创建配置目录并写入 daemon.json..."
sudo mkdir -p /etc/docker

# 使用 tee 命令将配置写入文件 (非交互式)
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker.anyhub.us.kg",
    "https://dockerhub.icu",
    "https://docker.1panel.live"
  ]
}
EOF

log "4. 重载配置并重启 Docker..."
sudo systemctl daemon-reload
sudo systemctl start containerd
sudo systemctl start docker

# 等待 Docker 守护进程完全启动
log "等待 Docker 启动..."
sleep 3

# 检查 Docker 是否存活
if ! sudo docker info > /dev/null 2>&1; then
    err "Docker 启动失败，请检查日志。"
    exit 1
fi

# log "5. 开始构建 Seed Emulator 镜像..."

# 构建 Base
log "Building: seedemu-base"
cd "${HOME}/seed-emulator/docker_images/seedemu-base"
dcbuild

# 构建 Router
log "Building: seedemu-router"
cd "${HOME}/seed-emulator/docker_images/seedemu-router"
dcbuild

log "所有任务完成！"
