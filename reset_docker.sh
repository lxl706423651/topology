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

# 不配置第三方 registry mirror，直接使用官方 Docker Hub。
# 第三方代理缓存若与 Docker Hub 元数据不同步，会触发 failed size validation。
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{}
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

log "5. 通过官方 Docker Hub 拉取基础镜像..."
docker pull docker.io/library/ubuntu:24.04

# 构建 Base
log "Building: seedemu-base"
cd "${HOME}/seed-emulator/docker_images/seedemu-base"
docker compose build

# 构建 Router
log "Building: seedemu-router"
cd "${HOME}/seed-emulator/docker_images/seedemu-router"
docker compose build

log "所有任务完成！"
