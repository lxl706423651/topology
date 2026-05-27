#!/bin/bash

# =================配置区域=================
# 检查参数
if [ "$#" -ne 2 ]; then
    echo "用法: $0 <AS号> <操作>"
    echo "------------------------------------------------"
    echo "可用操作 (Action):"
    echo "  reload     : 重置所有协议 (birdc restart all) -> 模拟软重启"
    echo "  ospf       : 仅重置 OSPF (birdc restart ospf1)"
    echo "  stop       : 优雅关闭 BIRD 进程 (birdc down)"
    echo "  start_dbg  : 启动 BIRD (带 -d 参数，输出日志到 stderr)"
    echo "  check      : 查看 BIRD 运行状态 (ps aux)"
    echo "------------------------------------------------"
    echo "示例: $0 1659 ospf"
    exit 1
fi

AS_ID=$1
ACTION=$2
PATTERN="as${AS_ID}brd-r"

# 1. 获取容器列表
echo "🔍 正在搜索 AS${AS_ID} 的运行中容器..."
CONTAINERS=$(docker ps --format "{{.Names}}" | grep "^${PATTERN}")

if [ -z "$CONTAINERS" ]; then
    echo "❌ 未找到正在运行的 AS${AS_ID} 容器。"
    echo "   (注意：此脚本只操作已启动的容器内的进程，请先确保 Docker 容器是 Up 状态)"
    exit 1
fi

COUNT=$(echo "$CONTAINERS" | wc -l)
echo "✅ 找到 $COUNT 个容器，正在执行: $ACTION"
echo "------------------------------------------------"

# 2. 循环处理每个容器
# 这里使用循环而不是 xargs，为了针对不同命令做逻辑判断
for container in $CONTAINERS; do
    echo -n "👉 处理容器 [$container]: "
    
    case "$ACTION" in
        reload)
            # 重置所有协议，相当于 BIRD 软重启
            docker exec $container birdc restart all
            ;;
            
        ospf)
            # 仅重置名为 ospf1 的协议实例
            # 注意：如果你的配置文件里协议名叫 MyOSPF，请修改这里的 ospf1
            docker exec $container birdc restart ospf1
            ;;
            
        stop)
            # 告诉 BIRD 进程优雅退出
            docker exec $container birdc down
            echo "BIRD 进程已停止 (Sent down command)"
            ;;
            
        start)
            # 在后台执行 bird -d
            # -d: 启用调试模式并前台运行 (配合 docker logs 使用)
            # docker exec -d: 让这个命令在 Docker 后台跑，不卡住脚本
            docker exec -d $container bird -d
            echo "BIRD 已启动 (bird -d mode)"
            ;;
            
        check)
            # 检查进程是否存在
            docker exec $container ps aux | grep bird | grep -v grep
            ;;
            
        *)
            echo "❌ 未知操作: $ACTION"
            exit 1
            ;;
    esac
done

echo "------------------------------------------------"
echo "🎉 所有操作执行完毕！"