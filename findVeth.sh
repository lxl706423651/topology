#!/bin/bash

# 1. 检查参数
if [ -z "$1" ]; then
    echo "用法: $0 <容器ID或名称>"
    exit 1
fi

CONTAINER=$1

# 2. 检查容器是否存在
if ! docker inspect "$CONTAINER" > /dev/null 2>&1; then
    echo "错误: 找不到容器 '$CONTAINER'"
    exit 1
fi

# 3. 第一步：获取容器内的接口名和索引ID
# 输出格式示例: net0 4844
# 逻辑：只匹配包含 @if 的行，去掉冒号，提取名称和ID
IFACE_LIST=$(docker exec "$CONTAINER" ip addr | awk '/@if/ {
    sub(/:$/, "", $2);       # 去掉第二列末尾的冒号
    split($2, parts, "@if"); # 按 @if 分割
    print parts[1], parts[2] # 打印名称(net0) 和 ID(4844)
}')

# 4. 第二步：遍历每个接口，去宿主机查询详情
# 读取变量 IFACE_LIST，按行处理
echo "$IFACE_LIST" | while read -r if_name if_id; do
    # 如果 ID 为空则跳过
    if [ -z "$if_id" ]; then continue; fi

    # 在宿主机执行 ip link show 并 grep 索引号
    # 示例输出: 5262: vethcd50985@if2: <BROADCAST...> ... master br-b3fc5b43a086 ...
    HOST_INFO=$(ip link show | grep "^$if_id:")

    # 如果宿主机找不到对应接口（可能容器已停止或网络异常），跳过
    if [ -z "$HOST_INFO" ]; then
        echo "$if_name: (在宿主机未找到对应的 veth)"
        continue
    fi

    # 解析宿主机输出，提取 veth 名称和 master 网桥
    OUTPUT=$(echo "$HOST_INFO" | awk '{
        # $2 是类似 vethcd50985@if2: 的字符串
        # 我们按 @ 分割，取第一部分得到 vethcd50985
        split($2, v, "@");
        veth_name = v[1];

        # 遍历所有列寻找 "master" 关键字，它的下一列就是网桥名
        bridge_name = "";
        for (i=1; i<=NF; i++) {
            if ($i == "master") {
                bridge_name = $(i+1);
                break;
            }
        }
        
        # 格式化输出: vethcd50985 master br-b3fc5b43a086
        if (bridge_name != "") {
            print veth_name, "master", bridge_name
        } else {
            print veth_name, "(无网桥)"
        }
    }')

    # 最终输出格式: net0: vethcd50985 master br-b3fc5b43a086
    echo "${if_name}: ${OUTPUT}"

done