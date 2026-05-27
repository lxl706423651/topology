#!/bin/bash

# 定义函数来执行命令并统计时间
execute_and_time() {
    # $1 是描述，$2 是具体的命令字符串
    local desc="$1"
    local cmd="$2"
    
    echo "----------------------------------------"
    echo "开始执行: $desc"
    
    # 定义 TIMEFORMAT 变量，让 time 只输出纯数字（秒数）
    # %R 表示以秒为单位的时间
    TIMEFORMAT=%R
    
    # 技巧：
    # 1. (time ...) 2>&1  -> 将 time 的输出（原本在 stderr）重定向到 stdout
    # 2. 我们使用 eval 来执行传入的命令字符串，处理复杂的管道和引号
    # 3. 这里的 time 是 bash 内置命令
    
    elapsed=$( { time eval "$cmd" > /dev/null; } 2>&1 )
    
    echo "$desc 执行完成，耗时: ${elapsed} 秒"
}

# --- 具体的命令 ---

# 注意：我加了 -r 参数给 xargs，防止没有容器/网络时报错
# 使用单引号 '' 包裹命令，这样里面的双引号 "" 就不需要转义了，看着更清爽
cmd1='docker ps -aq | xargs -r -P $(nproc) -n 50 docker rm -f'
execute_and_time "删除所有Docker容器" "$cmd1"

cmd2='docker network ls --filter "type=custom" -q | xargs -r -P $(nproc) -n 50 docker network rm'
execute_and_time "删除自定义Docker网络" "$cmd2"

echo "----------------------------------------"
echo "所有操作完成！"