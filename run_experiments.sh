#!/bin/bash
set -e

# 1. 开启脚本中的别名扩展功能 (必须开启，脚本默认关闭)
shopt -s expand_aliases

# 2. 直接在这里定义别名 (把你的命令贴在这里)
# 假设你的命令是 docker compose up -d，请替换为你真实的命令
alias up='python dockerBuild_Up.py 1 5 50 16 && python dockerBuild_Up.py 2 5 50 16'
alias down='docker ps -aq | xargs -P $(nproc) -n 50 docker rm -f && docker network ls --filter "type=custom" -q | xargs -P $(nproc) -n 50 docker network rm'

# 3. 开始执行任务
echo "=== 开始执行任务 1 ==="
#python ./autocoder1.py 1078 && up && python ./bird2_monitor.py && down

echo "=== 开始执行任务 2 ==="
python ./autocoder1.py 1897 && up && python ./bird2_monitor.py && down

echo "=== 开始执行任务 3 ==="
python ./autocoder1.py 2599 && up && python ./bird2_monitor.py && down

echo "=== 开始执行任务 4 ==="
python ./autocoder1.py 3083 && up && python ./bird2_monitor.py && down

# echo "=== 开始执行任务 16 ==="
# python ./autocoder1.py 214 && up && python ./bird2_monitor.py && down

# echo "=== 开始执行任务 24 ==="
# python ./autocoder1.py 214 && up && python ./bird2_monitor.py && down

echo "✅ 所有任务执行完成！"
