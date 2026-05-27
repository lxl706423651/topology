#!/bin/bash

# 定义日志文件和结果文件
LOG_PIPE="./logs/bpf_startup.pipe"
RESULT_FILE="./logs/rtnl_callers.txt"

# 清理旧的管道和文件
rm -f "$LOG_PIPE"
mkfifo "$LOG_PIPE"

echo "⏳ 正在启动 eBPF 追踪工具 (funccount & stackcount)..."
echo "⏳ 等待内核探针编译与挂载..."

# 1. 启动 stackcount (后台)
#    - stdout (数据) -> 写入结果文件
#    - stderr (启动日志) -> 同时输出到屏幕 和 命名管道
sudo /usr/sbin/stackcount-bpfcc rtnl_lock -D 120 \
    1> "$RESULT_FILE" \
    2> >(tee "$LOG_PIPE" >&2) & 
PID_STACK=$!

# 2. 启动 funccount (后台)
#    - stdout/stderr -> 同时输出到屏幕 和 命名管道
sudo /usr/sbin/funccount-bpfcc 'rtnl_lock' -d 120 \
    2>&1 | tee "$LOG_PIPE" &
PID_FUNC=$!

# 3. 核心逻辑：阻塞等待，直到在输出中看到 "Tracing" 关键字
#    grep -m 1 表示匹配到 1 次后立即退出，从而解除阻塞
grep -m 1 "Tracing" "$LOG_PIPE" > /dev/null

echo ""
echo "=========================================="
echo "✅ 探针已就绪！(Captured 'Tracing' signal)"
echo "🚀 此时执行 Bird 启动脚本最能抓到现场！"
echo "=========================================="

# === 选项 A: 脚本自动帮你执行 (取消下面注释并修改路径) ===
# python3 /path/to/your/start_bird_script.py

# === 选项 B: 如果你是手动在另一个窗口执行，现在可以动手了 ===
# 这里我们只是简单的等待两个追踪进程结束
wait $PID_FUNC $PID_STACK

echo "🏁 追踪结束。结果已保存至 $RESULT_FILE"
rm -f "$LOG_PIPE"
