#!/bin/bash

# ================= 配置区域 =================
# 监测时长 (秒)
DURATION=10000
# 日志保存目录
LOG_DIR="./logs/rtnl"
# 获取当前时间戳 (格式: YYYYMMDD_HHMMSS，例如 20260202_153000)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# ===========================================

# 1. 准备工作
# 创建目录 (如果不加 -p，目录不存在会报错)
mkdir -p "$LOG_DIR"

# 刷新 sudo 权限，防止后台任务因缺权限卡住
sudo -v

echo "📂 日志目录: $LOG_DIR"
echo "⏱️  时间戳ID: $TIMESTAMP"
echo "⏳ 监测时长: ${DURATION}秒"

# 2. 启动监测工具组合 (放入后台运行 &)

# [工具 A] 耗时分布：最能证明“卡顿”的证据
# -u: 以微秒显示; -d: 持续时长
echo "   -> 启动 funclatency (耗时分布)..."
sudo /usr/sbin/funclatency-bpfcc -u -d $DURATION rtnl_lock > "${LOG_DIR}/rtnl_latency_${TIMESTAMP}.txt" &

# [工具 B] 频率趋势：证明“震荡”的证据
# -i 1: 每秒输出一次; -d: 持续时长
echo "   -> 启动 funccount (频率趋势)..."
sudo /usr/sbin/funccount-bpfcc -i 1 -d $DURATION 'rtnl_lock' > "${LOG_DIR}/rtnl_trend_${TIMESTAMP}.txt" &

# [工具 C] 调用来源：证明“罪魁祸首”是 BIRD 内核同步
# -D: stackcount 的时长参数是大写 D
echo "   -> 启动 stackcount (调用栈)..."
sudo /usr/sbin/stackcount-bpfcc -D $DURATION rtnl_lock > "${LOG_DIR}/rtnl_stack_${TIMESTAMP}.txt" &

# 3. 等待所有后台任务结束
echo "=================================================="
echo "🚀 监测已在后台启动，请立即运行你的 BIRD 实验..."
echo "=================================================="

# wait 会阻塞当前脚本，直到上面所有后台进程结束
wait

echo ""
echo "✅ 监测结束 ($DURATION秒 已到)。"
echo "📊 耗时分布: ${LOG_DIR}/rtnl_latency_${TIMESTAMP}.txt"
echo "📊 频率趋势: ${LOG_DIR}/rtnl_trend_${TIMESTAMP}.txt"
echo "📊 调用堆栈: ${LOG_DIR}/rtnl_stack_${TIMESTAMP}.txt"
