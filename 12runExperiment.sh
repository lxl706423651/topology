#!/bin/bash

# ==========================================
# SEED Emulator 全链路自动化实验总控脚本
# ==========================================

# 1. 自动计算当前节点规模 (基于 output 目录)
if [ -d "output" ]; then
    NODE_COUNT=$(($(find output -maxdepth 1 -type d | wc -l) - 1))
    
    # 先计算除法
    iperfPairNum=$(( NODE_COUNT / 100 ))
    
    # 使用三元运算符：如果小于16，就赋值为16；否则保持原值
    iperfPairNum=$(( iperfPairNum < 16 ? 16 : iperfPairNum ))
else
    NODE_COUNT="unknown"
    # 目录不存在时，也确保保底为 16 对
    iperfPairNum=16  
fi

echo "节点总数: $NODE_COUNT"
echo "实际生成的 iperf 对数: $iperfPairNum"

# 2. 生成全局专属实验目录: evaluationResult/节点数_时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# 使用绝对路径，防止子脚本执行目录不同导致混乱
export EXP_LOG_DIR="$(pwd)/evaluationResult/${NODE_COUNT}nodes_${TIMESTAMP}"

# 创建主目录和需要的子分类目录
mkdir -p "$EXP_LOG_DIR"
mkdir -p "$EXP_LOG_DIR/memory"
mkdir -p "$EXP_LOG_DIR/mrt"

echo "==================================================="
echo "[*] 开始自动化实验流水线!"
echo "[*] 实验规模: $NODE_COUNT 节点"
echo "[*] 所有日志将全局定向至: $EXP_LOG_DIR"
echo "==================================================="

# [步骤 0]
echo -e "\n---> [0] 统计初始服务器内存"
./0SnapHostMemory.sh

# [步骤 1 & 2]
echo -e "\n---> [1-2] 构建并拉起 Docker 拓扑"
./buildTest1.sh
./dockerUp.sh

# [步骤 3]
echo -e "\n---> [3] 注入 BIRD MRT 配置"
./3InjectMrtlog.sh

# # [步骤 4.a & 4.b] 启动后台监控进程
# echo -e "\n---> [4.a/b] 启动 eBPF 和 pidstat 后台监控 (针对初始收敛)"
# sudo -E python3 4profier.py &
# PROFILER_PID_1=$!  # 记录下这个后台进程的 PID，稍后结束它

# [步骤 4] 启动路由计算并等待稳态
echo -e "\n---> [4] 启动 BIRD，等待初始路由收敛 (load_average < 50)"
sudo -E python3 start_bird0130.py

# # 杀掉第一阶段的 eBPF 监控
# sudo kill -INT $PROFILER_PID_1
# sleep 2

# [步骤 5]
echo -e "\n---> [5] 统计收敛后内存快照"
sudo -E python3 5MemorySnapshot.py
./0SnapHostMemory.sh

# [步骤 6]
echo -e "\n---> [6] 回收 MRT 日志并清理配置"
./6CollectMrt.sh

# [步骤 7] 第一次震荡测试
echo -e "\n---> [7] 注入 10 节点失效，观测原始收敛时间"
sudo -E python3 7AutoConvergenceTest.py

# # [步骤 8]
# echo -e "\n---> [8] 重启 eBPF 和 pidstat 监控 (针对内核优化后)"
# sudo -E python3 4profier.py &
# PROFILER_PID_2=$!

# [步骤 9]
echo -e "\n---> [9] 写入内核优化策略 (start_bird_kernel.py)"
sudo -E python3 start_bird_kernel.py

# # 清理最后残留的监控进程
# sudo kill -INT $PROFILER_PID_2
# [步骤 10]
echo -e "\n---> [10] 统计内核优化后的内存快照"
# sudo -E python3 5MemorySnapshot.py
./0SnapHostMemory.sh

# [步骤 10.5] 生成iperf pair
sudo python3 0preGenerateIperfPairs.py -n $iperfPairNum

# [步骤 11]
echo -e "\n---> [11] 启动 iperf 并发吞吐量测试"
sudo -E python3 8IperfTest.py -t 10 -u_bw 20M

# # [步骤 12] 第二次震荡测试
# echo -e "\n---> [12] 再次注入 10 节点失效，观测优化后收敛时间"
# sudo -E python3 7AutoConvergenceTest.py

echo -e "\n==================================================="
echo "[✅] 实验完美结束！"
echo "所有多维度数据已封存于: $EXP_LOG_DIR"
echo "==================================================="

docker stop seedemu_internet_map
cd ~/seed-emulator1211/tools/InternetMap2 && dcbuild && docker compose up -d
