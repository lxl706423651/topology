import subprocess
import re
import time
import sys
import datetime
import os
import json
import random
from collections import defaultdict

# ================= 配置区域 =================
SYSTEM_LOAD_THRESHOLD = 110.0  # 判定系统收敛的 Load Average 阈值
LOAD_CHECK_INTERVAL = 30      # 监控频率（秒）
PID_CACHE_FILE = "container_pids.json"

# 用于匹配并提取 ASN 的正则表达式 (例如: as1277brd-r219-1.219.4.253 提取出 1277)
ROUTER_NAME_REGEX = re.compile(r"^as(\d+)brd-r")

# 你的 kernel 协议在 BIRD 中的名称
# 提示：BIRD2 默认的 IPv4 kernel 协议名称通常是 kernel1，请根据 birdc show protocols 的结果修改
KERNEL_PROTO_NAME = "kernel1" 

# 实验目标：制造故障的容器列表
TARGET_CONTAINERS = [
    "as1277brd-r219-1.219.4.253",
    "as1304brd-r40-1.40.5.24",
    "as1841brd-r17-1.17.7.49",
    "as1488brd-r12-1.12.5.208",
    "as1485brd-r18-1.18.5.205",
    "as1725brd-r21-1.21.6.189",
    "as1113brd-ix1113-5.89.4.89",
    "as1700brd-r13-1.13.6.164",
    "as1797brd-r12-1.12.7.5",
    "as1531brd-r7-1.7.5.251",
]

# 路由更新间隔与分批延迟
BATCH_INTERVAL = 1  # 恢复阶段每个容器间的微小延迟，防止并发冲击内核

# ================= 工具函数 =================

class Logger:
    def __init__(self):
        if not os.path.exists("logs"):
            os.makedirs("logs")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"logs/exp_fault_convergence_{timestamp}.log"
        self.file = open(self.filename, "w", encoding='utf-8')
        print(f"=== 实验启动: RIB/FIB 解耦收敛脚本 ===\n日志文件: {self.filename}\n")

    def log(self, message):
        now = datetime.datetime.now().strftime("[%H:%M:%S]")
        full_msg = f"{now} {message}"
        print(full_msg)
        self.file.write(full_msg + "\n")
        self.file.flush()

logger = Logger()

def run_nsenter_cmd(pid, shell_cmd):
    """进入容器命名空间执行命令"""
    full_cmd = f"nsenter -t {pid} -n -m {shell_cmd}"
    return subprocess.run(full_cmd, shell=True, capture_output=True, text=True)

def wait_for_system_idle(phase_name):
    """
    监控宿主机 Load Average。
    逻辑：如果 Load > SYSTEM_LOAD_THRESHOLD，进入'冷却模式'，
    必须等待 Load 降到 RECOVERY_THRESHOLD (10) 以下才继续。
    """
    RECOVERY_THRESHOLD = 50.0  # 你要求的恢复阈值
    waiting_for_recovery = False # 标记：是否正在等待深度冷却

    logger.log(f"⏳ {phase_name}: Checking System Load (Threshold: {SYSTEM_LOAD_THRESHOLD})...")
    
    while True:
        try:
            load1, _, _ = os.getloadavg()
            
            # 1. 判断是否触发高负载逻辑
            # 一旦超过高阈值，就标记为 True，之后必须降到 10 才能变回 False
            if load1 > SYSTEM_LOAD_THRESHOLD:
                waiting_for_recovery = True
            
            # 2. 根据当前状态决定 目标阈值 (Target)
            # 如果处于冷却模式，目标是 10；否则目标是原定的阈值
            current_target = RECOVERY_THRESHOLD if waiting_for_recovery else SYSTEM_LOAD_THRESHOLD

            # 3. 显示状态
            # 如果当前负载高于目标，显示红灯；否则绿灯
            status_symbol = "🟢" if load1 < current_target else "🔴"
            mode_str = "[Cooling Down]" if waiting_for_recovery else "[Normal Check]"
            
            sys.stdout.write(f"\r {status_symbol} {mode_str} Load: {load1:.2f} (Target: <{current_target}) ")
            sys.stdout.flush()
            
            # 4. 判断是否满足退出条件
            if load1 < current_target:
                print("") # 换行
                logger.log(f"✅ System stabilized. Current Load: {load1:.2f}.")
                break
                
        except Exception as e:
            logger.log(f"Error reading load avg: {e}")
            break
            
        time.sleep(LOAD_CHECK_INTERVAL)

# ================= 主实验流程 =================

def main():
    if os.geteuid() != 0:
        logger.log("❌ 必须以 ROOT 权限运行。")
        return

    # 1. 加载 PID 映射
    if not os.path.exists(PID_CACHE_FILE):
        logger.log(f"❌ 找不到 {PID_CACHE_FILE}")
        return
    with open(PID_CACHE_FILE, 'r') as f:
        pid_map = json.load(f)

    # 过滤出所有路由器容器（排除掉已经明确要 down 的目标）
    all_routers = {k: v for k, v in pid_map.items() if "brd-r" in k}
    alive_routers = {k: v for k, v in all_routers.items() if k not in TARGET_CONTAINERS}

    logger.log(f"🚀 准备开始实验。总节点数: {len(all_routers)}, 存活节点数: {len(alive_routers)}")

    # --- 阶段 1: 全局冻结 FIB ---
    logger.log(f"第一阶段: 正在为所有节点应用 FIB 冻结策略 (禁用 {KERNEL_PROTO_NAME} 协议)...")
    for name, pid in all_routers.items():
        # 核心修改：直接 disable 协议，只要配置文件中有 persist; 现存的内核路由就不会丢
        run_nsenter_cmd(pid, f"birdc disable {KERNEL_PROTO_NAME}")
        
    logger.log("✅ 全局 FIB 锁定完成。原内核路由完好保留，新的 BGP 更新将不会写入内核。")

    # --- 阶段 2: 制造节点故障 ---
    logger.log(f"第二阶段: 正在制造节点故障，停止容器 {len(TARGET_CONTAINERS)} 个...")
    for name in TARGET_CONTAINERS:
        if name in pid_map:
            pid = pid_map[name]
            logger.log(f"   🔻 Down node: {name}")
            run_nsenter_cmd(pid, "birdc down")
    
    # --- 阶段 3: 等待收敛 ---
    # 此阶段 Bird 内部 RIB 快速收敛，没有任何内核 RTNL 锁干预
    time.sleep(90) # 给系统一个负载爬升的缓冲期
    wait_for_system_idle("RIB convergence")

    # --- 阶段 4: 分批恢复 FIB 写入 ---
    logger.log(f"第四阶段: 正在分批恢复存活节点的 FIB 写入权限 (启用 {KERNEL_PROTO_NAME} 协议)...")
    processed = 0
    total_alive = len(alive_routers)
    
    # ============= 修改区域: 按照 ASN 分组并排序 =============
    logger.log("📊 正在按 ASN 对存活节点进行分组排序...")
    as_groups = defaultdict(list)
    
    for name, pid in alive_routers.items():
        match = ROUTER_NAME_REGEX.match(name)
        if match:
            asn = int(match.group(1))
            as_groups[asn].append((name, pid))
        else:
            # 如果正则表达式没有成功匹配（通常不会发生），默认放入 ASN 0 分组
            as_groups[0].append((name, pid))

    sorted_asns = sorted(as_groups.keys())
    # =========================================================

    for asn in sorted_asns:
        logger.log(f"🏢 Processing AS {asn}...")
        for name, pid in as_groups[asn]:
            # 核心修改：重新 enable 协议，BIRD 会智能对比差异并进行增量写入
            run_nsenter_cmd(pid, f"birdc enable {KERNEL_PROTO_NAME}")
            
            processed += 1
            if processed % 10 == 0:
                logger.log(f"⏳ 进度: {processed}/{total_alive} 节点已恢复...")
                
            time.sleep(BATCH_INTERVAL)
            wait_for_system_idle("FIB convergence")

    print("\n")
    logger.log("🎉 实验完成。所有存活节点的 FIB 已根据最新的 RIB 完成增量同步。")

if __name__ == "__main__":
    main()
