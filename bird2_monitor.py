import subprocess
import re
import time
import sys
import datetime
import os
from collections import defaultdict

# ================= 配置区域 =================
# 监测间隔 (秒)
MONITOR_INTERVAL = 30

# 极速启动时的微小间隔，防止 docker 进程死锁
START_DELAY = 0.05

BIRD_START_CMD = "bird -d"
BIRD_CHECK_CMD = "birdc show protocols"
# ===============================================

class Logger:
    def __init__(self):
        # 生成带时间戳的日志文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"./logs/bird_monitor/bird_monitor_{timestamp}.log"
        self.file = open(self.filename, "w", encoding='utf-8')
        
        header = f"=== BIRD Massive Start & Monitor Log ===\nStart Time: {timestamp}\n========================================\n"
        print(header, end='')
        self.file.write(header)
        self.file.flush()

    def log(self, message):
        # 获取当前时间用于每行日志
        now = datetime.datetime.now().strftime("[%H:%M:%S]")
        full_msg = f"{now} {message}"
        
        # 同时输出到屏幕和文件
        print(full_msg)
        self.file.write(full_msg + "\n")
        self.file.flush()
        
    def close(self):
        self.file.close()

# 全局日志对象
logger = None

def get_containers():
    try:
        # 获取 ID 和 Names
        result = subprocess.run(['docker', 'ps', '--format', '{{.ID}} {{.Names}}'], stdout=subprocess.PIPE, check=True)
        lines = result.stdout.decode('utf-8').strip().split('\n')
        return [line.split() for line in lines if line]
    except Exception as e:
        if logger: logger.log(f"Error getting containers: {e}")
        return []

def parse_and_group(containers):
    as_groups = defaultdict(list)
    # 修改正则：支持 'brd' 和 'r' 后缀，甚至更多变体
    # ^as(\d+)  : 匹配 as 开头后跟数字 (AS号)
    # (?:brd|r) : 匹配 brd 或 r，(?:...) 表示非捕获组
    regex = re.compile(r'^as(\d+)(?:brd|r)')
    
    for cid, cname in containers:
        match = regex.match(cname)
        if match: 
            as_groups[int(match.group(1))].append((cid, cname))
    return as_groups

def start_bird_async(container_id):
    """
    非阻塞启动 BIRD (Fire and Forget)
    """
    try:
        subprocess.Popen(f"docker exec -d {container_id} {BIRD_START_CMD}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass

def is_bird_running(container_id):
    """
    检查 BIRD 进程是否存在
    """
    try:
        # 使用 birdc show status 检查守护进程是否响应
        # 这种方式比 ps aux 更可靠，因为它确认 bird 已经准备好接收命令
        cmd = f"docker exec {container_id} birdc show status"
        res = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    except:
        return False

def check_node_health(container_id):
    """
    检查节点健康状态
    返回: True (OSPF Running 且 iBGP Established), False (未就绪)
    """
    try:
        # 同步执行检查
        cmd = f"docker exec {container_id} {BIRD_CHECK_CMD}"
        res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 1. 进程未运行
        if res.returncode != 0: return False
        
        output = res.stdout.decode('utf-8')
        if not output: return False
        
        # 2. 检查 OSPF
        if "ospf" not in output:
            logger.log("bird error", output)
            return False 
        if "ospf" in output and "Running" not in output: return False
        
        # 3. 检查 iBGP
        lines = output.split('\n')
        ibgp_total = 0
        ibgp_est = 0
        for line in lines:
            if "ibgp" in line:
                ibgp_total += 1
                if "Established" in line: ibgp_est += 1
        
        # 如果有配置 iBGP 但没全部连上，算失败
        if ibgp_total > 0 and ibgp_est < ibgp_total: return False
        
        # 既没有 OSPF 也没有 iBGP？视为空配置，算通过或失败看需求，这里暂算通过
        return True
    except:
        return False

def main():
    global logger
    logger = Logger()
    
    logger.log("🔍 Fetching containers...")
    containers = get_containers()
    as_groups = parse_and_group(containers)
    
    # 1. 筛选监测目标 (Transit AS 的第一个节点)
    monitor_targets = []  # 格式: (cid, cname, as_num)
    all_routers = []      # 格式: (cid, cname)
    
    logger.log("🧩 Analyzing network topology...")
    
    sorted_as = sorted(as_groups.keys())
    
    for as_num in sorted_as:
        nodes = as_groups[as_num]
        
        # 将所有节点加入启动列表
        for node in nodes:
            all_routers.append(node)
            
        # 筛选逻辑：
        # 1. 节点数 > 1 (排除 Stub AS)
        # 2. 名字不含 -ix (排除 IX)
        is_single = len(nodes) == 1
        first_node_name = nodes[0][1]
        is_ix = "-ix" in first_node_name
        
        if not is_single and not is_ix:
            # 这是一个 Transit AS，取第一个节点作为监测探针
            target_node = nodes[0]
            monitor_targets.append((target_node[0], target_node[1], as_num))
            
    logger.log(f"📊 Topology Stats:")
    logger.log(f"   Total Routers: {len(all_routers)}")
    logger.log(f"   Total ASs: {len(as_groups)}")
    logger.log(f"   Transit ASs to Monitor: {len(monitor_targets)}")
    
    # 2. 阶段一：全量启动
    logger.log("-" * 40)
    logger.log(f"🚀 Phase 1: Mass Startup initiated...")
    start_time = time.time()
    
    for i, (cid, cname) in enumerate(all_routers):
        start_bird_async(cid)
        # 极小的延迟，避免系统调用堆积导致报错
        if i % 100 == 0: time.sleep(0.1) 
        else: time.sleep(START_DELAY)
        
    startup_duration = time.time() - start_time
    logger.log(f"✅ Startup commands sent to {len(all_routers)} routers in {startup_duration:.2f}s")
    
    # 4. 阶段二：循环监测
    logger.log("-" * 40)
    logger.log(f"👀 Phase 2: Monitoring Mode (Interval: {MONITOR_INTERVAL}s)")
    logger.log("   Waiting for convergence...")
    
    # 初次等待
    time.sleep(5) 
    
    check_round = 0
    while True:
        check_round += 1
        
        logger.log(f"   [Round {check_round}] Checking {len(monitor_targets)} Transit AS probes...")
        
        success_count = 0
        failed_list = []
        
        for cid, cname, as_num in monitor_targets:
            if check_node_health(cid):
                success_count += 1
            else:
                failed_list.append(f"AS{as_num}")
        
        total = len(monitor_targets)
        ratio = success_count / total if total > 0 else 1.0
        percentage = ratio * 100
        
        logger.log(f"   -> Result: {success_count}/{total} ready ({percentage:.2f}%)")
        
        if ratio == 1.0:
            total_time = time.time() - start_time
            logger.log("🎉 All Transit ASs converged!")
            logger.log(f"⏱️ Total Execution Time: {total_time:.2f}s")
            break
        else:
            if len(failed_list) < 10:
                logger.log(f"      Pending: {', '.join(failed_list)}")
            else:
                logger.log(f"      Pending count: {len(failed_list)} ASs")
                
            logger.log(f"   ... Sleeping {MONITOR_INTERVAL}s ...")
            time.sleep(MONITOR_INTERVAL)

    logger.close()

if __name__ == "__main__":
    main()