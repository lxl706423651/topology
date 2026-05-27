import subprocess
import re
import time
import sys
import random
from collections import defaultdict

# ================= 配置区域 =================
# 节点启动间隔
NODE_DELAY = 0.05

# 智能检查配置
CHECK_INTERVAL = 2.0  # 每次检查等待的秒数
MIN_WAIT = 2.0        # 启动完后至少等待的时间

# BIRD 命令
BIRD_START_CMD = "bird -d"
# 用于检查协议状态的命令
BIRD_CHECK_CMD = "birdc show protocols"
# ===========================================

def get_containers():
    """获取容器列表"""
    try:
        result = subprocess.run(
            ['docker', 'ps', '--format', '{{.ID}} {{.Names}}'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        lines = result.stdout.decode('utf-8').strip().split('\n')
        return [line.split() for line in lines if line]
    except subprocess.CalledProcessError as e:
        print(f"Error executing docker ps: {e}")
        sys.exit(1)

def parse_and_group(containers):
    """分组逻辑"""
    as_groups = defaultdict(list)
    regex = re.compile(r'^as(\d+)brd')
    for cid, cname in containers:
        match = regex.match(cname)
        if match:
            as_groups[int(match.group(1))].append((cid, cname))
    return as_groups

def is_bird_running(container_id):
    """检查 BIRD 是否已在运行"""
    try:
        # 尝试执行 birdc show status。如果成功(返回码0)，说明 bird 已经在运行。
        # 这里把 stdout/stderr 扔掉，只关心 returncode
        cmd = f"docker exec {container_id} birdc show status"
        result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except:
        return False

def start_bird(container_id, container_name):
    """启动 BIRD (支持断点续传)"""
    # 1. 先检查是否已经运行
    if is_bird_running(container_id):
        # 可选：如果你想看到跳过的日志，取消注释下面这行
        # print(f"    ⏩ {container_name} is already running. Skipping start.")
        return

    # 2. 未运行则启动
    try:
        subprocess.run(f"docker exec -d {container_id} {BIRD_START_CMD}", shell=True, check=True)
    except:
        pass

def check_node_health(container_id, container_name):
    """
    检查单个节点的健康状况
    返回: True (健康/收敛完成), False (未完成)
    """
    try:
        # 执行 birdc show protocols
        cmd = f"docker exec {container_id} {BIRD_CHECK_CMD}"
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        
        if "ospf" not in output:
            print('ERROR:',output)
            return False

        # 简单解析输出
        # 1. 检查 OSPF 是否 Running
        if "ospf" in output and "Running" not in output:
            # 单节点 AS 可能 OSPF 为 Solo 状态或类似，如果只有1个节点，OSPF检查可以放宽
            # 但此处我们主要靠外部的 len(nodes) 判断来跳过
            return False 

        # 2. 检查 iBGP 是否 Established
        # 只有当 iBGP 也 Established 了，才说明 OSPF 路由表真的稳定可用了
        lines = output.split('\n')
        ibgp_sessions = 0
        established_sessions = 0
        
        for line in lines:
            if "ibgp" in line: # 这是一个 iBGP 协议
                ibgp_sessions += 1
                if "Established" in line:
                    established_sessions += 1
        
        # 如果配置了 iBGP 且没有全部 Established，则视为未收敛
        if ibgp_sessions > 0 and established_sessions < ibgp_sessions:
            return False
            
        return True

    except Exception:
        return False

def wait_for_convergence(nodes, as_num, start_time):
    """
    智能等待 AS 收敛（无限等待模式）
    """
    print(f"   ⏳ Verifying AS {as_num} convergence...", end='', flush=True)
    time.sleep(MIN_WAIT) 
    
    # 选取探针：随机抽样检查，避免检查所有节点消耗过多资源
    if len(nodes) > 3:
        probes = [nodes[0], nodes[1], nodes[-1]]
    else:
        probes = nodes

    # 循环计数器
    attempt = 0
    
    while True:
        all_ready = True
        for cid, cname in probes:
            if not check_node_health(cid, cname):
                all_ready = False
                break
        
        if all_ready:
            # 收敛完成！计算耗时
            end_time = time.time()
            duration = end_time - start_time
            print(f" ✅ Converged! Total time: {duration:.2f}s")
            return
        
        # 未收敛，继续等待
        attempt += 1
        
        # 【修改点】计算真实的流逝时间，而不是简单的计数器
        current_elapsed = time.time() - start_time
        
        # 每10秒换行打印一次，避免一行太长，同时显示当前等待时间
        if attempt % 5 == 0:
            print(f" [{current_elapsed:.0f}s] ", end='', flush=True)
        else:
            print(".", end='', flush=True)
            
        time.sleep(CHECK_INTERVAL)

def main():
    print("🔍 Fetching containers...")
    containers = get_containers()
    as_groups = parse_and_group(containers)
    
    if not as_groups:
        print("❌ No AS groups found.")
        return

    sorted_as = sorted(as_groups.keys())
    print(f"🚀 Smart Starting {len(as_groups)} AS groups (Infinite Wait Mode)...")

    for i, as_num in enumerate(sorted_as):
        nodes = as_groups[as_num]
        
        # 判断类型
        first_name = nodes[0][1]
        is_ix = "-ix" in first_name
        is_single_node = len(nodes) == 1
        
        print(f"[{i+1}/{len(sorted_as)}] Starting AS {as_num} ({len(nodes)} nodes)...")
        
        # 【记录开始时间】
        start_time = time.time()
        
        # 1. 批量启动 (增加了状态检查)
        for cid, cname in nodes:
            start_bird(cid, cname)
            time.sleep(NODE_DELAY)
            
        # 2. 智能等待逻辑
        if is_ix:
            print(f"   ✨ AS {as_num} (IX) - Skipping check.")
        elif is_single_node:
            print(f"   ✨ AS {as_num} (Single Node) - Skipping check.")
            time.sleep(MIN_WAIT) 
        else:
            # 只有多节点的 Router AS 需要检查 iBGP 收敛
            wait_for_convergence(nodes, as_num, start_time)

    print("-" * 50)
    print("🎉 Smart Start Complete!")

if __name__ == "__main__":
    main()