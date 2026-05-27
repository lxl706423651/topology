import subprocess
import re
import time
import sys
import json
import os
from collections import defaultdict

# ================= 配置区域 =================
# AS 之间的等待时间 (秒)
BATCH_DELAY = 0.2 

# 节点之间的等待时间 (秒)
# 使用 nsenter 后非常快，0.05 是个安全值，如果机器性能好可以设为 0.01
NODE_DELAY = 0.05

# 关闭 BIRD 的具体命令
STOP_CMD = "birdc down"

# PID 文件路径
PID_FILE = "container_pids.json"
# ===========================================

def run_nsenter_cmd(pid, shell_cmd, async_mode=False):
    """
    使用 nsenter 进入容器执行任意 Shell 命令
    注意：此函数需要 root 权限运行
    """
    # -n: 进入网络命名空间
    # -m: 进入挂载命名空间 (为了能找到 birdc 命令和配置文件)
    # -t: 目标 PID
    full_cmd = f"nsenter -t {pid} -n -m {shell_cmd}"
    
    if async_mode:
        # 异步模式：扔出去执行就不管了，适合批量关闭场景
        subprocess.Popen(full_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None
    else:
        # 同步模式：等待结果返回
        return subprocess.run(full_cmd, shell=True, capture_output=True, text=True)

def load_pids(filepath):
    """读取 JSON 文件获取 Name -> PID 映射"""
    if not os.path.exists(filepath):
        print(f"❌ Error: {filepath} not found.")
        sys.exit(1)
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"❌ Error: Failed to decode {filepath}.")
        sys.exit(1)

def parse_and_group(pid_map):
    """根据容器名解析 AS 号并分组"""
    as_groups = defaultdict(list)
    # 这里的正则匹配的是容器名 key
    regex = re.compile(r'^as(\d+)brd')

    for cname, pid in pid_map.items():
        match = regex.match(cname)
        if match:
            as_num = int(match.group(1))
            # 存入 (pid, cname) 元组
            as_groups[as_num].append((pid, cname))
            
    return as_groups

def main():
    # 检查是否为 root，nsenter 必须用 root 运行
    if os.geteuid() != 0:
        print("⚠️  Warning: This script uses 'nsenter' which requires root privileges.")
        print("    Please run with 'sudo python3 script.py'")
        sys.exit(1)

    print(f"🔍 Loading PIDs from {PID_FILE}...")
    pid_map = load_pids(PID_FILE)
    
    print("🧩 Grouping by AS number...")
    as_groups = parse_and_group(pid_map)
    
    if not as_groups:
        print("❌ No matching containers found in JSON.")
        return

    total_routers = sum(len(nodes) for nodes in as_groups.values())
    print(f"✅ Found {total_routers} routers in {len(as_groups)} AS groups.")
    print(f"🛑 Stopping BIRD via nsenter | Batch Delay: {BATCH_DELAY}s | Node Delay: {NODE_DELAY}s")
    print("-" * 50)

    # 排序：倒序关闭（从大 AS 号开始）
    sorted_as = sorted(as_groups.keys(), reverse=True)

    for i, as_num in enumerate(sorted_as):
        nodes = as_groups[as_num]
        
        # 检查是否为 IX 节点 (用于日志显示)
        first_node_name = nodes[0][1]
        is_ix = "-ix" in first_node_name
        type_str = "IX" if is_ix else "Router AS"
        
        print(f"[{i+1}/{len(sorted_as)}] Stopping AS {as_num} ({len(nodes)} nodes, Type: {type_str})...")
        
        for pid, cname in nodes:
            # 执行核心逻辑
            # 使用 async_mode=True，因为我们不需要读取 "Bird is down" 的返回文本
            # 这样可以最大化发送指令的速度，靠 time.sleep 控制节奏
            run_nsenter_cmd(pid, STOP_CMD, async_mode=True)
            
            # 稍微延时，防止瞬间大量进程争抢 CPU
            if NODE_DELAY > 0:
                time.sleep(NODE_DELAY) 
            
        # 只有在不是最后一组时才等待
        if type_str == "IX":
            continue
        if i < len(sorted_as) - 1:
            time.sleep(BATCH_DELAY)

    print("-" * 50)
    print("🎉 All BIRD stop commands sent via nsenter!")

if __name__ == "__main__":
    main()