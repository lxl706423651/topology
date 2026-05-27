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

SYSTEM_LOAD_THRESHOLD = 200.0  # 当 1分钟 Load Average 低于此值时，认为系统收敛/空闲
LOAD_CHECK_INTERVAL = 30       # 判定间隔(秒)

PID_CACHE_FILE = "container_pids.json"

# 目标容器名称过滤：只匹配路由器 (-r)，忽略交换机 (-ix)
# 匹配逻辑：as[数字]brd-r...
ROUTER_NAME_REGEX = re.compile(r"^as(\d+)brd-r")

# 新的 Kernel 协议配置块 (全开模式)
# 注意：协议名必须是你现在使用的 "kernel"
NEW_KERNEL_CONFIG2 = """
protocol kernel {{
    merge paths on;
    persist;
    scan time {interval};
    ipv4 {{
        import none;
        export all;
    }};
}}
"""

NEW_KERNEL_CONFIG1 = """
protocol kernel {{
    merge paths on;
    persist;
    scan time {interval};
    ipv4 {{
        import none;
        # 核心修改在这里：加一个过滤器
        export filter {{
            # 允许直连路由写入内核（保证互联互通）
            if source = RTS_DEVICE then accept;
            # 允许 OSPF 路由写入内核（保证 iBGP Loopback 可达）
            if source = RTS_OSPF then accept;
            # 拒绝其他所有路由（包括 BGP 路由）写入内核！
            reject;
        }};
    }};
}}
"""
# 每次操作后的间隔 (秒)
INTERVAL = 0.5
# ===========================================

class Logger:
    def __init__(self):
        if not os.path.exists("logs"):
            os.makedirs("logs")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"logs/switch_kernel_simple_{timestamp}.log"
        self.file = open(self.filename, "w", encoding='utf-8')
        print(f"=== BIRD Kernel Switcher (Protocol 'kernel') ===\nLog: {self.filename}\n")

    def log(self, message):
        now = datetime.datetime.now().strftime("[%H:%M:%S]")
        full_msg = f"{now} {message}"
        print(full_msg)
        self.file.write(full_msg + "\n")
        self.file.flush()
        
    def close(self):
        self.file.close()

logger = None

def load_pids():
    if not os.path.exists(PID_CACHE_FILE):
        logger.log("❌ PID cache not found. Please run the init script first.")
        return {}
    with open(PID_CACHE_FILE, 'r') as f:
        return json.load(f)

def run_nsenter_cmd(pid, shell_cmd):
    """在容器内同步执行命令"""
    full_cmd = f"nsenter -t {pid} -n -m {shell_cmd}"
    return subprocess.run(full_cmd, shell=True, capture_output=True, text=True)

def read_remote_conf(pid):
    """读取容器内的配置文件"""
    res = run_nsenter_cmd(pid, "cat /etc/bird/conf/kernel.conf")
    if res.returncode != 0:
        return None
    return res.stdout

def write_remote_conf(pid, content):
    """写回配置文件"""
    temp_file = f"temp_bird_{pid}.conf"
    try:
        with open(temp_file, "w", encoding='utf-8') as f:
            f.write(content)
        cmd = f"cat {temp_file} | nsenter -t {pid} -n -m tee /etc/bird/conf/kernel.conf > /dev/null"
        subprocess.run(cmd, shell=True, check=True)
        return True
    except Exception as e:
        logger.log(f"❌ Write failed: {e}")
        return False
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def replace_kernel_block(content):
    """
    修改后的逻辑：
    不再去分析原文件内容，直接返回新的配置字符串。
    这会将目标文件 (kernel.conf) 的内容完全覆盖为 NEW_KERNEL_CONFIG。
    """
    # 直接忽略传入的 content (旧内容)
    # 直接返回新配置
    t=6000+random.randint(0, 1200)
    return NEW_KERNEL_CONFIG2.format(interval=t), "Success"


def group_router_containers(pid_map):
    """
    只筛选路由器 (-r) 容器
    """
    logger.log("📊 Grouping ROUTER containers (filtering out IX nodes)...")
    as_groups = defaultdict(list)
    
    matched_count = 0
    ignored_count = 0
    
    for name, pid in pid_map.items():
        match = ROUTER_NAME_REGEX.match(name)
        if match:
            asn = int(match.group(1))
            as_groups[asn].append((name, pid))
            matched_count += 1
        else:
            ignored_count += 1
            
    logger.log(f"✅ Selected {matched_count} Routers. Ignored {ignored_count} IX/Other nodes.")
    return as_groups

def wait_for_system_idle(phase_name):
    """监控宿主机 Load Average，等待系统收敛"""
    logger.log(f"⏳ {phase_name}: Waiting for System Load (1min) < {SYSTEM_LOAD_THRESHOLD}...")
    while True:
        try:
            load1, _, _ = os.getloadavg()
            status_symbol = "🟢" if load1 < SYSTEM_LOAD_THRESHOLD else "🔴"
            sys.stdout.write(f"\r {status_symbol} System Load: {load1:.2f} (Target: <{SYSTEM_LOAD_THRESHOLD}) ")
            sys.stdout.flush()
            
            if load1 < SYSTEM_LOAD_THRESHOLD:
                print("") 
                logger.log(f"✅ System stabilized. Current Load: {load1:.2f}.")
                break
        except Exception:
            break
        time.sleep(LOAD_CHECK_INTERVAL)

def main():
    global logger
    logger = Logger()
    
    if os.geteuid() != 0:
        logger.log("❌ Must be run as ROOT.")
        sys.exit(1)

    pid_map = load_pids()
    if not pid_map: return

    # 获取过滤后的路由器分组
    as_groups = group_router_containers(pid_map)
    sorted_asns = sorted(as_groups.keys())
    
    total_count = sum(len(v) for v in as_groups.values())
    processed = 0

    if total_count == 0:
        logger.log("⚠️ No router containers found. Check naming convention.")
        return

    logger.log(f"🚀 Starting configuration update on {total_count} ROUTER nodes...")
    t=time.time()
    for asn in sorted_asns:
        logger.log(f"🏢 Processing AS {asn}...")
        
        for name, pid in as_groups[asn]:
            # 1. 读取配置
            conf = read_remote_conf(pid)
            if not conf:
                logger.log(f"⚠️  {name}: Read failed.")
                continue

            # 2. 替换配置块
            new_conf, msg = replace_kernel_block(conf)
            if not new_conf:
                # 简单的幂等检查：如果已经是 export all 了，就不报错了
                if "import all; export all;" in conf.replace("\n", "").replace(" ", ""): 
                    pass
                else:
                    logger.log(f"⚠️  {name}: Replace failed - {msg}")
                continue
            
            # 3. 写入并重载
            if write_remote_conf(pid, new_conf):
                # [关键修改]
                # 协议名变了，所以 reload 命令必须变成 "reload kernel"
                # configure 用于更新配置，reload 用于强制刷新现有路由
                reload_cmd = "birdc configure && birdc reload kernel"
                
                run_nsenter_cmd(pid, reload_cmd)
                
                processed += 1
                logger.log(f"\r⏳ Progress: {processed}/{total_count}")
                sys.stdout.write(f"\r⏳ Progress: {processed}/{total_count}")
                sys.stdout.flush()
            wait_for_system_idle("Inter-Container Idle Wait")
            time.sleep(INTERVAL)


    print("")
    logger.log("🎉 All routers updated to Export All.")
    t2=time.time()
    logger.log(f"⏱ Total time: {t2 - t:.2f}) seconds.")
    logger.close()

if __name__ == "__main__":
    main()
