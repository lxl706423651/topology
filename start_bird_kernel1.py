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

SYSTEM_LOAD_THRESHOLD = 200.0  # 系统高负载阈值
PID_CACHE_FILE = "container_pids.json"

# 目标容器名称过滤：只匹配路由器 (-r)，忽略交换机 (-ix)
ROUTER_NAME_REGEX = re.compile(r"^as(\d+)brd-r")

# 新的 Kernel 协议配置块
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

# 基础间隔 (秒) - 有了智能检测后，这个可以设得很小
INTERVAL = 0.5 

# 判定路由收敛的参数
CONVERGENCE_CHECK_INTERVAL = 0.5 # 每次检查间隔
CONVERGENCE_STABLE_COUNT = 3     # 需要连续几次数量不变才算稳定 (4 * 0.5 = 2秒稳定期)
MIN_ROUTE_COUNT = 10             # 忽略少于10条路由的情况（防止空表被误判）
TIMEOUT_PER_CONTAINER = 60       # 单个容器最大等待时间，防止死锁

# ===========================================

class Logger:
    def __init__(self):
        if not os.path.exists("logs"):
            os.makedirs("logs")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"logs/switch_kernel_smart_{timestamp}.log"
        self.file = open(self.filename, "w", encoding='utf-8')
        print(f"=== BIRD Kernel Switcher (Smart Convergence) ===\nLog: {self.filename}\n")

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
    res = run_nsenter_cmd(pid, "cat /etc/bird/conf/kernel.conf")
    if res.returncode != 0:
        return None
    return res.stdout

def write_remote_conf(pid, content):
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
    # 随机化 scan time，防止所有容器同时扫描内核
    t = 60000 + random.randint(0, 12000)
    return NEW_KERNEL_CONFIG2.format(interval=t), "Success"

def group_router_containers(pid_map):
    logger.log("📊 Grouping ROUTER containers...")
    as_groups = defaultdict(list)
    matched_count = 0
    
    for name, pid in pid_map.items():
        match = ROUTER_NAME_REGEX.match(name)
        if match:
            asn = int(match.group(1))
            as_groups[asn].append((name, pid))
            matched_count += 1
            
    logger.log(f"✅ Selected {matched_count} Routers.")
    return as_groups

# --- 新增：智能收敛检测函数 ---
def wait_for_route_convergence(pid, container_name):
    """
    监控容器内核路由表条目数。
    当条目数不再增长且保持稳定时，返回 True。
    """
    start_time = time.time()
    last_count = -1
    stable_hits = 0
    
    # 简单的进度显示字符
    spinner = ['|', '/', '-', '\\']
    idx = 0

    while True:
        # 1. 超时保护
        if time.time() - start_time > TIMEOUT_PER_CONTAINER:
            logger.log(f"⚠️ {container_name}: Timeout waiting for routes (Count: {last_count})")
            return False

        # 2. 获取当前内核路由数
        # 使用 ip -4 route 避免统计 ipv6，wc -l 统计行数
        res = run_nsenter_cmd(pid, "ip -4 route show | wc -l")
        try:
            current_count = int(res.stdout.strip())
        except ValueError:
            current_count = 0

        # 3. 判定逻辑
        # 如果数量没变，且不是空表 (>10)
        if current_count == last_count and current_count > MIN_ROUTE_COUNT:
            stable_hits += 1
        else:
            # 如果数量还在变（变大或变小），重置计数器
            stable_hits = 0
            # 只有当数量确实变了才更新 last_count，确保判定连续性
            last_count = current_count

        # 4. 动态输出状态
        # 清除当前行，显示进度
        elapsed = time.time() - start_time
        sys.stdout.write(f"\r   ⏳ {container_name}: {current_count} routes | Stable: {stable_hits}/{CONVERGENCE_STABLE_COUNT} | {spinner[idx%4]} ")
        sys.stdout.flush()
        idx += 1

        # 5. 成功退出条件
        if stable_hits >= CONVERGENCE_STABLE_COUNT:
            sys.stdout.write(f" -> Done ({elapsed:.1f}s)\n") # 换行
            return True

        time.sleep(CONVERGENCE_CHECK_INTERVAL)

def wait_for_system_idle(phase_name):
    """
    监控宿主机 Load Average。
    逻辑：如果 Load > SYSTEM_LOAD_THRESHOLD，进入'冷却模式'，
    必须等待 Load 降到 RECOVERY_THRESHOLD (10) 以下才继续。
    """
    RECOVERY_THRESHOLD = 10.0 
    waiting_for_recovery = False 

    # 快速检查，如果负载低直接跳过，避免日志刷屏
    load1, _, _ = os.getloadavg()
    if load1 < SYSTEM_LOAD_THRESHOLD:
        return

    logger.log(f"🔥 {phase_name}: High Load ({load1:.2f}), entering cool-down mode...")
    
    while True:
        try:
            load1, _, _ = os.getloadavg()
            
            if load1 > SYSTEM_LOAD_THRESHOLD:
                waiting_for_recovery = True
            
            current_target = RECOVERY_THRESHOLD if waiting_for_recovery else SYSTEM_LOAD_THRESHOLD
            status_symbol = "🟢" if load1 < current_target else "🔴"
            mode_str = "[Cooling]" if waiting_for_recovery else "[Check]"
            
            sys.stdout.write(f"\r {status_symbol} {mode_str} Load: {load1:.2f} (Target: <{current_target}) ")
            sys.stdout.flush()
            
            if load1 < current_target:
                print("") 
                logger.log(f"✅ System stabilized. Current Load: {load1:.2f}.")
                break
                
        except Exception as e:
            logger.log(f"Error reading load avg: {e}")
            break
            
        time.sleep(30) # 冷却模式下检查间隔可以长一点

def main():
    global logger
    logger = Logger()
    
    if os.geteuid() != 0:
        logger.log("❌ Must be run as ROOT.")
        sys.exit(1)

    pid_map = load_pids()
    if not pid_map: return

    as_groups = group_router_containers(pid_map)
    sorted_asns = sorted(as_groups.keys())
    
    total_count = sum(len(v) for v in as_groups.values())
    processed = 0

    if total_count == 0:
        logger.log("⚠️ No router containers found.")
        return

    logger.log(f"🚀 Starting update on {total_count} nodes with Smart Convergence Check...")
    
    t_start = time.time()
    
    for asn in sorted_asns:
        # 记录当前AS的处理情况
        # logger.log(f"🏢 AS {asn} ...") 
        
        for name, pid in as_groups[asn]:
            # --- 1. 读取配置 ---
            conf = read_remote_conf(pid)
            if not conf:
                continue

            # --- 2. 判断是否需要修改 ---
            if "import all; export all;" in conf.replace("\n", "").replace(" ", ""): 
                # 已经改过了，但可能上次中断了，我们依然执行 wait_for_route_convergence
                logger.log(f"ℹ️  {name}: Already configured, checking convergence...")
            else:
                # 执行替换和写入
                new_conf, _ = replace_kernel_block(conf)
                if write_remote_conf(pid, new_conf):
                    # --- 3. 触发重载 ---
                    # 使用 configure 确保配置生效
                    # 使用 reload kernel 强制刷新路由协议
                    cmd = "birdc configure && birdc reload kernel"
                    run_nsenter_cmd(pid, cmd)
                else:
                    continue
            
            # --- 4. 智能等待收敛 (关键修改) ---
            # 不再死等固定时间，而是等路由表写完
            wait_for_route_convergence(pid, name)
            
            processed += 1

            wait_for_system_idle(f"Batch Check {processed}/{total_count}")
            
    print("")
    logger.log("🎉 All routers updated.")
    logger.log(f"⏱ Total time: {time.time() - t_start:.2f} seconds.")
    logger.close()

if __name__ == "__main__":
    main()
