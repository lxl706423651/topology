import subprocess
import re
import time
import sys
import datetime
import os
from collections import defaultdict
import json

# ================= 配置区域 =================
PROTOCOL_MAP = {
    # eBGP: 正则匹配 x_as 后跟数字 -> BIRD 通配符 "x_as*"
    r"^Ebgp_.*": '"Ebgp*"',
    
    # iBGP: 匹配 to_rr_... 或 to_cli_... -> BIRD 通配符 "to_*"
    r"^Ibgp_.*": '"Ibgp*"' 
}
PID_CACHE_FILE = "container_pids.json"  # PID 缓存文件路径
CONTAINER_FILTER = "as"                # 容器名过滤关键字

# 协议命名正则
REGEX_IBGP = r"^Ibgp_.*"  # 匹配 iBGP 协议
REGEX_EBGP = r"^Ebgp_.*"  # 匹配 eBGP 协议

# BIRD 启动命令
BIRD_START_CMD = "bird"

# --- 优化核心：系统负载判定参数 ---
SYSTEM_LOAD_THRESHOLD = 250.0  # 当 1分钟 Load Average 低于此值时，认为系统收敛/空闲
LOAD_CHECK_INTERVAL = 30       # 判定间隔(秒)
START_DELAY = 0.02             # 启动容器间的微小间隔（防止瞬间 CPU 爆表）
# ===========================================

class Logger:
    def __init__(self):
        if not os.path.exists("logs"):
            os.makedirs("logs")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"logs/bird_orchestrator/bird_orchestrator_{timestamp}.log"
        self.file = open(self.filename, "w", encoding='utf-8')
        
        header = f"=== BIRD Transit-Core Orchestrator (Load-Based) ===\nStart Time: {timestamp}\n=====================================================\n"
        print(header, end='')
        self.file.write(header)
        self.file.flush()

    def log(self, message):
        now = datetime.datetime.now().strftime("[%H:%M:%S]")
        full_msg = f"{now} {message}"
        print(full_msg)
        self.file.write(full_msg + "\n")
        self.file.flush()
        
    def close(self):
        self.file.close()

# 全局日志对象
logger = None

def init_container_pids():
    """扫描 Docker 容器 PID 并保存到 JSON"""
    logger.log("🔍 Scanning Docker containers for PIDs...")
    try:
        ps_cmd = f"docker ps -q -f name={CONTAINER_FILTER}"
        ids_output = subprocess.check_output(ps_cmd, shell=True).decode().strip()
        if not ids_output:
            logger.log("❌ No containers found.")
            return {}
        
        ids = ids_output.split()
        pid_map = {}
        ids_str = " ".join(ids)
        inspect_cmd = f"docker inspect -f '{{{{.Name}}}} {{{{.State.Pid}}}}' {ids_str}"
        output = subprocess.check_output(inspect_cmd, shell=True).decode().strip()
        
        for line in output.split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0].lstrip('/')
                pid = parts[1]
                pid_map[name] = pid
        
        with open(PID_CACHE_FILE, 'w') as f:
            json.dump(pid_map, f, indent=4)
            
        logger.log(f"✅ Cached {len(pid_map)} PIDs to {PID_CACHE_FILE}")
        return pid_map
    except Exception as e:
        logger.log(f"❌ Error initializing PIDs: {e}")
        return {}

def load_pids():
    """读取 PID 缓存，如果不存在则初始化"""
    return init_container_pids()
    if not os.path.exists(PID_CACHE_FILE):
        return init_container_pids()
    with open(PID_CACHE_FILE, 'r') as f:
        return json.load(f)

def run_nsenter_cmd(pid, shell_cmd, async_mode=False):
    """使用 nsenter 进入容器执行任意 Shell 命令"""
    # -n 网络, -m 挂载
    full_cmd = f"nsenter -t {pid} -n -m {shell_cmd}"
    if async_mode:
        subprocess.Popen(full_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None
    else:
        return subprocess.run(full_cmd, shell=True, capture_output=True, text=True)

def start_all_birds(pid_map):
    """第一阶段：启动所有节点的 BIRD 进程"""
    logger.log(f"🚀 Phase 1: Starting BIRD processes in {len(pid_map)} nodes...")
    count = 0
    for name, pid in pid_map.items():
        # 检查 BIRD 是否已在运行 (简单判断方式)
        check_cmd = f"nsenter -t {pid} -n -m pgrep bird"
        res = subprocess.run(check_cmd, shell=True, capture_output=True)
        
        if res.returncode != 0: # 不在运行
            run_nsenter_cmd(pid, BIRD_START_CMD, async_mode=True)
            count += 1
            if count % 200 == 0:
                logger.log(f" -> Started {count} nodes...")
                time.sleep(5)
            time.sleep(START_DELAY)
        else:
            # logger.log(f" ℹ️  Node {name} BIRD already running.")
            pass
            
    logger.log(f"✅ Finished triggering BIRD start on {count} new nodes.")

def batch_manage_protocols_smart(pid_map, regex_pattern, action="enable"):
    """智能批量操作协议（支持通配符优化）"""
    logger.log(f"📡 Batch {action} initiated (Regex: {regex_pattern})...")
    bird_glob = PROTOCOL_MAP.get(regex_pattern)
    count = 0
    
    if bird_glob:
        # 极速模式：下发通配符
        for name, pid in pid_map.items():
            cmd = f"birdc '{action} {bird_glob}'"
            run_nsenter_cmd(pid, cmd, async_mode=True) # 协议操作通常很快，可以用异步
            count += 1
        logger.log(f" -> Broadcast '{action} {bird_glob}' to {count} containers.")
    else:
        # 兼容模式：逐个操作
        pattern = re.compile(regex_pattern)
        for name, pid in pid_map.items():
            res = run_nsenter_cmd(pid, "birdc 'show protocols'")
            if not res or res.returncode != 0: continue
            
            for line in res.stdout.split('\n'):
                parts = line.split()
                if not parts or parts[0] == "Name": continue
                proto_name = parts[0]
                if pattern.match(proto_name):
                    run_nsenter_cmd(pid, f"birdc '{action} {proto_name}'", async_mode=True)
                    count += 1
        logger.log(f" -> Individually {action}d {count} protocols.")

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
def inject_bird_log_config(pid_map, target_names):
    """
    在指定容器的 /etc/bird/bird.conf 首行插入日志配置。
    具备幂等性：如果已存在则跳过。
    """
    # 要插入的日志配置行
    LOG_LINE = 'log "/var/log/bird.log" {debug, trace, info, remote, warning, error, fatal, bug};'
    CONFIG_FILE = "/etc/bird/bird.conf"
    
    logger.log(f"📝 Configuring BIRD logging for {len(target_names)} containers...")
    
    count_success = 0
    count_skipped = 0
    
    for name in target_names:
        if name not in pid_map:
            logger.log(f"⚠️ Container '{name}' not found in PID map, skipping.")
            continue

        pid = pid_map[name]
        
        # 1. 检查配置是否已存在 (使用 grep)
        # -F: 固定字符串, -q: 静默模式
        # 注意：这里我们需要转义引号以适应 shell 命令
        check_cmd = f"grep -Fq 'log \"/var/log/bird.log\"' {CONFIG_FILE}"
        res = run_nsenter_cmd(pid, check_cmd)
        
        if res.returncode == 0:
            # grep 返回 0 表示找到了，说明已经配置过
            # logger.log(f" ℹ️  {name}: Log config already exists.")
            count_skipped += 1
            continue
            
        # 2. 如果不存在，使用 sed 插入到第一行
        # sed -i '1i <内容>' <文件>
        # 这里的单引号和双引号嵌套需要非常小心
        sed_cmd = f"sed -i '1i {LOG_LINE}' {CONFIG_FILE}"
        
        inject_res = run_nsenter_cmd(pid, sed_cmd)
        
        if inject_res.returncode == 0:
            # 3. (可选) 如果 BIRD 已经在运行，需要 reload 才能生效
            # 简单判断 bird 进程是否存在，存在则 configure
            reload_cmd = "pgrep bird && birdc configure"
            run_nsenter_cmd(pid, reload_cmd, async_mode=True)
            
            logger.log(f"✅ {name}: Log config injected and reloaded.")
            count_success += 1
        else:
            logger.log(f"❌ {name}: Failed to inject config via sed.")

    logger.log(f"📝 Log injection finished. Injected: {count_success}, Skipped: {count_skipped}.")

def main():
    global logger
    logger = Logger()
    if os.geteuid() != 0:
        logger.log("❌ Error: Must be run as ROOT (sudo) to use nsenter.")
        sys.exit(1)

    # 0. 准备工作：获取 PID
    pid_map = load_pids()
    if not pid_map:
        logger.log("❌ No target containers found. Exit.")
        return

    total_start_time = time.time()

    # 1. 启动所有 BIRD 进程
    start_all_birds(pid_map)
    time.sleep(60)
    wait_for_system_idle("Phase 1: BIRD Base Convergence")
    # 2. 启动 iBGP
    logger.log("🔗 Phase 2: Enabling iBGP sessions...")
    batch_manage_protocols_smart(pid_map, REGEX_IBGP, action="enable")
    time.sleep(60)
    wait_for_system_idle("Phase 2: iBGP Convergence")

    # 3. 启动 eBGP
    logger.log("🌐 Phase 3: Enabling eBGP sessions...")
    batch_manage_protocols_smart(pid_map, REGEX_EBGP, action="enable")
    time.sleep(60)
    wait_for_system_idle("Phase 3: Final Network Convergence")

    total_duration = time.time() - total_start_time
    logger.log("-" * 45)
    logger.log(f"🎉 Orchestration Completed in {total_duration:.2f}s!")
    logger.close()

if __name__ == "__main__":
    main()