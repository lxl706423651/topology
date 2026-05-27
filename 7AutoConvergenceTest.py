#!/usr/bin/env python3
from bcc import BPF
import os
import time
import datetime
import subprocess
import threading
import sys

# ==========================================
# 1. 实验配置与双轨日志系统
# ==========================================
EXP_DIR = os.environ.get("EXP_LOG_DIR", "./logs")
LOG_DIR = os.path.join(EXP_DIR, "convergence") # 你可以自行分类

os.makedirs(LOG_DIR, exist_ok=True)
start_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

trace_file_path = os.path.join(LOG_DIR, f"lifecycle_trace_{start_time_str}.log")
session_file_path = os.path.join(LOG_DIR, f"lifecycle_session_{start_time_str}.log")

class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger(session_file_path)

# 【核心参数】
QUIET_WINDOW = 60.0  
WAIT_BETWEEN_PHASES = 10.0 # 两次实验之间的冷却等待时间

# TARGET_CONTAINERS = [
#     "as1272brd-r12-1.12.4.248",
#     "as1304brd-r9-1.9.5.24",
#     "as1518brd-r21-1.21.5.238",
#     "as1271brd-r40-1.40.4.247",
#     "as1277brd-r71-1.71.4.253",
#     "as1830brd-r7-1.7.7.38",
#     "as1269brd-r42-1.42.4.245",
#     "as1271brd-r7-1.7.4.247",
#     "as1450brd-r12-1.12.5.170",
#     "as1494brd-r7-1.7.5.214",
# ]
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
# 【双阶段状态机变量】
PHASE_DOWN = 1
PHASE_UP = 2
current_phase = PHASE_DOWN

t_start_unix = 0.0
t_last_event_unix = 0.0
action_injected = False
event_count_after_action = 0

# ==========================================
# 2. 构建容器映射
# ==========================================
def build_container_map():
    inode_to_name = {}
    print("[*] 正在扫描当前运行的 Docker 容器以构建 Netns 映射...")
    try:
        cids_output = subprocess.check_output(['docker', 'ps', '-q']).decode('utf-8').strip()
        if not cids_output:
            return inode_to_name
        cids = cids_output.split()
        inspect_fmt = '{{.Name}}|{{.State.Pid}}'
        cmd = ['docker', 'inspect', '-f', inspect_fmt] + cids
        inspect_out = subprocess.check_output(cmd).decode('utf-8').strip()
        for line in inspect_out.split('\n'):
            if not line: continue
            name, pid = line.split('|')
            name = name.lstrip('/')
            if pid != '0':
                try:
                    netns_path = f"/proc/{pid}/ns/net"
                    inode = os.stat(netns_path).st_ino
                    inode_to_name[inode] = name
                except FileNotFoundError:
                    pass
    except Exception as e:
        print(f"[!] 构建容器映射时出错: {e}")
    return inode_to_name

container_map = build_container_map()

# ==========================================
# 3. eBPF 内核探针代码
# ==========================================
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <linux/sched.h>
#include <linux/nsproxy.h>
#include <net/net_namespace.h>

#define THROTTLE_NS 500000000ULL

struct hash_key_t { u32 pid; u32 netns; u16 type; };
struct data_t { u32 pid; u32 netns; u16 type; char comm[TASK_COMM_LEN]; };

BPF_HASH(last_seen, struct hash_key_t, u64);
BPF_PERF_OUTPUT(events);

int trace_rtnetlink_rcv(struct pt_regs *ctx, struct sk_buff *skb, struct nlmsghdr *nlh) {
    struct data_t data = {}; struct hash_key_t key = {};
    u64 pid_tgid = bpf_get_current_pid_tgid();
    key.pid = pid_tgid >> 32; data.pid = key.pid;
    bpf_probe_read_kernel(&key.type, sizeof(key.type), &nlh->nlmsg_type);
    data.type = key.type;
    
    if (data.type != RTM_NEWROUTE && data.type != RTM_DELROUTE && data.type != RTM_GETROUTE) return 0;

    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    struct nsproxy *nsproxy; struct net *net; unsigned int inum = 0;
    
    bpf_probe_read_kernel(&nsproxy, sizeof(nsproxy), &task->nsproxy);
    if (nsproxy) {
        bpf_probe_read_kernel(&net, sizeof(net), &nsproxy->net_ns);
        if (net) bpf_probe_read_kernel(&inum, sizeof(inum), &net->ns.inum);
    }
    key.netns = inum; data.netns = inum;

    u64 *last_ts = last_seen.lookup(&key);
    u64 now = bpf_ktime_get_ns();
    if (last_ts && (now - *last_ts < THROTTLE_NS)) return 0;
    last_seen.update(&key, &now);

    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""
b = BPF(text=bpf_text)
b.attach_kprobe(event="rtnetlink_rcv_msg", fn_name="trace_rtnetlink_rcv")
op_map = {24: "ADD/UPDATE", 25: "DELETE", 26: "READ/DUMP"}

# ==========================================
# 4. 双阶段操作注入函数
# ==========================================
def inject_down():
    global t_start_unix, action_injected
    print("\n" + "▼"*50)
    print(f"[阶段 1/2] 正在对 {len(TARGET_CONTAINERS)} 个节点并发执行失效 (birdc down)...")
    t_start_unix = time.time()
    
    procs = [subprocess.Popen(["docker", "exec", c, "birdc", "down"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for c in TARGET_CONTAINERS]
    for p in procs: p.wait()
        
    action_injected = True
    print(f"[!] 失效注入完毕！等待全网路由撤销收敛 (静默窗口: {QUIET_WINDOW} 秒)...")
    print("▼"*50 + "\n")

def inject_up():
    global t_start_unix, action_injected, event_count_after_action
    
    # 状态重置
    action_injected = False
    event_count_after_action = 0
    
    print("\n" + "▲" * 50)
    print(f"[阶段 2/2] 正在通过 docker exec 重新拉起 {len(TARGET_CONTAINERS)} 个节点的 BIRD 进程...")
    
    # 记录极其精确的恢复起始时间 (T_start)
    t_start_unix = time.time()
    
    procs = []
    for c in TARGET_CONTAINERS:
        # 使用 docker exec 执行 bird 命令
        # -c 指定配置文件路径
        # 注意：如果容器内已经有遗留的控制套接字文件，可能需要先清理或 BIRD 会自动覆盖
        cmd = ["docker", "exec", c, "bird", "-c", "/etc/bird/bird.conf"]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
    
    # 等待所有启动命令下发完成
    for p in procs:
        p.wait()
        
    action_injected = True
    print(f"[!] BIRD 进程已重新拉起！等待 TCP 握手与 BGP 路由宣告收敛...")
    print("▲" * 50 + "\n")

# 定时器：启动 4 秒后执行第一阶段 (失效)
threading.Timer(4.0, inject_down).start()

# ==========================================
# 5. eBPF 事件处理回调
# ==========================================
header = f"{'TIMESTAMP':<24} {'PID':<8} {'COMM':<16} {'CONTAINER (INODE)':<35} {'OPERATION':<15}"
print(f"[*] 全生命周期测试台已就绪。双规日志已开启。等待第一阶段触发...\n")
print(header)

with open(trace_file_path, "a") as f: f.write(header + "\n")

def print_event(cpu, data, size):
    global t_last_event_unix, event_count_after_action
    event = b["events"].event(data)
    
    current_time = time.time()
    ts_str = datetime.datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    op_str = op_map.get(event.type, f"UNKNOWN ({event.type})")
    c_name = container_map.get(event.netns, "Host/Unknown")
    # 修正后的 log_line 构造
    container_info = f"{c_name} ({event.netns})"
    log_line = f"{ts_str:<24} {event.pid:<8} {event.comm.decode('utf-8'):<16} {container_info:<35} {op_str:<15}"
    print(log_line)
    
    with open(trace_file_path, "a") as f: f.write(log_line + "\n")
        
    if action_injected and event.type in [24, 25]:
        t_last_event_unix = current_time
        event_count_after_action += 1

b["events"].open_perf_buffer(print_event)

# ==========================================
# 6. 带状态机的主循环
# ==========================================
try:
    while current_phase <= PHASE_UP:
        b.perf_buffer_poll(timeout=500)
        
        # 必须至少产生了一次路由修改，且静默了 QUIET_WINDOW 才算收敛
        if action_injected and event_count_after_action > 0:
            current_time = time.time()
            if current_time - t_last_event_unix > QUIET_WINDOW:
                
                # 算账时间！
                convergence_time = t_last_event_unix - t_start_unix
                phase_name = "【节点失效 (Failure)】" if current_phase == PHASE_DOWN else "【节点恢复 (Recovery)】"
                
                print("\n" + "="*50)
                print(f"[✅] 自动检测到 {phase_name} 阶段的路由已收敛！")
                print(f"  -> 操作起始时间 (T_start): {datetime.datetime.fromtimestamp(t_start_unix).strftime('%H:%M:%S.%f')[:-3]}")
                print(f"  -> 最后路由更新 (T_end)  : {datetime.datetime.fromtimestamp(t_last_event_unix).strftime('%H:%M:%S.%f')[:-3]}")
                print(f"  -> 期间 FIB 更新操作次数 : {event_count_after_action} 次")
                print(f"  -> \033[92m绝对收敛时间: {convergence_time:.3f} 秒\033[0m")
                print("="*50 + "\n")
                
                # 状态机推进
                if current_phase == PHASE_DOWN:
                    current_phase = PHASE_UP
                    action_injected = False # 暂停记录，忽略期间的本底噪声
                    print(f"[*] 将在 {WAIT_BETWEEN_PHASES} 秒后开始测试节点恢复 (上线) 过程，请保持探针运行...")
                    threading.Timer(WAIT_BETWEEN_PHASES, inject_up).start()
                else:
                    print("[🎉] 全生命周期（失效+恢复）双阶段测试完美结束！")
                    sys.stdout = sys.stdout.terminal
                    sys.exit(0)
                
except KeyboardInterrupt:
    print("\n[!] 收到强行中断信号，退出监控。")
    sys.stdout = sys.stdout.terminal
    sys.exit(0)