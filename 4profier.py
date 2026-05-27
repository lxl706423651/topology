#!/usr/bin/env python3
from bcc import BPF
import os
import time
import datetime
import subprocess
import signal
import sys

# ==========================================
# 1. 基础配置与日志目录初始化
# ==========================================
EXP_DIR = os.environ.get("EXP_LOG_DIR", "./logs")
LOG_DIR = os.path.join(EXP_DIR, "seed_profiler") # 你可以自行分类

os.makedirs(LOG_DIR, exist_ok=True)

start_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
rtnl_log_path = os.path.join(LOG_DIR, f"rtnl_trace_{start_time_str}.log")
csw_log_path = os.path.join(LOG_DIR, f"csw_trace_{start_time_str}.log")

print(f"[*] 初始化实验日志目录: {LOG_DIR}")

# ==========================================
# 2. 构建 Docker 容器映射字典 (Inode -> Name)
# ==========================================
def build_container_map():
    inode_to_name = {}
    print("[*] 正在扫描 Docker 容器构建 Netns 映射...")
    try:
        cids_output = subprocess.check_output(['docker', 'ps', '-q']).decode('utf-8').strip()
        if not cids_output:
            print("[!] 警告: 当前没有运行中的 Docker 容器。")
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
        print(f"[!] 构建容器映射出错: {e}")
        
    print(f"[*] 成功映射 {len(inode_to_name)} 个容器。")
    return inode_to_name

container_map = build_container_map()

# ==========================================
# 3. 启动后台 pidstat 监控进程
# ==========================================
pidstat_process = None

def start_pidstat_monitor():
    global pidstat_process
    print("[*] 正在查找 BIRD 进程以启动宏观上下文切换监控...")
    try:
        # 获取所有 bird 进程 PID
        pids_output = subprocess.check_output(['pgrep', 'bird']).decode('utf-8').strip()
        if not pids_output:
            print("[!] 未找到 BIRD 进程，跳过 pidstat 监控。")
            return
            
        pid_list = pids_output.replace('\n', ',')
        
        # 写入精简后的 pidstat 表头
        with open(csw_log_path, "w") as f:
            f.write("Timestamp,UID,PID,cswch/s,nvcswch/s,Command\n")
            
        # 启动后台 pidstat (每秒采样 1 次)
        # 去掉 -u 参数，仅保留 -w (上下文切换)。
        # 利用 awk 提取 pidstat -h 原生输出的第1列(时间戳)、第2列(UID)、第3列(PID)、第4列(cswch/s)、第5列(nvcswch/s)和第6列(Command)
        awk_cmd = f"awk '/^[0-9]/ {{print $1\",\"$2\",\"$3\",\"$4\",\"$5\",\"$6}}' >> {csw_log_path}"
        full_cmd = f"pidstat -w -p {pid_list} -h 1 | {awk_cmd}"
        
        pidstat_process = subprocess.Popen(full_cmd, shell=True, preexec_fn=os.setsid)
        print(f"[*] 后台 pidstat 已启动 (仅关注上下文切换)，日志写入: {csw_log_path}")
    except subprocess.CalledProcessError:
        print("[!] 未找到 BIRD 进程，跳过 pidstat 监控。")

start_pidstat_monitor()

# ==========================================
# 4. 定义并编译 BPF 代码 (锁时间 + Netns)
# ==========================================
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/nsproxy.h>
#include <net/net_namespace.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>

BPF_HASH(wait_start, u32, u64);
BPF_HASH(hold_start, u32, u64);
BPF_HASH(temp_wait, u32, u64);
// 新增：用于跨函数传递当前线程正在处理的操作类型
BPF_HASH(current_op, u32, u16); 

struct data_t {
    u32 pid;
    u32 netns;
    char comm[TASK_COMM_LEN];
    u64 wait_us;
    u64 hold_us;
    u16 op_type; // 新增：操作类型
};

BPF_PERF_OUTPUT(lock_events);

// 挂载 rtnetlink_rcv_msg 提取操作类型
int trace_rtnetlink_rcv_msg(struct pt_regs *ctx, struct sk_buff *skb, struct nlmsghdr *nlh) {
    u32 pid = bpf_get_current_pid_tgid();
    u16 type = 0;
    bpf_probe_read_kernel(&type, sizeof(type), &nlh->nlmsg_type);
    
    // 🛡️ 核心优化：只捕获真正的路由表操作 (24=ADD, 25=DEL, 26=GET)
    // 防止在 Netlink 批量发送 (Batching) 时，被其他底层无关消息覆盖了真实操作类型
    if (type == 24 || type == 25 || type == 26) {
        current_op.update(&pid, &type);
    }
    return 0;
}

int trace_rtnl_lock_entry(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    wait_start.update(&pid, &ts);
    return 0;
}

int trace_rtnl_lock_return(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    u64 *wsp = wait_start.lookup(&pid);
    if (wsp != 0) {
        u64 wait_us = (ts - *wsp) / 1000;
        wait_start.delete(&pid);
        temp_wait.update(&pid, &wait_us);
        hold_start.update(&pid, &ts);
    }
    return 0;
}

int trace_rtnl_unlock_entry(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();
    
    u64 *hsp = hold_start.lookup(&pid);
    u64 *w_us = temp_wait.lookup(&pid);
    
    if (hsp != 0 && w_us != 0) {
        struct data_t data = {};
        data.pid = pid;
        bpf_get_current_comm(&data.comm, sizeof(data.comm));
        data.wait_us = *w_us;
        data.hold_us = (ts - *hsp) / 1000;
        
        // 提取并清理操作类型
        u16 *op_ptr = current_op.lookup(&pid);
        if (op_ptr != 0) {
            data.op_type = *op_ptr;
            current_op.delete(&pid);
        } else {
            data.op_type = 0; // 0 代表非路由相关的内核内部持锁操作
        }
        
        // 抓取 Netns Inode
        struct task_struct *task = (struct task_struct *)bpf_get_current_task();
        struct nsproxy *nsproxy;
        struct net *net;
        unsigned int inum = 0;
        
        bpf_probe_read_kernel(&nsproxy, sizeof(nsproxy), &task->nsproxy);
        if (nsproxy) {
            bpf_probe_read_kernel(&net, sizeof(net), &nsproxy->net_ns);
            if (net) {
                bpf_probe_read_kernel(&inum, sizeof(inum), &net->ns.inum);
            }
        }
        data.netns = inum;
        
        lock_events.perf_submit(ctx, &data, sizeof(data));
        
        hold_start.delete(&pid);
        temp_wait.delete(&pid);
    }
    return 0;
}
"""

print("[*] 正在编译和挂载 eBPF 探针 (设置 maxactive 应对高并发)...")
b = BPF(text=bpf_text)
b.attach_kprobe(event="rtnetlink_rcv_msg", fn_name="trace_rtnetlink_rcv_msg")
b.attach_kprobe(event="rtnl_lock", fn_name="trace_rtnl_lock_entry")
# 极其重要：使用 maxactive 防止并发风暴导致 kretprobe 丢失
b.attach_kretprobe(event="rtnl_lock", fn_name="trace_rtnl_lock_return", maxactive=1024)
b.attach_kprobe(event="rtnl_unlock", fn_name="trace_rtnl_unlock_entry")

# ==========================================
# 5. 数据处理与主循环
# ==========================================

# 定义操作类型字典 (Netlink 协议号)
op_map = {
    24: "ADD/UPDATE",
    25: "DELETE",
    26: "READ/DUMP"
}

header = f"{'TIMESTAMP':<24} {'PID':<8} {'COMM':<16} {'OPERATION':<15} {'CONTAINER (INODE)':<35} {'WAIT (us)':<10} {'HOLD (us)':<10}"
print("\n" + header)

with open(rtnl_log_path, "w") as f:
    f.write(header + "\n")

def print_event(cpu, data, size):
    event = b["lock_events"].event(data)
    ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # 映射容器名
    c_name = container_map.get(event.netns, "Host/Unknown")
    container_info = f"{c_name} ({event.netns})"
    
    # 映射操作类型
    if event.op_type in op_map:
        op_str = op_map[event.op_type]
    elif event.op_type != 0:
        op_str = f"OTHER ({event.op_type})"
    else:
        # 对应你日志里的 kworker 等内核后台线程，它们持锁但不是为了改路由表
        op_str = "SYSTEM_TASK" 
    
    log_line = f"{ts_str:<24} {event.pid:<8} {event.comm.decode('utf-8'):<16} {op_str:<15} {container_info:<35} {event.wait_us:<10} {event.hold_us:<10}"
    
    print(log_line)
    with open(rtnl_log_path, "a") as f:
        f.write(log_line + "\n")

b["lock_events"].open_perf_buffer(print_event)

# 优雅退出处理
def signal_handler(sig, frame):
    print("\n[*] 收到停止信号，正在清理后台进程并保存日志...")
    if pidstat_process:
        try:
            # 杀掉整个进程组，确保 awk 和 pidstat 都退出
            os.killpg(os.getpgid(pidstat_process.pid), signal.SIGTERM)
        except Exception:
            pass
    print(f"[*] eBPF 锁分析日志: {rtnl_log_path}")
    print(f"[*] pidstat 宏观监控日志: {csw_log_path}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

print("[*] 开始全景性能追踪... 按 Ctrl+C 停止。")
while True:
    b.perf_buffer_poll()
