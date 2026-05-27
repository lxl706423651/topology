#!/usr/bin/env python3
from bcc import BPF
import os
import time
import datetime
import subprocess

# ==========================================
# 1. 日志目录与文件初始化
# ==========================================
LOG_DIR = "./logs/rtnl/"
os.makedirs(LOG_DIR, exist_ok=True)

start_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOG_DIR, f"rtnl_trace_{start_time_str}.log")

# ==========================================
# 2. 构建 Docker 容器映射字典 (Inode -> Name)
# ==========================================
def build_container_map():
    inode_to_name = {}
    print("正在扫描当前运行的 Docker 容器以构建 Netns 映射...")
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
        print(f"构建容器映射时出错: {e}")
        
    print(f"成功映射了 {len(inode_to_name)} 个容器的网络命名空间。")
    return inode_to_name

container_map = build_container_map()

# ==========================================
# 3. 定义 BPF C 语言代码 (新增内核级限流)
# ==========================================
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <linux/sched.h>
#include <linux/nsproxy.h>
#include <net/net_namespace.h>

// 定义限流时间窗口：500,000,000 纳秒 = 0.5 秒
// 意味着同一个容器在 0.5 秒内的连续相同操作，只会输出一次
#define THROTTLE_NS 500000000ULL

// 用于哈希表的 Key (联合标识一个独一无二的动作)
struct hash_key_t {
    u32 pid;
    u32 netns;
    u16 type;
};

// 用于发送给用户态的数据
struct data_t {
    u32 pid;
    u32 netns;
    u16 type;
    char comm[TASK_COMM_LEN];
};

// 记录最后一次发生的时间戳
BPF_HASH(last_seen, struct hash_key_t, u64);
BPF_PERF_OUTPUT(events);

int trace_rtnetlink_rcv(struct pt_regs *ctx, struct sk_buff *skb, struct nlmsghdr *nlh) {
    struct data_t data = {};
    struct hash_key_t key = {};
    
    u64 pid_tgid = bpf_get_current_pid_tgid();
    key.pid = pid_tgid >> 32;
    data.pid = key.pid;
    
    // 读取 Netlink 消息类型
    bpf_probe_read_kernel(&key.type, sizeof(key.type), &nlh->nlmsg_type);
    data.type = key.type;
    
    // 仅过滤路由相关的操作
    if (data.type != RTM_NEWROUTE && data.type != RTM_DELROUTE && data.type != RTM_GETROUTE) {
        return 0;
    }

    // 获取当前进程的网络命名空间 Inode
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
    key.netns = inum;
    data.netns = inum;

    // ==========================================
    // 核心改进：内核级限流去重逻辑
    // ==========================================
    u64 *last_ts = last_seen.lookup(&key);
    u64 now = bpf_ktime_get_ns();
    
    // 如果找到了记录，并且当前时间距离上次记录小于 THROTTLE_NS (0.5秒)
    if (last_ts && (now - *last_ts < THROTTLE_NS)) {
        return 0; // 直接返回，不在内核侧产生任何事件，彻底释放 I/O 和 CPU 压力
    }
    
    // 更新时间戳并放行
    last_seen.update(&key, &now);

    // 获取进程名并提交事件给 Python
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""

# ==========================================
# 4. 编译、注入 BPF 代码并处理输出
# ==========================================
b = BPF(text=bpf_text)
b.attach_kprobe(event="rtnetlink_rcv_msg", fn_name="trace_rtnetlink_rcv")

op_map = {
    24: "ADD/UPDATE (24)",
    25: "DELETE (25)",
    26: "READ/DUMP (26)"
}

print(f"正在追踪路由 Netlink 操作 (已开启内核级 0.5 秒防抖)...")
print(f"日志将实时写入: {log_file_path}")
print("按 Ctrl+C 停止。\n")

header = f"{'TIMESTAMP':<24} {'PID':<8} {'COMM':<16} {'CONTAINER (INODE)':<35} {'OPERATION':<15}"
print(header)

with open(log_file_path, "a") as f:
    f.write(header + "\n")

def print_event(cpu, data, size):
    event = b["events"].event(data)
    ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    op_str = op_map.get(event.type, f"UNKNOWN ({event.type})")
    
    c_name = container_map.get(event.netns, "Host/Unknown")
    container_info = f"{c_name} ({event.netns})"
    
    log_line = f"{ts_str:<24} {event.pid:<8} {event.comm.decode('utf-8'):<16} {container_info:<35} {op_str:<15}"
    
    print(log_line)
    with open(log_file_path, "a") as f:
        f.write(log_line + "\n")

b["events"].open_perf_buffer(print_event)

while True:
    try:
        b.perf_buffer_poll()
    except KeyboardInterrupt:
        print(f"\n追踪已停止。日志已保存在: {log_file_path}")
        exit()
