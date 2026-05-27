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

# 生成带时间戳的日志文件名
start_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(LOG_DIR, f"rtnl_trace_{start_time_str}.log")

# ==========================================
# 2. 构建 Docker 容器映射字典 (Inode -> Name)
# ==========================================
def build_container_map():
    inode_to_name = {}
    print("正在扫描当前运行的 Docker 容器以构建 Netns 映射...")
    try:
        # 获取所有运行中的容器 ID
        cids_output = subprocess.check_output(['docker', 'ps', '-q']).decode('utf-8').strip()
        if not cids_output:
            print("警告: 当前没有运行中的 Docker 容器。")
            return inode_to_name

        cids = cids_output.split()
        
        # 批量获取容器名和宿主机 PID
        inspect_fmt = '{{.Name}}|{{.State.Pid}}'
        cmd = ['docker', 'inspect', '-f', inspect_fmt] + cids
        inspect_out = subprocess.check_output(cmd).decode('utf-8').strip()

        for line in inspect_out.split('\n'):
            if not line: continue
            name, pid = line.split('|')
            name = name.lstrip('/') # docker inspect 返回的 name 带有前导斜杠
            
            if pid != '0':
                try:
                    # 获取该 PID 对应的 Netns Inode
                    netns_path = f"/proc/{pid}/ns/net"
                    inode = os.stat(netns_path).st_ino
                    inode_to_name[inode] = name
                except FileNotFoundError:
                    # 进程可能瞬间退出了
                    pass
    except Exception as e:
        print(f"构建容器映射时出错: {e}")
        
    print(f"成功映射了 {len(inode_to_name)} 个容器的网络命名空间。")
    return inode_to_name

container_map = build_container_map()

# ==========================================
# 3. 定义 BPF C 语言代码
# ==========================================
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <linux/sched.h>
#include <linux/nsproxy.h>
#include <net/net_namespace.h>

struct data_t {
    u32 pid;
    u32 netns;
    u16 type;
    char comm[TASK_COMM_LEN];
};

BPF_PERF_OUTPUT(events);

int trace_rtnetlink_rcv(struct pt_regs *ctx, struct sk_buff *skb, struct nlmsghdr *nlh) {
    struct data_t data = {};
    
    // 获取进程 PID 和进程名
    data.pid = bpf_get_current_pid_tgid() >> 32;
    bpf_get_current_comm(&data.comm, sizeof(data.comm));
    
    // 读取 Netlink 消息类型
    bpf_probe_read_kernel(&data.type, sizeof(data.type), &nlh->nlmsg_type);
    
    // 仅过滤路由相关的操作 (24: 新增/更新, 25: 删除, 26: 查询/Dump)
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
    data.netns = inum;

    events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""

# ==========================================
# 4. 编译、注入 BPF 代码并处理输出
# ==========================================
b = BPF(text=bpf_text)
b.attach_kprobe(event="rtnetlink_rcv_msg", fn_name="trace_rtnetlink_rcv")

# 操作类型映射字典
op_map = {
    24: "ADD/UPDATE (24)",
    25: "DELETE (25)",
    26: "READ/DUMP (26)"
}

print(f"正在追踪路由 Netlink 操作... 日志将实时写入: {log_file_path}")
print("按 Ctrl+C 停止。\n")

header = f"{'TIMESTAMP':<24} {'COMM':<16} {'CONTAINER (INODE)':<35} {'OPERATION':<15}"
print(header)

# 写入日志文件表头
with open(log_file_path, "a") as f:
    f.write(header + "\n")

def print_event(cpu, data, size):
    event = b["events"].event(data)
    
    # 1. 获取当前高精度时间戳
    ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # 2. 解析操作类型
    op_str = op_map.get(event.type, f"UNKNOWN ({event.type})")
    
    # 3. 映射容器名 (如果找不到，说明是宿主机发起的请求，或者容器是脚本启动后新创建的)
    c_name = container_map.get(event.netns, "Host/Unknown")
    container_info = f"{c_name} ({event.netns})"
    
    # 4. 格式化输出字符串
    log_line = f"{ts_str:<24} {event.comm.decode('utf-8'):<16} {container_info:<35} {op_str:<15}"
    
    # 打印到控制台
    print(log_line)
    
    # 追加写入到文件
    with open(log_file_path, "a") as f:
        f.write(log_line + "\n")

b["events"].open_perf_buffer(print_event)

while True:
    try:
        b.perf_buffer_poll()
    except KeyboardInterrupt:
        print(f"\n追踪已停止。日志已保存在: {log_file_path}")
        exit()
