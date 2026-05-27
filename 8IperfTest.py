import json
import subprocess
import time
import argparse
import sys
import os

EXP_DIR = os.environ.get("EXP_LOG_DIR", "./logs")
LOG_DIR = os.path.join(EXP_DIR, "iperfResult") # 你可以自行分类

def parse_args():
    parser = argparse.ArgumentParser(description="并发执行 iperf3 (双端口隔离 TCP+UDP) 与 Ping 测试，生成综合报告")
    parser.add_argument("-f", "--file", type=str, default="iperf_tasks.json", help="任务文件路径")
    parser.add_argument("-t", "--time", type=int, default=100, help="TCP 带宽测试时长(秒) (默认: 100)")
    parser.add_argument("-u_bw", "--udp_bw", type=str, default="10M", help="UDP 测试的探测带宽 (默认: 10M)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        with open(args.file, 'r') as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"[错误] 无法读取任务文件 {args.file}: {e}")
        sys.exit(1)
        
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"总计加载了 {len(tasks)} 个测试对。准备启动双端口服务端...")
    
    server_processes = []
    client_processes = []
    
    # 设定两组不同的基础端口，彻底隔离 TCP 和 UDP
    tcp_base_port = 5201
    udp_base_port = 15201
    
    # 1. 启动所有服务端 (每个节点起两个 iperf3 进程，分别监听不同端口)
    for idx, task in enumerate(tasks):
        tcp_port = tcp_base_port + idx
        udp_port = udp_base_port + idx
        server_node = task['server_node']
        
        cmd_tcp_server = f"docker exec -d {server_node} iperf3 -s -p {tcp_port}"
        cmd_udp_server = f"docker exec -d {server_node} iperf3 -s -p {udp_port}"
        
        try:
            subprocess.run(cmd_tcp_server, shell=True, check=True)
            subprocess.run(cmd_udp_server, shell=True, check=True)
            if server_node not in server_processes:
                server_processes.append(server_node)
        except subprocess.CalledProcessError as e:
            print(f"[警告] 无法在节点 {server_node} 启动 iperf3 服务端: {e}")

    print("所有双端口服务端已启动，等待 2 秒确保服务就绪...")
    time.sleep(2)
    
    # 预计总耗时 = TCP打流时间 + UDP打流(10秒) + Ping(约10秒)
    est_time = args.time + 10 + 10
    print(f"开始并发执行综合测流！TCP 和 UDP 端口已隔离。预计最长耗时 {est_time} 秒，请耐心等待...\n")
    
    # 2. 启动所有客户端执行测试流水线
    for idx, task in enumerate(tasks):
        tcp_port = tcp_base_port + idx
        udp_port = udp_base_port + idx
        client_node = task['client_node']
        server_ip = task['server_ip']
        
        # 报告文件路径
        res_file = f"{LOG_DIR}/{client_node}_to_{task['server_node']}_hop_{task['hop_count']}.txt"
        
        # 流水线：TCP 连 tcp_port，UDP 连 udp_port。互不干扰，完全去掉易错的 sleep
        cmd_pipeline = (
            f"echo '=== 1. TCP Bandwidth (TCP 吞吐量) ===' > {res_file}; "
            f"docker exec {client_node} iperf3 -c {server_ip} -p {tcp_port} -t {args.time} >> {res_file}; "
            f"echo '\n=== 2. UDP Jitter & Packet Loss (UDP 抖动与丢包率) ===' >> {res_file}; "
            f"docker exec {client_node} iperf3 -c {server_ip} -p {udp_port} -u -b {args.udp_bw} -t 10 >> {res_file} 2>&1; "
            f"echo '\n=== 3. ICMP Latency (端到端时延) ===' >> {res_file}; "
            f"docker exec {client_node} ping -c 10 -q {server_ip} >> {res_file}"
        )
        
        # 使用 Popen 并发执行整条流水线
        p = subprocess.Popen(cmd_pipeline, shell=True)
        client_processes.append((p, client_node, task['server_node']))
        
    # 3. 阻塞等待所有客户端的流水线执行完毕
    for p, c_node, s_node in client_processes:
        p.wait()
        
    print("所有测试流水线执行完毕！正在统一清理服务端进程...")
    
    # 4. 清理服务端进程 (统一清理所有的 iperf3)
    for server_node in server_processes:
        subprocess.run(f"docker exec {server_node} pkill iperf3", shell=True, stderr=subprocess.DEVNULL)
        
    # 5. 自动修复结果文件夹权限
    if os.environ.get('SUDO_UID') and os.environ.get('SUDO_GID'):
        uid = int(os.environ.get('SUDO_UID'))
        gid = int(os.environ.get('SUDO_GID'))
        subprocess.run(f"chown -R {uid}:{gid} {LOG_DIR}", shell=True)
        
    print(f"清理完毕。综合测试报告已保存在 {LOG_DIR}/ 目录下。")

if __name__ == "__main__":
    main()