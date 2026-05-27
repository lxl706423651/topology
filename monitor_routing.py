#!/usr/bin/env python3
import time
import threading
from scapy.all import sniff, IP, TCP
import argparse
# from scapy.all import load_contrib
# load_contrib('ospf')
# from scapy.contrib.ospf import OSPF_Hdr #可以看它是 Hello 包还是 LSA Update 包
# 统计计数器
stats = {
    "total_packets": 0,
    "ospf": 0,
    "bgp_total": 0,
    "bgp_ebgp": 0,  # 判定标准: TTL == 1
    "bgp_ibgp": 0,  # 判定标准: TTL > 1
    "others": 0
}

# 线程锁
lock = threading.Lock()

def process_packet(packet):
    """
    处理每一个捕获的包，进行分类统计
    """
    global stats
    
    if not packet.haslayer(IP):
        return

    with lock:
        stats["total_packets"] += 1
        ip_layer = packet[IP]
        
        # 1. 检测 OSPF (Protocol 89)
        if ip_layer.proto == 89:
            stats["ospf"] += 1
            return

        # 2. 检测 BGP (TCP Port 179)
        if packet.haslayer(TCP):
            tcp_layer = packet[TCP]
            # BGP 使用端口 179 (源或目的端口)
            if tcp_layer.sport == 179 or tcp_layer.dport == 179:
                stats["bgp_total"] += 1
                
                # 核心逻辑：区分 iBGP 和 eBGP
                # 默认情况下，eBGP 直连 peers 发送的包 TTL 为 1
                # iBGP 或 eBGP Multihop 的 TTL 通常 > 1 (如 64 或 255)
                # 在大规模仿真中，如果没有特殊的 multihop 配置，TTL=1 是识别 eBGP 最准的特征
                
                if ip_layer.ttl == 1:
                    stats["bgp_ebgp"] += 1
                else:
                    stats["bgp_ibgp"] += 1
                return

        stats["others"] += 1

def monitor_loop(interface):
    print(f"[*] 开始在接口 {interface} 上监听网络控制平面流量...")
    print(f"[*] 过滤规则: OSPF 或 TCP 179 (BGP)")
    
    # 过滤器：只抓取 OSPF 或 BGP 端口的流量，减少性能开销
    # BPF 语法: "proto 89 or (tcp port 179)"
    try:
        sniff(iface=interface, 
              filter="proto 89 or (tcp port 179)", 
              prn=process_packet, 
              store=0) # store=0 表示不保存包到内存，防止内存溢出
    except Exception as e:
        print(f"[!] 抓包出错: {e}")
        exit(1)

def display_stats(interval):
    """
    定期打印统计结果并计算频率
    """
    global stats
    while True:
        time.sleep(interval)
        with lock:
            print("-" * 50)
            print(f"[{time.strftime('%H:%M:%S')}] 过去 {interval} 秒内的发包统计:")
            print(f" > OSPF 包数量:      {stats['ospf']} ({(stats['ospf']/interval):.2f} pkts/sec)")
            print(f" > BGP 总数:         {stats['bgp_total']}")
            print(f"    - eBGP (TTL=1):  {stats['bgp_ebgp']} ({(stats['bgp_ebgp']/interval):.2f} pkts/sec)")
            print(f"    - iBGP (TTL>1):  {stats['bgp_ibgp']} ({(stats['bgp_ibgp']/interval):.2f} pkts/sec)")
            print("-" * 50)
            
            # 重置计数器（如果你想看累计值，就把下面这几行注释掉）
            stats = {k: 0 for k in stats}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docker网络路由协议流量分析器")
    parser.add_argument("-i", "--interface", required=True, help="要监听的网卡接口 (如 eth0, br-xxx, vethxxx)")
    parser.add_argument("-t", "--time", type=int, default=5, help="统计输出间隔时间 (秒)")
    
    args = parser.parse_args()

    # 启动打印线程
    display_thread = threading.Thread(target=display_stats, args=(args.time,), daemon=True)
    display_thread.start()

    # 主线程进行抓包
    monitor_loop(args.interface)