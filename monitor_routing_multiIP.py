#!/usr/bin/env python3
import time
import threading
from scapy.all import sniff, IP, TCP
import subprocess
import sys
import os
import datetime

# ================= 配置区域 =================

# 1. 目标 IP 列表
TARGET_IPS = [
    "5.248.36.1",
    "5.248.31.1",
    "8.58.3.1",
    "8.58.4.1",
    "1.12.0.1",
    "1.29.0.1"
]

# 2. 统计/写入日志间隔 (秒)
STATS_INTERVAL = 60

# 3. 基础日志目录
BASE_LOG_DIR = "./log"

# ===========================================

def get_interface_by_ip(target_ip):
    """通过 IP 反查网卡接口名"""
    try:
        # 使用 -F 确保精确匹配 IP 字符串
        cmd = f"ip -br -4 addr | grep -F '{target_ip}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        output = result.stdout.strip()
        if output:
            # output 格式: br-xxx UP 1.2.3.4/24 ...
            return output.split()[0]
        else:
            return None
    except Exception:
        return None

class IPMonitorTask:
    """
    单个 IP 的监控任务类。
    负责：独立的计数器、独立的文件句柄、独立的抓包线程。
    """
    def __init__(self, ip, interface):
        self.ip = ip
        self.interface = interface
        self.lock = threading.Lock()
        
        # 独立的计数器
        self.stats = {
            "total": 0, "ospf": 0, "bgp_total": 0, "bgp_ebgp": 0, "bgp_ibgp": 0
        }

        # 1. 创建该 IP 专属的目录
        self.my_log_dir = os.path.join(BASE_LOG_DIR, self.ip)
        if not os.path.exists(self.my_log_dir):
            os.makedirs(self.my_log_dir)

        # 2. 创建日志文件 (带时间戳)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.my_log_dir, f"stats_{timestamp}.log")
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        
        # 写入文件头
        self.write_to_file(f"=== 监控启动: {ip} ({interface}) ===")
        print(f"[*] 已初始化: IP {ip} -> 目录 {self.my_log_dir}")

    def write_to_file(self, msg):
        """写入日志并刷新缓冲区"""
        now = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        self.log_file.write(f"{now} {msg}\n")
        self.log_file.flush()

    def process_packet(self, packet):
        """回调函数：只处理本接口的包"""
        if not packet.haslayer(IP):
            return

        with self.lock:
            self.stats["total"] += 1
            ip_layer = packet[IP]

            # OSPF
            if ip_layer.proto == 89:
                self.stats["ospf"] += 1
                return

            # BGP
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                if tcp_layer.sport == 179 or tcp_layer.dport == 179:
                    self.stats["bgp_total"] += 1
                    if ip_layer.ttl == 1:
                        self.stats["bgp_ebgp"] += 1
                    else:
                        self.stats["bgp_ibgp"] += 1

    def start(self):
        """启动该 IP 的抓包线程"""
        t = threading.Thread(target=self._sniff_loop, daemon=True)
        t.start()

    def _sniff_loop(self):
        try:
            # prn 是每收到一个包就调用的回调函数
            sniff(iface=self.interface, 
                  filter="proto 89 or (tcp port 179)", 
                  prn=self.process_packet, 
                  store=0)
        except Exception as e:
            self.write_to_file(f"ERROR: 抓包线程崩溃 - {e}")

    def report_and_reset(self):
        """
        被主定时器调用：
        1. 读取当前统计
        2. 写入日志
        3. 重置计数器
        """
        with self.lock:
            # 计算速率
            s = self.stats
            # 格式化输出
            lines = [
                f"--- 周期统计 ({STATS_INTERVAL}s) ---",
                f"Total: {s['total']} | OSPF: {s['ospf']} | BGP Total: {s['bgp_total']}",
                f"Details -> eBGP(TTL=1): {s['bgp_ebgp']} | iBGP(TTL>1): {s['bgp_ibgp']}"
            ]
            
            # 写入文件
            for line in lines:
                self.write_to_file(line)
            
            # 重置
            self.stats = {k: 0 for k in self.stats}

def main():
    print(f"=== 多路网络监控启动 (日志分离模式) ===")
    
    # 1. 扫描并创建任务列表
    tasks = []
    
    for ip in TARGET_IPS:
        iface = get_interface_by_ip(ip)
        if iface:
            # 创建监控对象
            task = IPMonitorTask(ip, iface)
            tasks.append(task)
            # 启动抓包线程
            task.start()
        else:
            print(f"[!] 警告: 无法找到 IP {ip} 对应的接口，跳过。")

    if not tasks:
        print("没有有效的监控任务，退出。")
        sys.exit(1)

    print(f"[*] 成功启动 {len(tasks)} 个监控线程。正在后台运行...")
    print(f"[*] 日志将每 {STATS_INTERVAL} 秒写入 ./log/<IP>/ 目录中。")
    print(f"[*] 按 Ctrl+C 停止。")

    # 2. 主循环：负责定期让每个任务写日志
    try:
        while True:
            time.sleep(STATS_INTERVAL)
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 正在写入所有日志...")
            
            for task in tasks:
                task.report_and_reset()
                
    except KeyboardInterrupt:
        print("\n[*] 停止监控。")
        for task in tasks:
            task.log_file.close()
        sys.exit(0)

if __name__ == "__main__":
    # 确保有 root 权限，否则无法抓包
    if os.geteuid() != 0:
        print("请使用 sudo 运行此脚本！")
        sys.exit(1)
    main()