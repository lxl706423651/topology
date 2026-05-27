#!/usr/bin/env python3
import subprocess
import concurrent.futures
import re
import os
import datetime
import csv

# ==========================================
# 1. 实验配置
# ==========================================
EXP_DIR = os.environ.get("EXP_LOG_DIR", "./logs")
LOG_DIR = os.path.join(EXP_DIR, "hostMemory") # 你可以自行分类

os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = os.path.join(LOG_DIR, f"memory_snapshot_{timestamp}.csv")

# ==========================================
# 2. 核心功能：读取 Cgroup 获取容器总内存
# ==========================================
def get_container_memory(pid):
    """通过宿主机 PID 绕过 docker stats，直接从内核 Cgroup 读取物理内存"""
    try:
        # 读取进程的 cgroup 挂载路径
        with open(f"/proc/{pid}/cgroup", "r") as f:
            cgroup_paths = f.read().strip().split('\n')
        
        # 判断是 cgroup v1 还是 v2
        is_v2 = any(line.startswith("0::/") for line in cgroup_paths)
        
        if is_v2:
            # Cgroup v2 路径解析
            cgroup_path = [line.split(':')[2] for line in cgroup_paths if line.startswith("0::/")][0]
            mem_file = f"/sys/fs/cgroup{cgroup_path}/memory.current"
        else:
            # Cgroup v1 路径解析 (寻找 memory 子系统)
            cgroup_path = [line.split(':')[2] for line in cgroup_paths if "memory" in line.split(':')[1]][0]
            mem_file = f"/sys/fs/cgroup/memory{cgroup_path}/memory.usage_in_bytes"
            
        with open(mem_file, "r") as f:
            mem_bytes = int(f.read().strip())
            return mem_bytes / (1024 * 1024) # 转换为 MB
    except Exception as e:
        return -1.0

# ==========================================
# 3. 核心功能：查询 BIRD 内部的路由表内存
# ==========================================
def get_bird_routing_memory(container_name):
    """进入容器执行 birdc show memory 并用正则提取 Routing tables 大小"""
    try:
        # 使用 docker exec 非交互式执行 birdc
        cmd = ["docker", "exec", container_name, "birdc", "show", "memory"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        
        # 寻找 "Routing tables:      XXX kB" 这一行
        match = re.search(r"Routing tables:\s+([\d.]+)\s+([A-Za-z]+)", output)
        if match:
            val = float(match.group(1))
            unit = match.group(2).lower()
            # 统一转换为 MB
            if unit == "b": return val / (1024 * 1024)
            elif unit == "kb": return val / 1024
            elif unit == "mb": return val
            elif unit == "gb": return val * 1024
        return -1.0
    except subprocess.CalledProcessError:
        # BIRD 可能未启动或容器内没有 birdc
        return -1.0

# ==========================================
# 4. 组合任务：供线程池调用
# ==========================================
def collect_node_memory(node_info):
    cid, cname, cpid = node_info
    
    # 1. 瞬间获取 Cgroup 系统级内存 (开销极低)
    total_mem_mb = get_container_memory(cpid)
    
    # 2. 获取 BIRD 应用级内存
    bird_mem_mb = get_bird_routing_memory(cname)
    
    return {
        "Container_Name": cname,
        "Total_Mem_MB": round(total_mem_mb, 2),
        "BIRD_Routing_Mem_MB": round(bird_mem_mb, 2),
        "Other_Mem_MB": round(total_mem_mb - bird_mem_mb, 2) if total_mem_mb > 0 and bird_mem_mb > 0 else -1.0
    }

# ==========================================
# 5. 主程序：并发采集
# ==========================================
if __name__ == "__main__":
    print("[*] 正在扫描 Docker 容器...")
    try:
        cids_output = subprocess.check_output(['docker', 'ps', '-q'], text=True).strip()
        if not cids_output:
            print("[!] 未找到运行中的容器。")
            exit(1)
            
        # 批量获取容器名和 PID
        inspect_fmt = '{{.Id}}|{{.Name}}|{{.State.Pid}}'
        cmd = ['docker', 'inspect', '-f', inspect_fmt] + cids_output.split()
        inspect_out = subprocess.check_output(cmd, text=True).strip()
        
        nodes = []
        for line in inspect_out.split('\n'):
            if not line: continue
            cid, cname, cpid = line.split('|')
            cname = cname.lstrip('/')
            if cpid != '0':
                nodes.append((cid, cname, cpid))
                
        print(f"[*] 发现 {len(nodes)} 个节点。开始并发抓取内存快照 (线程数: 50)...")
        
        results = []
        # 使用多线程并发执行，大幅缩短 4000 个节点的时间偏差
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            # 提交所有任务
            future_to_node = {executor.submit(collect_node_memory, node): node for node in nodes}
            
            for future in concurrent.futures.as_completed(future_to_node):
                data = future.result()
                results.append(data)
        
        # 写入 CSV 文件
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Container_Name", "Total_Mem_MB", "BIRD_Routing_Mem_MB", "Other_Mem_MB"])
            writer.writeheader()
            writer.writerows(results)
            
        print(f"\n[*] 采集完成！内存快照已保存至: {csv_file}")
        
    except Exception as e:
        print(f"[!] 脚本运行出错: {e}")
