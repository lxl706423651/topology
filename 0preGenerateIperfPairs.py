import json
import random
import re
import argparse
import sys
import subprocess
import os

def parse_args():
    parser = argparse.ArgumentParser(description="生成跨 AS 与 AS 内部的 iperf3 测试对 (包含连通性校验)")
    parser.add_argument("-n", "--num", type=int, default=16, help="需要生成的总对数 (默认: 16)")
    parser.add_argument("-f", "--file", type=str, default="container_pids.json", help="JSON 文件的路径")
    parser.add_argument("-s", "--seed", type=int, default=42, help="随机种子，保证每次生成结果一致 (默认: 42)")
    parser.add_argument("-o", "--out", type=str, default="iperf_tasks.json", help="输出的中间任务文件")
    return parser.parse_args()

def load_nodes(filepath):
    nodes = []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            for container_name, pid in data.items():
                parts = container_name.split('-')
                if len(parts) >= 3:
                    as_info = parts[0]
                    node_id = parts[1]
                    ip_addr = '-'.join(parts[2:])
                    nodes.append({
                        'container': container_name,
                        'as': as_info,
                        'node': node_id,
                        'ip': ip_addr,
                        'pid': pid
                    })
        return nodes
    except Exception as e:
        print(f"[错误] 解析 JSON 失败: {e}")
        sys.exit(1)

def run_ping(source_node, target_ip):
    pid = source_node['pid']
    cmd = f"nsenter -t {pid} -n ping -c 1 -W 1 {target_ip}"
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True)
        match = re.search(r'ttl=(\d+)', output)
        if match:
            ttl = int(match.group(1))
            initial_ttl = 64 if ttl <= 64 else (128 if ttl <= 128 else 255)
            return initial_ttl - ttl
    except:
        pass
    return -1

def is_same_as(node_a, node_b):
    return node_a['as'] == node_b['as']

def main():
    args = parse_args()
    total_needed = args.num
    
    # 固定随机种子
    random.seed(args.seed)
    
    nodes = load_nodes(args.file)
    if len(nodes) < 2:
        print("[错误] 节点数量不足。")
        sys.exit(1)

    print(f"当前随机种子: {args.seed} | 开始寻找目标核心对...")
    
    target_hops = [1, 3, 5, 10]
    found_core_pairs = {hop: {'same': [], 'diff': []} for hop in target_hops}
    
    core_completed = False
    attempts = 0
    max_attempts = 10000 
    
    all_selected_pairs = set()
    output_tasks = [] 

    # ================= 1. 寻找核心测试对 =================
    while not core_completed and attempts < max_attempts:
        attempts += 1
        node_a, node_b = random.sample(nodes, 2)
        as_type = 'same' if is_same_as(node_a, node_b) else 'diff'
        
        needs_this_type = any(len(found_core_pairs[hop][as_type]) < 2 for hop in target_hops)
        if not needs_this_type:
            continue
            
        hops = run_ping(node_a, node_b['ip'])
        
        if hops in target_hops and len(found_core_pairs[hops][as_type]) < 2:
            pair_tuple = tuple(sorted([node_a['container'], node_b['container']]))
            if pair_tuple not in all_selected_pairs:
                found_core_pairs[hops][as_type].append((node_a, node_b))
                all_selected_pairs.add(pair_tuple)
                sys.stdout.write(f"\r[+] 找到核心对: {hops}跳 | {'AS内部' if as_type == 'same' else '跨AS'} | {node_a['container']} -> {node_b['container']}\n")
                
        core_completed = all(len(found_core_pairs[h]['same']) == 2 and len(found_core_pairs[h]['diff']) == 2 for h in target_hops)
        
        if attempts % 500 == 0:
            sys.stdout.write(f"\r正在搜索中... 已尝试 {attempts} 次组合")
            sys.stdout.flush()

    print("\n")
    if not core_completed:
        print("[警告] 达到最大尝试次数，未能找齐所有指定跳数的核心对。")

    for hop in target_hops:
        for as_type in ['same', 'diff']:
            for a, b in found_core_pairs[hop][as_type]:
                output_tasks.append({
                    "type": "core",
                    "hop_count": hop,
                    "as_status": "Intra-AS" if as_type == 'same' else "Inter-AS",
                    "client_node": a['container'],
                    "server_node": b['container'],
                    "server_ip": b['ip']
                })

    # ================= 2. 补充并验证连通性的噪声对 =================
    noise_needed = total_needed - len(output_tasks)
    noise_attempts = 0
    max_noise_attempts = 20000 # 给予噪声组足够的随机测试冗余
    
    if noise_needed > 0:
        print(f"开始寻找并验证剩余的 {noise_needed} 个连通噪声对...")
        
    while noise_needed > 0 and noise_attempts < max_noise_attempts:
        noise_attempts += 1
        node_a, node_b = random.sample(nodes, 2)
        
        # 规则 1: 排除 ix 节点
        if 'ix' in node_a['node'] or 'ix' in node_b['node']:
            continue
            
        pair_tuple = tuple(sorted([node_a['container'], node_b['container']]))
        if pair_tuple not in all_selected_pairs:
            # 规则 2: 必须能 ping 通 (获取真实跳数)
            hops = run_ping(node_a, node_b['ip'])
            
            if hops != -1: # -1 代表超时不可达
                all_selected_pairs.add(pair_tuple)
                as_type = 'same' if is_same_as(node_a, node_b) else 'diff'
                
                output_tasks.append({
                    "type": "noise",
                    "hop_count": hops, # 记录实际测试出的跳数
                    "as_status": "Intra-AS" if as_type == 'same' else "Inter-AS",
                    "client_node": node_a['container'],
                    "server_node": node_b['container'],
                    "server_ip": node_b['ip']
                })
                
                sys.stdout.write(f"\r[+] 验证并添加噪声对: {node_a['container']} -> {node_b['container']} (还需 {noise_needed - 1} 对)        ")
                sys.stdout.flush()
                noise_needed -= 1

    print("\n")
    if noise_needed > 0:
        print(f"[警告] 经过大量尝试，仍差 {noise_needed} 对连通的噪声组合未能找到。拓扑规模可能受限。")

    # ================= 3. 保存结果 =================
    with open(args.out, 'w') as f:
        json.dump(output_tasks, f, indent=4)

    if os.environ.get('SUDO_UID') and os.environ.get('SUDO_GID'):
        uid = int(os.environ.get('SUDO_UID'))
        gid = int(os.environ.get('SUDO_GID'))
        os.chown(args.out, uid, gid)
    print(f"成功生成 {len(output_tasks)} 对有效连通测试任务，已保存至 {args.out}")

if __name__ == "__main__":
    main()