#!/usr/bin/env python3
# encoding: utf-8
import sys
from seedemu import *
import sys, os
import random
from ipaddress import IPv4Address, IPv4Network


transit_asn=151
# ==========================================
# 1. IP 地址分配逻辑 (全局)
# ==========================================
assignment = {}
for index in range(1, 10000):
    asn = index  # ASN从1开始按顺序分配
    # IPv4前16位为ASN，后16位从1开始递增
    ip_first_part = 1 + (asn // 256)
    ip_second_part = asn % 256
    ip_third_part = 0
    ip_fourth_part = 0
    
    ipv4 = f"{ip_first_part}.{ip_second_part}.{ip_third_part}.{ip_fourth_part}/16"
    assignment[asn] = ipv4

# ==========================================
# 2. 辅助函数：计算图的直径
# ==========================================
def get_graph_diameter(nodes, adj_list):
    """
    使用 BFS 计算无向图的直径（任意两点间最短路径的最大值）。
    """
    if not adj_list:
        return 0
    
    max_distance = 0
    
    for start_node in range(nodes):
        # BFS 初始化
        visited = {start_node}
        queue = [(start_node, 0)]
        max_dist_from_start = 0
        
        # 简单队列实现 BFS
        head = 0
        while head < len(queue):
            current, dist = queue[head]
            head += 1
            max_dist_from_start = max(max_dist_from_start, dist)
            
            for neighbor in adj_list.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        # 检查连通性 (如果遍历不到所有节点，直径为无穷大)
        if len(visited) != nodes:
            return float('inf')
            
        max_distance = max(max_distance, max_dist_from_start)
        
    return max_distance

# ==========================================
# 3. 核心功能：创建受直径约束的 Stub AS
# ==========================================
def create_constrained_stub_as(base, ebgp, asn, ix_to_connect, num_routers, max_hops_h, assignment):
    """
    创建一个 Stub AS，内部包含 num_routers 个路由器。
    拓扑约束：任意两个路由器之间的跳数 <= max_hops_h。
    优化策略：使用多级步长和弦环 (Hierarchical Chordal Ring) 确定性快速降低直径。
    """
    print(f"[-] Creating Stub AS {asn}: {num_routers} routers, Max Diameter <= {max_hops_h}")
    
    my_as = base.createAutonomousSystem(asn)
    
    # --- 步骤 A: 生成满足直径约束的图拓扑 (Adjacency List) ---
    adj = {i: [] for i in range(num_routers)}
    edges = set()

    # A1. 初始构建：环形连接 (0-1-2-...-(N-1)-0)
    for i in range(num_routers):
        u = i
        v = (i + 1) % num_routers
        edge = tuple(sorted((u, v)))
        if edge not in edges:
            adj[u].append(v)
            adj[v].append(u)
            edges.add(edge)

    # [优化修改] A2. 迭代优化：确定性多级步长策略
    # 生成步长列表：例如 N=100 -> [50, 25, 12, 6, 3, 2]
    # 优先连最远的(N/2)，然后连次远的(N/4)，以此类推，直到满足直径要求
    strides = []
    curr_step = num_routers // 2
    while curr_step > 1:
        strides.append(curr_step)
        curr_step //= 2
    
    stride_idx = 0      # 当前使用的是第几个步长
    node_idx = 0        # 当前遍历到的节点索引
    iteration = 0

    while True:
        # 1. 计算当前直径 (这一步比较耗时，但为了严格满足约束必须检查)
        # 如果追求极致生成速度，可以每添加 N 个点查一次，但为了最少边数，这里每次查
        d = get_graph_diameter(num_routers, adj)
        if d <= max_hops_h:
            break
        
        # 2. 检查是否还有可用的确定性步长
        if stride_idx >= len(strides):
            # 极少见情况：所有对半步长都加完了直径还不达标（通常发生在 h 要求极小如 h=2 时）
            # 此时回退到步长为 2 的密集连接，或者直接全连接（但在 seedemu 中尽量避免）
            print(f"Warning: AS {asn} used all heuristic strides. Diameter is {d}, target {max_hops_h}.")
            break

        # 3. 按当前步长添加一条边
        current_stride = strides[stride_idx]
        added_valid_edge = False
        
        # 尝试寻找一条未存在的边
        # 我们使用 while 循环来跳过那些已经存在的边（例如偶数N时的对称边）
        while node_idx < num_routers:
            u = node_idx
            v = (u + current_stride) % num_routers
            
            node_idx += 1 # 准备下一次连接下一个点
            
            edge = tuple(sorted((u, v)))
            if u != v and edge not in edges:
                adj[u].append(v)
                adj[v].append(u)
                edges.add(edge)
                added_valid_edge = True
                iteration += 1
                break # 添加了一条边，跳出内层循环去检查直径
        
        # 4. 状态维护
        # 如果当前步长把所有节点都遍历了一遍 (node_idx 跑完了一圈)
        if node_idx >= num_routers:
            stride_idx += 1 # 切换到下一个更小的步长 (例如从 N/2 切换到 N/4)
            node_idx = 0    # 重置节点指针

        # 如果这一轮循环没有添加任何边（说明当前步长的所有边都已存在），且没跑完所有步长，继续循环
        if not added_valid_edge and stride_idx < len(strides):
            continue
        elif not added_valid_edge:
            break

    

    print(f"    Topology generated: {len(edges)} links (Ring + {iteration} extra) to satisfy diameter <= {max_hops_h}.")

    # --- 步骤 B & C 保持不变 ---
    # ... (原有代码 B1, B2, C 部分) ...
    
    # B1. 创建路由器
    router_objs = []
    for i in range(num_routers):
        r = my_as.createRouter(f'r{i}')
        router_objs.append(r)

    # B2. 创建链路
    as_prefix = IPv4Network(assignment[asn])
    subnets_gen = as_prefix.subnets(new_prefix=28) 
    next(subnets_gen) 
    
    for u, v in edges:
        net_name = f'link_{u}_{v}'
        try:
            subnet = next(subnets_gen)
            network = my_as.createNetwork(net_name, str(subnet))
            router_objs[u].joinNetwork(net_name, str(subnet[-1]))
            router_objs[v].joinNetwork(net_name, str(subnet[-2]))
        except StopIteration:
            print("Error: Not enough IP subnets!")
            break

    # C. 外部连接
    gateway_router = router_objs[0]
    ix_ip = str(IPv4Network(assignment[ix_to_connect])[asn])
    gateway_router.joinNetwork(f'ix{ix_to_connect}', ix_ip)
    
    ebgp.addPrivatePeering(ix_to_connect, transit_asn, asn, PeerRelationship.Peer)
    
    return my_as

# ==========================================
# 4. 主程序
# ==========================================
def run(dumpfile = None):
    # Set the platform information
    if dumpfile is None:
        script_name = os.path.basename(__file__)
        platform = Platform.AMD64
        if len(sys.argv) > 1:
            # 注意：命令行参数默认都是 String 类型，如果是数字需要转换
            x = int(sys.argv[1]) 
            print(f"接收到的参数 x 是: {x}")
        else:
            print("请提供参数 x")
            x = 1 # 设置默认值

    base = Base()
    ebgp = Ebgp()

    # 1. 创建单个 IX
    # ==========================================
    IX_ID = 100
    prefix = assignment[IX_ID]
    address = str(IPv4Network(prefix)[IX_ID])
    
    print(f"[-] Creating IX-{IX_ID}...")
    ix_obj = base.createInternetExchange(IX_ID, prefix, rsAddress=address)
    ix_obj.getPeeringLan().setDisplayName(f'IX-{IX_ID}')

    transit_asn=151
    transit_as = base.createAutonomousSystem(transit_asn)
    router= transit_as.createRouter('router_{}'.format(IX_ID))
    router.joinNetwork('ix{}'.format(IX_ID), str(IPv4Network(assignment[IX_ID])[transit_asn]))

    # 2. 创建边缘网络 (Edge Network)
    # 需求：1个 IX，1个 Stub AS，接入该 Stub AS 的路由器有 N 个，直径 <= h
    # ==========================================
    NUM_ROUTERS_IN_STUB = 300  # 路由器数量 N
    MAX_HOPS_H = 20            # 最大跳数 h (直径约束)

    for i in range(200,200+x):
        create_constrained_stub_as(
            base, 
            ebgp, 
            i, 
            IX_ID, 
            NUM_ROUTERS_IN_STUB, 
            MAX_HOPS_H, 
            assignment
        )
    
    

    # Finalize
    emu = Emulator()
    emu.addLayer(base)
    emu.addLayer(Routing())  
    emu.addLayer(ebgp)
    emu.addLayer(Ibgp()) 
    emu.addLayer(Ospf())

    if dumpfile is not None:
        emu.dump(dumpfile)
    else:
        emu.render()
        docker = Docker(internetMapEnabled=True, platform=platform)
        emu.compile(docker, './sbuboutput', override = True)

if __name__ == "__main__":
    run()