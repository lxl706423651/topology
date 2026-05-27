#!/usr/bin/env python3
# encoding: utf-8

from seedemu import *
import sys, os
from ipaddress import IPv4Address,IPv4Network

assignment = {}
for index in range(1,10000):
    asn = index  # ASN从1开始按顺序分配
    # IPv4前16位为ASN，后16位从1开始递增（每个item唯一）
    # 转换为点分十进制格式（前16位拆分为前两个字节，后16位拆分为后两个字节）
    ip_first_part = 1+(asn //256)  # 前8位
    ip_second_part = asn %256  # 后8位（前16位的低8位）
    ip_third_part = 0  # 后16位的高8位（固定为0，可根据需要调整）
    ip_fourth_part = 0  # 后16位的低8位（从1开始）
    
    ipv4 = f"{ip_first_part}.{ip_second_part}.{ip_third_part}.{ip_fourth_part}/16"
    assignment[asn] = ipv4
def create_full_mesh_backbone(base, ebgp, num_ixs, start_ix_id=100, start_asn=200, assignment=assignment):
    """
    创建一个 IX 全连接的骨干网。
    
    参数:
    - base: Seedemu Base layer 对象
    - ebgp: Seedemu Ebgp layer 对象
    - num_ixs: IX 的数量 (节点数)
    - start_ix_id: IX ID 的起始编号
    - start_asn: Transit AS 的起始 ASN
    
    返回:
    - current_asn: 下一个可用的 ASN (方便后续添加其他 AS)
    """
    
    # 1. 创建所有的 IX (作为拓扑的节点)
    ix_ids = []
    # 用于记录连接到特定 IX 的所有 AS 列表，方便后续建立对等关系
    ix_peers = {} 
    
    print(f"[-] Creating {num_ixs} IXs...")
    for i in range(num_ixs):
        ix_id = start_ix_id + i
        prefix = assignment[ix_id]
        address=str(IPv4Network(prefix)[ix_id])
        ix_obj=base.createInternetExchange(ix_id,prefix,rsAddress=address)
        ix_obj.getPeeringLan().setDisplayName(f'IX-{ix_id}')  # 设置显示名称
        ix_ids.append(ix_id)
        ix_peers[ix_id] = [] # 初始化该 IX 的对等列表

    # 2. 创建 Transit AS (作为拓扑的边)
    # 每一对 IX 之间创建一个 AS
    current_asn = start_asn
    
    # 使用双重循环生成全连接 (Combinations)
    for i in range(len(ix_ids)):
        for j in range(i + 1, len(ix_ids)):
            ix_a = ix_ids[i]
            ix_b = ix_ids[j]
            
            # 创建 AS
            transit_as = base.createAutonomousSystem(current_asn)
            
            # 创建 AS 内部网络 (Seedemu 通常建议 Router 连接一个内部网络)
            #transit_as.createNetwork('net0')
            
            # 创建唯一的 BGP Router

            router= transit_as.createRouter('router_{}_{}'.format(ix_a, ix_b))
            router.joinNetwork('ix{}'.format(ix_a),str(IPv4Network(assignment[ix_a])[current_asn]))
            router.joinNetwork('ix{}'.format(ix_b),str(IPv4Network(assignment[ix_b])[current_asn]))

            #### 一个TransitAS创建两个路由器
            # router1 = transit_as.createRouter('r{}'.format(ix_a))
            # router1.joinNetwork('ix{}'.format(ix_a),str(IPv4Network(assignment[ix_a])[current_asn]))
            # router2 = transit_as.createRouter('r{}'.format(ix_b))
            # router2.joinNetwork('ix{}'.format(ix_b),str(IPv4Network(assignment[ix_b])[current_asn]))
            
            # subnets=list(IPv4Network(assignment[current_asn]).subnets(prefixlen_diff=8))[1]
            # name = 'net_{}_{}'.format(ix_a, ix_b)
            # transit_as.createNetwork(name,str(subnets))
            # router1.joinNetwork(name)
            # router2.joinNetwork(name)

            ####
            # 记录这个 AS 连接到了这两个 IX，以便稍后建立 BGP Session
            ix_peers[ix_a].append(current_asn)
            ix_peers[ix_b].append(current_asn)
            

            current_asn += 1

    print(f"[-] Created {current_asn - start_asn} Transit ASes to fully connect IXs.")

    # 3. 建立 BGP Peering (在 IX 内部全连接)
    # 为了让网络真正通畅，连接到同一个 IX 的所有 Transit AS 之间应该建立 Peer 关系
    # 这里我们假设骨干网内部互联关系为 Peer (对等)
    
    print("[-] Configuring BGP Peering at IXs...")
    for ix_id, asn_list in ix_peers.items():
        # 在该 IX 内，让所有连接进来的 AS 两两建立 Peering
        for i in range(len(asn_list)):
            for j in range(i + 1, len(asn_list)):
                as1 = asn_list[i]
                as2 = asn_list[j]
                
                ebgp.addPrivatePeering(
                    ix_id, 
                    as1, 
                    as2, 
                    abRelationship=PeerRelationship.Peer
                )
                
    return current_asn

def run(dumpfile = None):
    # Set the platform information
    if dumpfile is None:
        script_name = os.path.basename(__file__)
        if len(sys.argv) == 1:
            platform = Platform.AMD64
        elif len(sys.argv) == 2:
            if sys.argv[1].lower() == 'amd':
                platform = Platform.AMD64
            elif sys.argv[1].lower() == 'arm':
                platform = Platform.ARM64
            else:
                print(f"Usage:  {script_name} amd|arm")
                sys.exit(1)
        else:
            print(f"Usage:  {script_name} amd|arm")
            sys.exit(1)

    ###############################################################################
    # 初始化层
    base = Base()
    ebgp = Ebgp()

    ###############################################################################
    # [核心修改] 调用函数生成全连接骨干网
    # 例如：指定 5 个 IX。
    # 这将生成 5 个 IX 和 10 个 Transit AS (C(5,2)=10)。
    # 起始 IX ID 为 100，起始 ASN 为 200
    
    NUM_IX = 70
    next_avail_asn = create_full_mesh_backbone(base, ebgp, NUM_IX, start_ix_id=1, start_asn=200)

    ###############################################################################
    # [可选] 添加一个 Stub AS 来验证网络 (连接到第一个 IX: 100)
    # 这样你可以从这个 host ping 其他网络
    
    # stub_asn = next_avail_asn
    # stub_as = base.createAutonomousSystem(stub_asn)
    # stub_as.createNetwork('net0')
    # stub_as.createRouter('r0').joinNetwork('net0').joinNetwork(100) # 连接到 IX 100
    # stub_as.createHost('host0').joinNetwork('net0')
    
    # # 建立 peering: Stub AS 与连接到 IX 100 的所有 Transit AS 互联
    # # 这里为了简单，我们让它和 IX 100 上所有的 AS Peer，或者你可以手动指定只 Peer 某一个
    # # 这里演示手动 Peer 第一个 Transit AS (ASN 200)
    # ebgp.addPrivatePeering(100, stub_asn, 200, PeerRelationship.Unfiltered)

    ###############################################################################
    # Add all the necessary layers 
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
        emu.compile(docker, './output', override = True)

if __name__ == "__main__":
    run()