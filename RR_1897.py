from seedemu.layers import Base, Routing, Ebgp, Ibgp, Ospf, PeerRelationship
from seedemu.compiler import Docker, Platform
from seedemu.core import Emulator
import os
import sys
import networkx as nx
import re
import pickle
from typing import List, Tuple, Dict
from seedemu.layers import Base, Ebgp, Routing, Ibgp, Ospf
from seedemu.layers.Ebgp import PeerRelationship
from seedemu.core import Binding, Filter, Emulator, Service, Router, AutonomousSystem
from typing import List, Tuple, Dict
import time
import threading
import psutil
from datetime import datetime
from ipaddress import IPv4Address,IPv4Network
import random
from seedemu.services import TrafficService, TrafficServiceType
from seedemu.layers import EtcHosts
with open("assignment.pkl", "rb") as f:
    assignment = pickle.load(f)
node_prefix={}
for key,value in assignment.items():
    node_prefix[value['asn']]=value['ipv4']

def get_assignment(asn,assignment=assignment):
    t=len(assignment)
    if asn not in assignment.keys():
        assignment[asn]={'asn':t+1,'ipv4':f'{(t+1) //256}.{(t+1) %256}.0.0/16'}
    return assignment

def generate_connected_pairs(nodes, extra_edges_count=0):
    """
    生成随机连接对，确保图是连通的。
    :param nodes: 节点列表
    :param extra_edges_count: 在保证连通后，额外增加多少条随机边
    """
    if len(nodes) < 2:
        return []
    elif len(nodes) == 2:
        return [tuple(sorted((nodes[0], nodes[1])))]
    
    pairs = set() # 使用集合避免重复边
    for i in range(len(nodes)):
        pair1 = tuple(sorted((nodes[i],nodes[(i + 1) % len(nodes)])))
        pairs.add(pair1)
    for i in range(len(nodes)):
        if len(pairs)<254:
            break;
        if i!= (int(i + 1+len(nodes)/2) % len(nodes)):
            pair2 = tuple(sorted((nodes[i],nodes[int(i + 1+len(nodes)/2) % len(nodes)])))
            pairs.add(pair2)

    return list(pairs)

def makeStubAsWithHosts(emu: Emulator, base: Base, asn: int, prefix: str, exchange: int, hosts_total: int):

    # Create AS and internal network
    network = "net0"

    stub_as = base.createAutonomousSystem(asn)
    stub_as.createNetwork(network,prefix)

    # Create a BGP router
    # Attach the router to both the internal and external networks
    router = stub_as.createRouter('r{}'.format(exchange))
    router.joinNetwork(network)
    router.joinNetwork('ix{}'.format(exchange),str(IPv4Network(node_prefix[exchange])[asn]))

    for counter in range(hosts_total):
       name = 'host_{}'.format(counter)
       host = stub_as.createHost(name)
       host.joinNetwork(network)

def makeTransitAs(base: Base, asn: int, prefix: str, exchanges: List[int],
    intra_ix_links: List[Tuple[int, int]],node_prefix=node_prefix,rrNum=0) -> AutonomousSystem:
    """!
    @brief create a transit AS.

    @param base reference to the base layer.
    @param asn ASN of the newly created AS.
    @param exchanges list of IXP IDs to join.
    @param intra_ix_links list of tuple of IXP IDs, to create intra-IX links at.

    @returns transit AS object.
    """

    transit_as = base.createAutonomousSystem(asn)
    #transit_as.setSubnets(prefix)
    routers: Dict[int, Router] = {}

    # Create a BGP router for each internet exchange (for peering purpose)
    for ix in exchanges:
        routers[ix] = transit_as.createRouter('r{}'.format(ix))
        routers[ix].addSoftware("iperf3")  #####################################################
        routers[ix].joinNetwork('ix{}'.format(ix),str(IPv4Network(node_prefix[ix])[asn]))
        #print(ix,str(IPv4Network(node_prefix[ix])[asn]))
        #raise ValueError
    if(rrNum==1):
        rr_index= list(routers.keys())[0]
        routers[rr_index].makeRouteReflector(True)
    elif(rrNum>1):
        for i in range(rrNum):
            transit_as.createCluster(f'10.0.0.{i+1}')
        index=0
        for key,value in routers.items():
            if index<rrNum:
                value.makeRouteReflector(True)
                value.joinBgpCluster(f'10.0.0.{index+1}')
                index+=1
            else:
                value.joinBgpCluster(f'10.0.0.{index%rrNum+1}')
                index+=1
    # For each pair, create an internal network to connect the BGP routers
    # from two internet exchanges. There is no need to create a full-mesh
    # network among the BGP routers. As long as they can reach each other
    # over a single or multiple hops, it is OK.
    t=len(intra_ix_links)
    i=1
    subnets=list(IPv4Network(prefix).subnets(prefixlen_diff=8))
    for (a, b) in intra_ix_links:
        assert i<255, "net >= 255"
        name = 'net_{}_{}'.format(a, b)
        transit_as.createNetwork(name,str(subnets[i]))
        routers[a].joinNetwork(name)#,str(subnets[i][253])
        routers[b].joinNetwork(name)#,str(subnets[i][254])
        i+=1

    return transit_as


def load_topology_data(filename: str) -> dict:
    """从文件加载拓扑数据（兼容 key: value 格式）"""
    with open(filename, 'r') as f:
        content = f.read().strip()
    
    # 处理格式：添加外层大括号，替换冒号为冒号+引号，处理列表
    # 1. 替换 key: 为 'key':
    content = re.sub(r'^(\w+):', r'"\1":', content, flags=re.MULTILINE)
    # 2. 每行末尾添加逗号（最后一行除外）
    lines = content.split('\n')
    lines = [line + ',' for line in lines[:-1]] + [lines[-1]] if lines else []
    content = '\n'.join(lines)
    # 3. 包裹成字典
    content = '{' + content + '}'
    
    # 安全解析为字典
    try:
        return eval(content)  # 此处使用eval是因为处理后的格式已符合Python字典规范
    except Exception as e:
        raise ValueError(f"解析拓扑数据失败: {e}")

def run(dumpfile=None, hosts_per_as=2): 
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
            x = 214 # 设置默认值

    # 加载拓扑数据
    try:
        TOPOLOGY_DATA = load_topology_data(f'real_topology_{x}.txt')
        #TOPOLOGY_DATA = load_topology_data('real_topology_1897.txt')
    except FileNotFoundError:
        print(f"错误: 未找到real_topology_{x}.txt文件")
        sys.exit(1)
    except Exception as e:
        print(f"解析拓扑数据出错: {e}")
        sys.exit(1)

    
    emu   = Emulator()
    ebgp  = Ebgp()
    base  = Base()
    
    ###############################################################################
    # 创建互联网交换点(IXP)
    ix_objects = {}
    for ixp in TOPOLOGY_DATA["ixps"]:
        prefix = assignment[ixp]['ipv4']
        ix = assignment[ixp]['asn']
        address=str(IPv4Network(prefix)[ix])
        ix_obj = base.createInternetExchange(ix,prefix,rsAddress=address)
        ix_obj.getPeeringLan().setDisplayName(f'IX-{ix}')  # 设置显示名称
        ix_objects[ix] = ix_obj
        print(f"创建IXP: {ix} (显示名称: IX-{ix})")
    
    ###############################################################################
    # 收集Transit AS的连接信息
    transit_info = {}
    for asn in TOPOLOGY_DATA["transit_asns"]:
        asn = assignment[asn]['asn']
        connected_ixs = set()
        intra_links = []
        for (ix_a, ix_b, t_asn, _) in TOPOLOGY_DATA["ix_ix_transit_edges"]:
            ix_a = assignment[ix_a]['asn']
            ix_b = assignment[ix_b]['asn']
            t_asn = assignment[t_asn]['asn']
            if t_asn == asn:
                connected_ixs.add(ix_a)
                connected_ixs.add(ix_b)
                intra_links.append((ix_a, ix_b))
        transit_info[asn] = (sorted(connected_ixs), intra_links)
    # with open("transit_info.pkl", "rb") as f:
    #     transit_info = pickle.load(f)
    
    ###############################################################################
    # 创建Transit Autonomous Systems
    Aslist=[3,1,0]
    index =0
    for asnuber in TOPOLOGY_DATA["transit_asns"]:
        asn = assignment[asnuber]['asn']
        prefix = assignment[asnuber]['ipv4']
        exchanges, intra_links = transit_info[asn]
        links=[]
        clique=sorted(exchanges)
        links=generate_connected_pairs(clique, min(2*len(clique),253-len(clique)))
        makeTransitAs(base, asn, prefix, exchanges, list(set(links)),rrNum=1)
        print(f"创建Transit AS{asn}: 连接IXP{exchanges}, 内部链路{links}")
        index+=1
    
    ###############################################################################
    # 创建Stub AS并添加主机

    stub_ix_map = {}
    for (provider, customer, ix, rel) in TOPOLOGY_DATA["as_as_ix_edges"]:
        if rel == '-1' and customer in TOPOLOGY_DATA["stub_asns"]:
            stub_ix_map[customer] = ix
    print(TOPOLOGY_DATA["stub_asns"])
    assignment_temp=assignment.copy()
    for stub_asn in TOPOLOGY_DATA["stub_asns"]:
        assignment_temp=get_assignment(stub_asn,assignment_temp)
        asn=assignment_temp[stub_asn]['asn']
        prefix=assignment_temp[stub_asn]['ipv4']
        ix = assignment_temp[stub_ix_map[stub_asn]]['asn']
        makeStubAsWithHosts(emu, base, asn, prefix, ix, hosts_total=0)
        print(f"创建Stub AS{asn}: 连接IXP{ix}, 主机数量{hosts_per_as}")
    
    # 配置私有对等关系
    for (a, b, ix, rel) in TOPOLOGY_DATA["as_as_ix_edges"]:
        a = assignment_temp[a]['asn']
        b = assignment_temp[b]['asn']
        ix = assignment_temp[ix]['asn']
        rel = int(rel)
        # 转换关系: -1→Provider, 0→Peer
        if rel == -1:
            relationship = PeerRelationship.Provider
        elif rel == 0:
            relationship = PeerRelationship.Peer###########
        else:
            raise ValueError(f"无效关系值: {rel} (仅支持-1和0)")
        
        ebgp.addPrivatePeerings(ix, [a], [b], relationship)
        print(f"IXP{ix}配置私有对等: AS{a}与AS{b} (关系: {rel})")

    t=time.time()

    print("开始编译仿真网络...")
    ######################################################
    # #########################
    # 添加所有层到仿真器
    emu.addLayer(base)
    emu.addLayer(Routing())
    emu.addLayer(ebgp)
    emu.addLayer(Ibgp())
    emu.addLayer(Ospf())
    
    ###############################################################################
    # 输出结果
    if dumpfile is not None:
        # 保存到文件供其他仿真器使用
        emu.dump(dumpfile)
    else:
        emu.render()
        # 附加Internet Map容器并编译
        docker = Docker(platform=platform)
        emu.compile(docker, output_dir, override=True)#f'real_topology_{x}.txt'
        print(f"仿真网络编译完成，输出目录: {output_dir}")
    print(f"编译时间: {time.time()-t} 秒")

if __name__ == "__main__":
    run()
