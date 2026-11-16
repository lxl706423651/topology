#!/usr/bin/env python3
"""Read real_topology_X.json and the global assignment, then build SeedEMU.

Usage: python createNetwork0915.py 300
Original identifiers remain in JSON; simulation AS/IX IDs come from 2026assignment.pkl.
AS-router count excludes the separate IX route-server infrastructure nodes.
"""
from __future__ import annotations

import os
import time
from ipaddress import IPv4Network
import sys
from collections import defaultdict
from pathlib import Path
from seedemu.core import Emulator
from seedemu.layers import Base, Routing, Ebgp, Ibgp, Ospf, PeerRelationship
from seedemu.compiler import Docker, Platform

# Only these two input paths are configurable globals.
ASSIGNMENT_FILE = Path("/home/lxl/dataTopology/20260915final/2026assignment.pkl")
TOPOLOGY_TEMPLATE = "/home/lxl/dataTopology/20260915final/real_topology_{X}.json"

sys.path.insert(0, str(ASSIGNMENT_FILE.parent))
from topology_io import read_topology
from build_assignment import load_assignment

# -------------------- Part 1: input and address planning --------------------

def get_assignment(asn, assignment):
    """RR原函数的查询入口；精确子图禁止在这里临时新增AS或修改全量映射。"""
    key = str(asn)
    if key not in assignment:
        raise ValueError(f"Missing global assignment: {key}; rebuild build_assignment.py first")
    return assignment


def generate_connected_pairs(nodes, extra_edges_count=0):
    """沿用RR的连环基线；额外边数量明确生效，结果保持确定性。"""
    nodes = sorted(set(nodes))
    if type(extra_edges_count) is not int or extra_edges_count < 0:
        raise ValueError("extra_edges_count must be a nonnegative integer")
    if len(nodes) < 2:
        return []
    if len(nodes) == 2:
        return [(nodes[0], nodes[1])]
    pairs = {tuple(sorted((a, b))) for a, b in zip(nodes, nodes[1:] + nodes[:1])}
    # RR原来的额外边分支受len(pairs)<254控制且未使用extra_edges_count。
    # 此处保留连环，明确按数量补弦；默认0，避免规模阈值悄悄改变拓扑。
    remaining = extra_edges_count
    for offset in range(2, len(nodes) // 2 + 1):
        for index, a in enumerate(nodes):
            if remaining == 0:
                return sorted(pairs)
            pair = tuple(sorted((a, nodes[(index + offset) % len(nodes)])))
            if pair not in pairs:
                pairs.add(pair)
                remaining -= 1
    return sorted(pairs)


def _load_json(data):
    """将固定AS–IX清单适配成RR的ixps/transit_asns/stub_asns等字段。"""
    result = dict(data)
    # 子图JSON没有定义Stub AS；所有AS均走Transit构造，不自动推断Stub。
    stubs = set()
    result["ixps"] = sorted({ix for ixes in data["as_ix"].values() for ix in ixes})
    result["transit_asns"] = [str(a) for a in sorted(set(data["as_ix"]) - stubs)]
    result["stub_asns"] = [str(a) for a in sorted(stubs)]
    result["ix_ix_transit_edges"] = [(a, b, asn, 0) for asn in result["transit_asns"]
                                      for a, b in generate_connected_pairs(data["as_ix"][int(asn)])]
    return result


def load_topology_data(filename):
    """保留RR读取函数名；读取经过验证的JSON，替换原先的文本eval解析。"""
    return _load_json(read_topology(filename))


def makeStubAsWithHosts(emu, base, asn, prefix, exchange, hosts_total, *, node_prefix):
    """沿用RR：创建AS、net0、单router及可选host，再接入指定IX。"""
    if type(hosts_total) is not int or hosts_total < 0:
        raise ValueError("hosts_total must be a nonnegative integer")
    stub_as = base.createAutonomousSystem(asn)
    stub_as.createNetwork("net0", prefix)
    router = stub_as.createRouter(f"r{exchange}")
    router.joinNetwork("net0")
    router.joinNetwork(f"ix{exchange}", str(IPv4Network(node_prefix[exchange])[asn]))
    for counter in range(hosts_total):
        stub_as.createHost(f"host_{counter}").joinNetwork("net0")
    return stub_as


def makeTransitAs(base, asn, prefix, exchanges, intra_ix_links, node_prefix, rrNum=0):
    """沿用RR：每IX一台router、RR/cluster、按给定IX对创建内部网络。"""
    exchanges = sorted(exchanges)
    if not exchanges or len(set(exchanges)) != len(exchanges):
        raise ValueError("Transit exchanges must be nonempty and unique")
    if type(rrNum) is not int or rrNum < 0 or (rrNum and len(exchanges) < 2 * rrNum):
        raise ValueError("Each RR needs at least one client; a single-router AS uses rrNum=0")
    links = sorted({tuple(sorted(pair)) for pair in intra_ix_links})
    reached = {exchanges[0]}
    for a, b in links:
        if a == b or a not in exchanges or b not in exchanges:
            raise ValueError("Internal link references an unknown/same router")
    while True:
        expanded = reached | {b for a, b in links if a in reached} | {a for a, b in links if b in reached}
        if expanded == reached:
            break
        reached = expanded
    if reached != set(exchanges):
        raise ValueError("Internal IX-router links are disconnected")
    # RR原例的/24与net_<ixA>_<ixB>命名保留；超出254条才切换/29。
    mask = 24 if len(links) <= 254 else 29
    subnet_iter = IPv4Network(prefix).subnets(new_prefix=mask)
    first_subnet = next(subnet_iter)
    if len(links) >= 2 ** (mask - IPv4Network(prefix).prefixlen):
        raise ValueError("AS prefix has insufficient internal subnets")
    transit_as = base.createAutonomousSystem(asn)
    routers = {}
    for ix in exchanges:
        routers[ix] = transit_as.createRouter(f"r{ix}")
        routers[ix].addSoftware("iperf3")
        routers[ix].joinNetwork(f"ix{ix}", str(IPv4Network(node_prefix[ix])[asn]))
    if rrNum == 1:
        routers[exchanges[0]].makeRouteReflector(True)
    elif rrNum > 1:
        for i in range(rrNum):
            transit_as.createCluster(f"10.0.0.{i + 1}")
        for index, router in enumerate(routers.values()):
            if index < rrNum:
                router.makeRouteReflector(True)
            router.joinBgpCluster(f"10.0.0.{index % rrNum + 1}")
    for a, b in links:
        subnet = next(subnet_iter)
        name = f"net_{a}_{b}"
        transit_as.createNetwork(name, str(subnet))
        if mask == 24:
            routers[a].joinNetwork(name)
            routers[b].joinNetwork(name)
        else:
            # /29从倒数第二个地址降序分配（偏移6、5），为Docker网关保留偏移1。
            routers[a].joinNetwork(name, str(subnet[-2]))
            routers[b].joinNetwork(name, str(subnet[-3]))
    if not links:
        # 单IX Transit也需要本地前缀；原函数在该边界不会创建内部网。
        transit_as.createNetwork("net0", str(first_subnet))
        routers[exchanges[0]].joinNetwork("net0")
    return transit_as

# -------------------- RR_1897主流程：参数、读取、构建、五层、输出 --------------------

def run(dumpfile=None, hosts_per_as=2):
    """按RR_1897顺序在本函数内完成网络构建；上方函数仅提供辅助操作。"""
    t = time.time()
    # Set the platform information. dump分支同样需要明确的x，避免原例未初始化。
    script_name = os.path.basename(__file__)
    platform = Platform.AMD64
    try:
        if len(sys.argv) > 2:
            raise ValueError(f"用法：python {script_name} [X]；仅接受一个规模参数")
        if len(sys.argv) > 1:
            x = int(sys.argv[1])
            print(f"接收到的参数 x 是: {x}")
        else:
            print("请提供参数 x，默认使用214")
            x = 214
        if x <= 0:
            raise ValueError("X必须为正整数")
        if type(hosts_per_as) is not int or hosts_per_as < 0:
            raise ValueError("hosts_per_as必须为非负整数")
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        sys.exit(1)

    # 加载拓扑数据：沿用RR读取入口，使用现有JSON及顶部绝对路径模板。
    topology_file = Path(TOPOLOGY_TEMPLATE.format(X=x))
    try:
        TOPOLOGY_DATA = load_topology_data(topology_file)
        assignment = load_assignment(ASSIGNMENT_FILE, ASSIGNMENT_FILE.with_name("full_topology.json"))
    except FileNotFoundError as exc:
        print(f"错误: 未找到输入文件 {exc.filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"解析拓扑数据出错: {exc}", file=sys.stderr)
        sys.exit(1)

    # 同一IX的RS三元组按成员去重汇总，供创建RS和配置RS peering使用。
    rs_members = defaultdict(set)
    for members, ix, _ in TOPOLOGY_DATA["rs_groups"]:
        rs_members[ix].update(members)

    output_dir = Path.cwd() / f"output_2026_{x}"

    node_prefix = {value["asn"]: value["ipv4"] for value in assignment.values()}
    emu = Emulator()
    ebgp = Ebgp()
    base = Base()

    # 1. 创建IX，沿用assignment[ix]['asn']及IPv4Network(prefix)[ix]。
    ix_objects = {}
    for ixp in sorted(TOPOLOGY_DATA["ixps"], key=lambda key: assignment[key]["asn"]):
        ix, prefix = assignment[ixp]["asn"], assignment[ixp]["ipv4"]
        ix_obj = base.createInternetExchange(ix, prefix, rsAddress=str(IPv4Network(prefix)[ix]),
                                             create_rs=bool(rs_members.get(ixp)))
        ix_obj.getPeeringLan().setDisplayName(f"IX-{ix} ({ixp})")
        ix_objects[ix] = ix_obj

    # 2. 收集Transit连接；AS–IX完整清单是权威输入，不能因边删掉而丢router。
    transit_info = {}
    for original_asn in TOPOLOGY_DATA["transit_asns"]:
        asn = get_assignment(original_asn, assignment)[original_asn]["asn"]
        exchanges = sorted(assignment[ix]["asn"] for ix in TOPOLOGY_DATA["as_ix"][int(original_asn)])
        intra_links = [(assignment[a]["asn"], assignment[b]["asn"])
                       for a, b, member, _ in TOPOLOGY_DATA["ix_ix_transit_edges"] if member == original_asn]
        transit_info[asn] = (exchanges, intra_links)

    # 3. 创建Transit AS，调用RR同名函数；不增加物理router来充当RR。
    for original_asn in TOPOLOGY_DATA["transit_asns"]:
        asn = assignment[original_asn]["asn"]
        exchanges, intra_links = transit_info[asn]
        makeTransitAs(base, asn, assignment[original_asn]["ipv4"], exchanges,
                      intra_links, node_prefix=node_prefix, rrNum=1 if len(exchanges) > 1 else 0)

    # 4. 创建单router Stub；现有完整assignment只查询，不临时新增成员。
    assignment_temp = assignment.copy()
    for stub_asn in TOPOLOGY_DATA["stub_asns"]:
        assignment_temp = get_assignment(stub_asn, assignment_temp)
        asn = assignment_temp[stub_asn]["asn"]
        ixp = TOPOLOGY_DATA["as_ix"][int(stub_asn)][0]
        makeStubAsWithHosts(emu, base, asn, assignment_temp[stub_asn]["ipv4"],
                            assignment_temp[ixp]["asn"], hosts_total=hosts_per_as, node_prefix=node_prefix)

    # 5. 配置私有对等；沿用RR的addPrivatePeerings及Provider方向。
    for a, b, ix, rel in TOPOLOGY_DATA["as_as_ix_edges"]:
        a, b, ix = assignment_temp[str(a)]["asn"], assignment_temp[str(b)]["asn"], assignment_temp[ix]["asn"]
        relationship = PeerRelationship.Provider if int(rel) == -1 else PeerRelationship.Peer
        ebgp.addPrivatePeerings(ix, [a], [b], relationship)
    # 新数据的RS三元组：同IX一个逻辑RS，成员通过映射后ASN加入。
    for ix, members in sorted(rs_members.items()):
        ebgp.addRsPeers(assignment[ix]["asn"], sorted(assignment[str(a)]["asn"] for a in members))

    # 6. 与RR_1897完全相同的五层及默认Routing()。
    emu.addLayer(base)
    emu.addLayer(Routing())
    emu.addLayer(ebgp)
    emu.addLayer(Ibgp())
    emu.addLayer(Ospf())
    count = 0
    for original_asn, ixes in TOPOLOGY_DATA["as_ix"].items():
        autonomous_system = base.getAutonomousSystem(assignment[str(original_asn)]["asn"])
        count += len(autonomous_system.getRouters())
        for ix in ixes:
            autonomous_system.getRouter(f"r{assignment[ix]['asn']}").setDisplayName(f"AS{original_asn} @ {ix}")
    print(f"已构建{count}个AS router，另有{len(rs_members)}个IX route server、"
          f"{len(TOPOLOGY_DATA['stub_asns']) * hosts_per_as}台Stub host。")

    # 输出结果：与RR_1897一样，在run内完成dump或render/compile。
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
