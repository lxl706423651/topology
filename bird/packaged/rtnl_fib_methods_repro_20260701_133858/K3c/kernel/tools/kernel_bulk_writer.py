#!/usr/bin/env python3
"""User-space writer for the experimental BIRD_RT_BULK kernel API.

This is intentionally independent from the existing F/full/delta writers.  It
is used by the K1/K2 kernel experiments in bird/0618ExperimentDesign.md.

First version:
  * IPv4 only.
  * Synthetic /32 route generation only.
  * Single target network namespace per process.
  * Raw generic-netlink sockets, no libnl dependency.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import struct
import time
from pathlib import Path


NETLINK_GENERIC = 16
GENL_ID_CTRL = 0x10
CTRL_CMD_GETFAMILY = 3
CTRL_ATTR_FAMILY_ID = 1
CTRL_ATTR_FAMILY_NAME = 2

NLM_F_REQUEST = 0x01
NLM_F_ACK = 0x04
NLMSG_ERROR = 0x02

BIRD_RTBULK_GENL_NAME = "BIRD_RT_BULK"
BIRD_RTBULK_GENL_VERSION = 1
BIRD_RTBULK_CMD_APPLY_V4 = 1

BIRD_RTBULK_A_OPS = 1
BIRD_RTBULK_A_SUCCESS_COUNT = 2
BIRD_RTBULK_A_FAIL_COUNT = 3
BIRD_RTBULK_A_FIRST_FAIL_INDEX = 4
BIRD_RTBULK_A_FIRST_FAIL_ERRNO = 5
BIRD_RTBULK_A_APPLY_NSEC = 6

BIRD_RTBULK_OP_REPLACE = 1
BIRD_RTBULK_OP_ADD = 2
BIRD_RTBULK_OP_DEL = 3

RT_TABLE_MAIN = 254
RTPROT_STATIC = 4
RT_SCOPE_UNIVERSE = 0
RT_SCOPE_LINK = 253
RT_SCOPE_NOWHERE = 255
RTN_UNICAST = 1
RTN_BLACKHOLE = 6

NLMSG_HDR = struct.Struct("=IHHII")
GENL_HDR = struct.Struct("=BBH")
NLA_HDR = struct.Struct("=HH")
RTBULK_OP = struct.Struct("=BBBBB3s4s4siIII")
MAX_OPS_PER_ATTR = 2000


def align4(length: int) -> int:
    return (length + 3) & ~3


def nla(attr_type: int, payload: bytes) -> bytes:
    raw_len = NLA_HDR.size + len(payload)
    return NLA_HDR.pack(raw_len, attr_type) + payload + (b"\0" * (align4(raw_len) - raw_len))


def parse_attrs(buf: bytes) -> dict[int, bytes]:
    attrs: dict[int, bytes] = {}
    offset = 0
    while offset + NLA_HDR.size <= len(buf):
        length, attr_type = NLA_HDR.unpack_from(buf, offset)
        if length < NLA_HDR.size or offset + length > len(buf):
            break
        attrs[attr_type] = buf[offset + NLA_HDR.size : offset + length]
        offset += align4(length)
    return attrs


class GenlSocket:
    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, NETLINK_GENERIC)
        # K3b can place many BIRD_RTBULK_A_OPS attrs in one generic-netlink
        # request.  The default socket buffers are often too small for
        # multi-hundred-KB route batches.
        for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
            try:
                self.sock.setsockopt(socket.SOL_SOCKET, opt, 8 * 1024 * 1024)
            except OSError:
                pass
        self.sock.bind((0, 0))
        self.seq = 1

    def request(self, nlmsg_type: int, cmd: int, version: int, attrs: bytes, ack: bool = False) -> bytes:
        seq = self.seq
        self.seq += 1
        flags = NLM_F_REQUEST | (NLM_F_ACK if ack else 0)
        payload = GENL_HDR.pack(cmd, version, 0) + attrs
        msg = NLMSG_HDR.pack(NLMSG_HDR.size + len(payload), nlmsg_type, flags, seq, 0) + payload
        self.sock.send(msg)
        chunks = []
        while True:
            data = self.sock.recv(1024 * 1024)
            offset = 0
            while offset + NLMSG_HDR.size <= len(data):
                length, msg_type, _flags, msg_seq, _pid = NLMSG_HDR.unpack_from(data, offset)
                if length < NLMSG_HDR.size or offset + length > len(data):
                    raise RuntimeError("malformed netlink response")
                payload = data[offset + NLMSG_HDR.size : offset + length]
                offset += align4(length)
                if msg_seq != seq:
                    continue
                if msg_type == NLMSG_ERROR:
                    errno_value = struct.unpack_from("=i", payload, 0)[0]
                    if errno_value:
                        raise OSError(-errno_value, os.strerror(-errno_value))
                    continue
                chunks.append(payload)
            if chunks:
                return b"".join(chunks)

    def resolve_family(self, name: str) -> int:
        attrs = nla(CTRL_ATTR_FAMILY_NAME, name.encode() + b"\0")
        payload = self.request(GENL_ID_CTRL, CTRL_CMD_GETFAMILY, 2, attrs)
        if len(payload) < GENL_HDR.size:
            raise RuntimeError("generic netlink family reply missing header")
        parsed = parse_attrs(payload[GENL_HDR.size :])
        family_id = parsed.get(CTRL_ATTR_FAMILY_ID)
        if not family_id:
            raise RuntimeError(f"generic netlink family not found: {name}")
        return struct.unpack_from("=H", family_id, 0)[0]


def enter_netns(args: argparse.Namespace) -> str | None:
    target = None
    if args.netns_pid:
        target = f"/proc/{args.netns_pid}/ns/net"
    elif args.netns_path:
        target = args.netns_path
    if not target:
        return None
    fd = os.open(target, os.O_RDONLY)
    try:
        os.setns(fd, 0)
    finally:
        os.close(fd)
    return target


def make_op(args: argparse.Namespace, addr_int: int, ifindex: int) -> bytes:
    dst = int(ipaddress.IPv4Address(addr_int)).to_bytes(4, "big")
    gateway = b"\0\0\0\0"
    if args.route_type == "blackhole":
        route_type = RTN_BLACKHOLE
        scope = RT_SCOPE_UNIVERSE
        oif = 0
    else:
        route_type = RTN_UNICAST
        scope = RT_SCOPE_LINK
        oif = ifindex
    op = {
        "replace": BIRD_RTBULK_OP_REPLACE,
        "add": BIRD_RTBULK_OP_ADD,
        "del": BIRD_RTBULK_OP_DEL,
    }[args.op]
    return RTBULK_OP.pack(
        op,
        args.prefix_len,
        route_type,
        scope,
        args.protocol,
        b"\0\0\0",
        dst,
        gateway,
        oif,
        args.table,
        args.metric,
        0,
    )


def apply_batch(sock: GenlSocket, family_id: int, ops: bytes, ops_per_attr: int = MAX_OPS_PER_ATTR) -> dict[str, int]:
    if ops_per_attr <= 0 or ops_per_attr > MAX_OPS_PER_ATTR:
        raise ValueError(f"ops_per_attr must be in 1..{MAX_OPS_PER_ATTR}")
    attr_bytes = ops_per_attr * RTBULK_OP.size
    attrs = b"".join(nla(BIRD_RTBULK_A_OPS, ops[start : start + attr_bytes]) for start in range(0, len(ops), attr_bytes))
    payload = sock.request(
        family_id,
        BIRD_RTBULK_CMD_APPLY_V4,
        BIRD_RTBULK_GENL_VERSION,
        attrs,
    )
    if len(payload) < GENL_HDR.size:
        raise RuntimeError("BIRD_RT_BULK reply missing generic-netlink header")
    attrs = parse_attrs(payload[GENL_HDR.size :])

    def u32(attr_type: int) -> int:
        value = attrs.get(attr_type, b"\0\0\0\0")
        return struct.unpack_from("=I", value, 0)[0]

    def s32(attr_type: int) -> int:
        value = attrs.get(attr_type, b"\0\0\0\0")
        return struct.unpack_from("=i", value, 0)[0]

    def u64(attr_type: int) -> int:
        value = attrs.get(attr_type, b"\0" * 8)
        return struct.unpack_from("=Q", value, 0)[0]

    return {
        "success_count": u32(BIRD_RTBULK_A_SUCCESS_COUNT),
        "fail_count": u32(BIRD_RTBULK_A_FAIL_COUNT),
        "first_fail_index": u32(BIRD_RTBULK_A_FIRST_FAIL_INDEX),
        "first_fail_errno": s32(BIRD_RTBULK_A_FIRST_FAIL_ERRNO),
        "apply_nsec": u64(BIRD_RTBULK_A_APPLY_NSEC),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    entered = enter_netns(args)
    if args.ops_per_attr > MAX_OPS_PER_ATTR:
        raise ValueError(f"--ops-per-attr must be <= {MAX_OPS_PER_ATTR}")
    if args.prefix_len != 32:
        raise ValueError("first writer version only supports /32 synthetic routes")

    ifindex = 0
    if args.route_type == "dev":
        ifindex = socket.if_nametoindex(args.dev)

    family = GenlSocket()
    family_id = family.resolve_family(BIRD_RTBULK_GENL_NAME)

    base = int(ipaddress.IPv4Address(args.base))
    route_count = args.routes
    next_route = 0
    batches = 0
    successes = 0
    failures = 0
    first_fail = None
    kernel_apply_nsec = 0
    started = time.monotonic()

    while next_route < route_count:
        todo = min(args.batch, route_count - next_route)
        ops = bytearray()
        for i in range(todo):
            ops.extend(make_op(args, base + args.start_offset + next_route + i, ifindex))
        stats = apply_batch(family, family_id, bytes(ops), args.ops_per_attr)
        batches += 1
        successes += stats["success_count"]
        failures += stats["fail_count"]
        kernel_apply_nsec += stats["apply_nsec"]
        if stats["fail_count"] and first_fail is None:
            first_fail = {
                "batch": batches,
                "batch_index": stats["first_fail_index"],
                "global_index": next_route + stats["first_fail_index"],
                "errno": stats["first_fail_errno"],
            }
        next_route += todo

    ended = time.monotonic()
    summary = {
        "mode": "kernel-bulk-writer",
        "family_id": family_id,
        "netns": entered,
        "route_type": args.route_type,
        "op": args.op,
        "routes_requested": route_count,
        "batch_size": args.batch,
        "batches": batches,
        "success_count": successes,
        "fail_count": failures,
        "first_fail": first_fail,
        "wall_seconds": ended - started,
        "kernel_apply_seconds": kernel_apply_nsec / 1_000_000_000,
        "routes_per_second": successes / (ended - started) if ended > started else None,
        "base": args.base,
        "start_offset": args.start_offset,
        "prefix_len": args.prefix_len,
        "table": args.table,
        "metric": args.metric,
        "protocol": args.protocol,
        "dev": args.dev if args.route_type == "dev" else None,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write synthetic IPv4 routes through BIRD_RT_BULK.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--netns-pid", type=int, help="Enter /proc/<pid>/ns/net before opening generic netlink.")
    target.add_argument("--netns-path", help="Enter this network namespace path before opening generic netlink.")
    parser.add_argument("--routes", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--ops-per-attr", type=int, default=MAX_OPS_PER_ATTR)
    parser.add_argument("--base", default="10.0.0.0")
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--prefix-len", type=int, default=32)
    parser.add_argument("--route-type", choices=["dev", "blackhole"], default="dev")
    parser.add_argument("--dev", default="lo")
    parser.add_argument("--op", choices=["replace", "add", "del"], default="replace")
    parser.add_argument("--table", type=int, default=RT_TABLE_MAIN)
    parser.add_argument("--metric", type=int, default=232)
    parser.add_argument("--protocol", type=int, default=232)
    parser.add_argument("--summary-path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.summary_path:
        path = Path(args.summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)
    return 0 if summary["fail_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
