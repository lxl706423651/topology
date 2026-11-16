#!/usr/bin/env python3
"""Write IPv4 routes through standard rtnetlink without invoking iproute2.

This is the K3a/K3b userspace writer.  It intentionally uses ordinary
RTM_NEWROUTE/RTM_DELROUTE messages, so the kernel path remains the standard
rtnetlink route handler.  On the K3a kernel, simple IPv4 main-table
``proto bird metric 232`` routes are handled by the per-netns route lock fast
path.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
from functools import partial
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import socket
import statistics
import struct
import time


NETLINK_ROUTE = 0
RTM_NEWROUTE = 24
RTM_DELROUTE = 25
NLMSG_ERROR = 2
NLM_F_REQUEST = 0x01
NLM_F_ACK = 0x04
NLM_F_REPLACE = 0x100
NLM_F_EXCL = 0x200
NLM_F_CREATE = 0x400

AF_INET = 2
RT_TABLE_MAIN = 254
RTPROT_BIRD = 12
RT_SCOPE_UNIVERSE = 0
RT_SCOPE_LINK = 253
RT_SCOPE_NOWHERE = 255
RTN_UNICAST = 1
RTN_BLACKHOLE = 6
RTN_UNREACHABLE = 7
RTN_PROHIBIT = 8
RTNH_F_ONLINK = 4
ESRCH = 3

RTA_DST = 1
RTA_OIF = 4
RTA_GATEWAY = 5
RTA_PRIORITY = 6
RTA_TABLE = 15

NLMSG_HDR = struct.Struct("=IHHII")
RTMSG = struct.Struct("=BBBBBBBBI")
RTA_HDR = struct.Struct("=HH")

ROUTER_RE = re.compile(r"(?:^|[-_])as(\d+)brd-r(\d+)")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
CLONE_NEWNET = 0x40000000
LIBC = ctypes.CDLL(None, use_errno=True)


def align4(length: int) -> int:
    return (length + 3) & ~3


def rta(attr_type: int, payload: bytes) -> bytes:
    raw_len = RTA_HDR.size + len(payload)
    return RTA_HDR.pack(raw_len, attr_type) + payload + (b"\0" * (align4(raw_len) - raw_len))


def run_cmd(cmd: list[str], *, timeout: float = 120, check: bool = False):
    import subprocess

    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"{' '.join(cmd)} failed: {msg[-2000:]}")
    return proc


def safe_name(name: str) -> str:
    return SAFE_NAME_RE.sub("_", name).strip("_") or "container"


def command_file_paths(commands_dir: Path, name: str) -> dict[str, Path]:
    base = safe_name(name)
    return {
        "v4_replace": commands_dir / f"{base}.v4.replace",
        "v4_delete": commands_dir / f"{base}.v4.delete",
        "meta": commands_dir / f"{base}.meta.json",
    }


def docker_brdnodes(project: str, limit: int) -> list[dict]:
    ps = run_cmd(
        ["docker", "ps", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.ID}}"],
        check=True,
    )
    ids = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    if not ids:
        return []
    inspect = run_cmd(["docker", "inspect", "-f", "{{.Id}} {{.Name}} {{.State.Pid}} {{.State.Running}}"] + ids, check=True)
    nodes: list[dict] = []
    for line in inspect.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        cid, name, pid_text, running = parts[:4]
        name = name.lstrip("/")
        match = ROUTER_RE.search(name)
        if running.lower() != "true" or not match:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid <= 0:
            continue
        nodes.append({"id": cid, "name": name, "pid": pid, "as": int(match.group(1)), "r": int(match.group(2))})
    nodes.sort(key=lambda item: (item["as"], item["r"], item["name"]))
    if limit > 0:
        nodes = nodes[:limit]
    return nodes


def setns_path(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        if hasattr(os, "setns"):
            os.setns(fd, CLONE_NEWNET)
        else:
            rc = LIBC.setns(fd, CLONE_NEWNET)
            if rc != 0:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err), path)
    finally:
        os.close(fd)


class RtNetlinkSocket:
    def __init__(self, ack_mode: str, timeout: float) -> None:
        self.sock = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, NETLINK_ROUTE)
        self.sock.bind((0, 0))
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.ack_mode = ack_mode
        self.seq = 1

    def close(self) -> None:
        self.sock.close()

    def route_message(self, item: dict, args: argparse.Namespace) -> tuple[int, bytes, bool]:
        seq = self.seq
        self.seq += 1
        op = item["op"]
        network: ipaddress.IPv4Network = item["prefix"]
        msg_type = RTM_DELROUTE if op == "delete" else RTM_NEWROUTE
        flags = NLM_F_REQUEST
        if self.ack_mode == "each":
            flags |= NLM_F_ACK
        if op == "add":
            flags |= NLM_F_CREATE | NLM_F_EXCL
        elif op == "replace":
            flags |= NLM_F_CREATE | NLM_F_REPLACE

        attrs = [rta(RTA_DST, int(network.network_address).to_bytes(4, "big"))]
        if item.get("via"):
            attrs.append(rta(RTA_GATEWAY, int(ipaddress.IPv4Address(item["via"])).to_bytes(4, "big")))
        if item.get("dev"):
            attrs.append(rta(RTA_OIF, struct.pack("=I", socket.if_nametoindex(item["dev"]))))
        attrs.append(rta(RTA_PRIORITY, struct.pack("=I", args.metric)))
        if args.table > 255:
            attrs.append(rta(RTA_TABLE, struct.pack("=I", args.table)))

        scope = item["scope"]
        route_type = item["type"]
        rtmsg = RTMSG.pack(
            AF_INET,
            network.prefixlen,
            0,
            0,
            args.table if args.table <= 255 else 0,
            args.protocol,
            scope,
            route_type,
            item.get("flags", 0),
        )
        payload = rtmsg + b"".join(attrs)
        msg = NLMSG_HDR.pack(NLMSG_HDR.size + len(payload), msg_type, flags, seq, 0) + payload
        return seq, msg, op == "delete"

    def send_batch(self, items: list[dict], args: argparse.Namespace) -> dict:
        if not items:
            return {"success_count": 0, "fail_count": 0, "ignored_delete_miss": 0, "first_fail": None}
        seq_delete: dict[int, bool] = {}
        messages = []
        for item in items:
            seq, msg, is_delete = self.route_message(item, args)
            messages.append(msg)
            seq_delete[seq] = is_delete
        self.sock.send(b"".join(messages))
        if self.ack_mode == "none":
            return {"success_count": len(items), "fail_count": 0, "ignored_delete_miss": 0, "first_fail": None}
        return self._read_acks(seq_delete, args)

    def _read_acks(self, pending: dict[int, bool], args: argparse.Namespace) -> dict:
        success_count = 0
        fail_count = 0
        ignored_delete_miss = 0
        first_fail = None
        while pending:
            data = self.sock.recv(4 * 1024 * 1024)
            offset = 0
            while offset + NLMSG_HDR.size <= len(data):
                length, msg_type, _flags, msg_seq, _pid = NLMSG_HDR.unpack_from(data, offset)
                if length < NLMSG_HDR.size or offset + length > len(data):
                    raise RuntimeError("malformed rtnetlink response")
                payload = data[offset + NLMSG_HDR.size : offset + length]
                offset += align4(length)
                if msg_seq not in pending:
                    continue
                is_delete = pending.pop(msg_seq)
                if msg_type != NLMSG_ERROR:
                    continue
                errno_value = struct.unpack_from("=i", payload, 0)[0]
                if errno_value == 0:
                    success_count += 1
                    continue
                errno_abs = -errno_value
                ignored = bool(args.ignore_delete_miss and is_delete and errno_abs == ESRCH)
                if ignored:
                    ignored_delete_miss += 1
                else:
                    fail_count += 1
                    if first_fail is None:
                        first_fail = {"seq": msg_seq, "errno": errno_value}
        return {
            "success_count": success_count,
            "fail_count": fail_count,
            "ignored_delete_miss": ignored_delete_miss,
            "first_fail": first_fail,
        }


def parse_route_command(line: str, forced_op: str, args: argparse.Namespace) -> dict | None:
    tokens = shlex.split(line)
    if len(tokens) < 3 or tokens[0] != "route":
        raise ValueError(f"unsupported command: {line!r}")
    action = tokens[1]
    idx = 2
    route_type = RTN_UNICAST
    prefix_token = tokens[idx]
    if prefix_token in {"blackhole", "unreachable", "prohibit", "prohibited"}:
        route_type = {
            "blackhole": RTN_BLACKHOLE,
            "unreachable": RTN_UNREACHABLE,
            "prohibit": RTN_PROHIBIT,
            "prohibited": RTN_PROHIBIT,
        }[prefix_token]
        idx += 1
        prefix_token = tokens[idx]
    if prefix_token == "default":
        prefix_token = "0.0.0.0/0"
    network = ipaddress.ip_network(prefix_token, strict=False)
    if network.version != 4:
        return None

    via = ""
    dev = ""
    flags = 0
    multipath = False
    idx += 1
    while idx < len(tokens):
        tok = tokens[idx]
        if tok == "via" and idx + 1 < len(tokens):
            via = tokens[idx + 1]
            idx += 2
        elif tok == "dev" and idx + 1 < len(tokens):
            dev = tokens[idx + 1]
            idx += 2
        elif tok == "onlink":
            flags |= RTNH_F_ONLINK
            idx += 1
        elif tok == "nexthop":
            multipath = True
            if args.multipath_policy == "skip":
                return None
            idx += 1
            while idx < len(tokens) and tokens[idx] != "nexthop":
                if tokens[idx] == "via" and idx + 1 < len(tokens) and not via:
                    via = tokens[idx + 1]
                    idx += 2
                elif tokens[idx] == "dev" and idx + 1 < len(tokens) and not dev:
                    dev = tokens[idx + 1]
                    idx += 2
                elif tokens[idx] == "onlink":
                    flags |= RTNH_F_ONLINK
                    idx += 1
                elif tokens[idx] == "weight" and idx + 1 < len(tokens):
                    idx += 2
                else:
                    idx += 1
            if args.multipath_policy == "first":
                break
        else:
            idx += 1

    if forced_op == "delete" or action in {"del", "delete"}:
        op = "delete"
    elif action == "add":
        op = "add"
    elif action == "replace":
        op = "replace"
    else:
        raise ValueError(f"unsupported route action in {line!r}")

    if route_type == RTN_UNICAST and op != "delete" and not dev:
        return None
    scope = RT_SCOPE_UNIVERSE if via or route_type != RTN_UNICAST else RT_SCOPE_LINK
    if op == "delete":
        scope = RT_SCOPE_NOWHERE
    return {
        "op": op,
        "prefix": network,
        "type": route_type,
        "scope": scope,
        "via": via,
        "dev": dev,
        "flags": flags,
        "multipath": multipath,
    }


def read_command_file(path: Path, forced_op: str, args: argparse.Namespace) -> tuple[list[dict], dict]:
    parsed: list[dict] = []
    stats = {"commands": 0, "parsed": 0, "skipped": 0, "multipath_first": 0, "parse_errors": 0}
    if not path.is_file():
        return parsed, stats
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stats["commands"] += 1
        try:
            item = parse_route_command(line, forced_op, args)
        except Exception:
            stats["parse_errors"] += 1
            if not args.keep_going:
                raise
            continue
        if item is None:
            stats["skipped"] += 1
            continue
        if item.get("multipath"):
            stats["multipath_first"] += 1
        parsed.append(item)
        stats["parsed"] += 1
    return parsed, stats


def synthetic_items(args: argparse.Namespace) -> list[dict]:
    base = int(ipaddress.IPv4Address(args.synthetic_base))
    out = []
    for index in range(args.synthetic_routes):
        network = ipaddress.IPv4Network((base + args.synthetic_start_offset + index, args.synthetic_prefix_len), strict=False)
        route_type = RTN_BLACKHOLE if args.synthetic_route_type == "blackhole" else RTN_UNICAST
        out.append(
            {
                "op": args.synthetic_op,
                "prefix": network,
                "type": route_type,
                "scope": RT_SCOPE_UNIVERSE if route_type != RTN_UNICAST else RT_SCOPE_LINK,
                "via": "",
                "dev": "" if route_type != RTN_UNICAST else args.synthetic_dev,
                "flags": 0,
            }
        )
    return out


def apply_items(items: list[dict], args: argparse.Namespace) -> dict:
    sock = RtNetlinkSocket(args.ack_mode, args.netlink_timeout)
    started = time.monotonic()
    out = {
        "batches": 0,
        "success_count": 0,
        "fail_count": 0,
        "ignored_delete_miss": 0,
        "first_fail": None,
    }
    try:
        for start in range(0, len(items), args.batch):
            stats = sock.send_batch(items[start : start + args.batch], args)
            out["batches"] += 1
            out["success_count"] += stats["success_count"]
            out["fail_count"] += stats["fail_count"]
            out["ignored_delete_miss"] += stats["ignored_delete_miss"]
            if stats["first_fail"] and out["first_fail"] is None:
                fail = dict(stats["first_fail"])
                fail["batch"] = out["batches"]
                out["first_fail"] = fail
    finally:
        sock.close()
    out["kernel_submit_seconds"] = time.monotonic() - started
    return out


def write_one(node: dict, args: argparse.Namespace) -> dict:
    started = time.monotonic()
    paths = command_file_paths(args.commands_dir, node["name"])
    result = {
        "name": node["name"],
        "id": node["id"][:12],
        "pid": node["pid"],
        "as": node["as"],
        "r": node["r"],
        "success": False,
        "command_files": {key: str(value) for key, value in paths.items()},
    }
    original_ns = os.open("/proc/self/ns/net", os.O_RDONLY)
    try:
        replace_cmds, replace_stats = read_command_file(paths["v4_replace"], "replace", args)
        delete_cmds, delete_stats = read_command_file(paths["v4_delete"], "delete", args)
        result["replace_stats"] = replace_stats
        result["delete_stats"] = delete_stats
        result["commands"] = replace_stats["commands"] + delete_stats["commands"]
        result["ops"] = replace_stats["parsed"] + delete_stats["parsed"]
        result["skipped"] = replace_stats["skipped"] + delete_stats["skipped"]
        result["parse_errors"] = replace_stats["parse_errors"] + delete_stats["parse_errors"]
        result["multipath_first"] = replace_stats["multipath_first"] + delete_stats["multipath_first"]
        if result["ops"] == 0:
            result["success"] = result["parse_errors"] == 0
            return result
        setns_path(f"/proc/{node['pid']}/ns/net")
        apply_result = apply_items(replace_cmds + delete_cmds, args)
        result["apply_result"] = apply_result
        result["batches"] = apply_result["batches"]
        result["success_count"] = apply_result["success_count"]
        result["fail_count"] = apply_result["fail_count"]
        result["ignored_delete_miss"] = apply_result["ignored_delete_miss"]
        result["kernel_submit_seconds"] = apply_result["kernel_submit_seconds"]
        result["success"] = result["fail_count"] == 0 and result["parse_errors"] == 0
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        try:
            rc = LIBC.setns(original_ns, CLONE_NEWNET)
            if rc != 0:
                result["restore_error"] = os.strerror(ctypes.get_errno())
        finally:
            os.close(original_ns)
        result["write_seconds"] = time.monotonic() - started
    return result


def run_synthetic(args: argparse.Namespace) -> dict:
    original_ns = os.open("/proc/self/ns/net", os.O_RDONLY)
    started = time.monotonic()
    try:
        if args.netns_pid:
            setns_path(f"/proc/{args.netns_pid}/ns/net")
        elif args.netns_path:
            setns_path(args.netns_path)
        items = synthetic_items(args)
        stats = apply_items(items, args)
        stats.update(
            {
                "mode": "rtnetlink-synthetic-writer",
                "routes_requested": args.synthetic_routes,
                "wall_seconds": time.monotonic() - started,
                "routes_per_second": stats["success_count"] / (time.monotonic() - started) if time.monotonic() > started else None,
            }
        )
        stats["success"] = stats["fail_count"] == 0
        return stats
    finally:
        try:
            LIBC.setns(original_ns, CLONE_NEWNET)
        finally:
            os.close(original_ns)


def numeric_summary(rows: list[dict], key: str) -> dict:
    vals = [float(row.get(key) or 0.0) for row in rows]
    if not vals:
        return {"sum": 0.0}
    ordered = sorted(vals)
    return {
        "sum": sum(vals),
        "avg": statistics.fmean(vals),
        "p50": statistics.median(vals),
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": max(vals),
    }


def aggregate(args: argparse.Namespace, nodes: list[dict], rows: list[dict], started: float) -> dict:
    ok = [row for row in rows if row.get("success")]
    failed = [row for row in rows if not row.get("success")]
    return {
        "mode": "k3a-rtnetlink-command-writer",
        "project_name": args.project_name,
        "commands_dir": str(args.commands_dir),
        "jobs": args.jobs,
        "batch_size": args.batch,
        "ack_mode": args.ack_mode,
        "metric": args.metric,
        "protocol": args.protocol,
        "table": args.table,
        "multipath_policy": args.multipath_policy,
        "ignore_delete_miss": args.ignore_delete_miss,
        "containers": len(nodes),
        "write_successes": len(ok),
        "write_failures": len(failed),
        "success": len(nodes) > 0 and not failed,
        "wall_seconds": time.monotonic() - started,
        "commands": sum(int(row.get("commands") or 0) for row in rows),
        "ops_requested": sum(int(row.get("ops") or 0) for row in rows),
        "success_count": sum(int(row.get("success_count") or 0) for row in rows),
        "fail_count": sum(int(row.get("fail_count") or 0) for row in rows),
        "ignored_delete_miss": sum(int(row.get("ignored_delete_miss") or 0) for row in rows),
        "skipped": sum(int(row.get("skipped") or 0) for row in rows),
        "parse_errors": sum(int(row.get("parse_errors") or 0) for row in rows),
        "multipath_first": sum(int(row.get("multipath_first") or 0) for row in rows),
        "batches": sum(int(row.get("batches") or 0) for row in rows),
        "kernel_submit_seconds_sum": sum(float(row.get("kernel_submit_seconds") or 0.0) for row in rows),
        "write_seconds": numeric_summary(rows, "write_seconds"),
        "kernel_submit_seconds": numeric_summary(rows, "kernel_submit_seconds"),
        "failed_write_samples": failed[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-name", default="topology-291e63c1")
    parser.add_argument("--commands-dir", type=Path)
    parser.add_argument("--metric", type=int, default=232)
    parser.add_argument("--protocol", type=int, default=RTPROT_BIRD)
    parser.add_argument("--table", type=int, default=RT_TABLE_MAIN)
    parser.add_argument("--multipath-policy", choices=["skip", "first"], default="first")
    parser.add_argument("--ignore-delete-miss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ack-mode", choices=["each", "none"], default="each")
    parser.add_argument("--netlink-timeout", type=float, default=120)
    parser.add_argument("--limit-containers", type=int, default=0)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--details-path", type=Path, default=None)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--netns-pid", type=int)
    target.add_argument("--netns-path")
    parser.add_argument("--synthetic-routes", type=int, default=0)
    parser.add_argument("--synthetic-base", default="10.0.0.0")
    parser.add_argument("--synthetic-start-offset", type=int, default=0)
    parser.add_argument("--synthetic-prefix-len", type=int, default=32)
    parser.add_argument("--synthetic-route-type", choices=["dev", "blackhole"], default="dev")
    parser.add_argument("--synthetic-dev", default="lo")
    parser.add_argument("--synthetic-op", choices=["replace", "add", "delete"], default="replace")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.geteuid() != 0:
        raise SystemExit("rtnetlink_route_writer.py requires root for setns and route updates; run with sudo -n")
    if args.batch <= 0:
        raise SystemExit("--batch must be positive")
    if args.synthetic_routes:
        summary = run_synthetic(args)
        if args.summary_path:
            args.summary_path.parent.mkdir(parents=True, exist_ok=True)
            args.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary.get("success") else 1
    if not args.commands_dir:
        raise SystemExit("--commands-dir is required unless --synthetic-routes is set")
    args.commands_dir = args.commands_dir.expanduser().resolve()
    if not args.commands_dir.is_dir():
        raise SystemExit(f"missing --commands-dir: {args.commands_dir}")
    started = time.monotonic()
    nodes = docker_brdnodes(args.project_name, args.limit_containers)
    if not nodes:
        raise SystemExit(f"no running brdnodes found for project {args.project_name}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        rows = list(executor.map(partial(write_one, args=args), nodes))
    summary = aggregate(args, nodes, rows, started)
    if args.details_path:
        args.details_path.parent.mkdir(parents=True, exist_ok=True)
        args.details_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["success"] and summary["fail_count"] == 0 and summary["parse_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
