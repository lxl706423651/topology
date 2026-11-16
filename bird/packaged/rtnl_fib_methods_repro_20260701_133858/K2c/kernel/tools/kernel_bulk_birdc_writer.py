#!/usr/bin/env python3
"""Write real BIRD primary IPv4 routes through the K1 BIRD_RT_BULK API.

This tool is intentionally separate from the existing F/delta writers.  It
dumps selected BIRD routes from running brdnode containers, translates them
into the compact K1 kernel route-op format, enters each container netns, and
submits route batches through the experimental generic-netlink family.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
from collections import OrderedDict
from functools import partial
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import time

from kernel_bulk_writer import (
    BIRD_RTBULK_OP_ADD,
    BIRD_RTBULK_OP_DEL,
    BIRD_RTBULK_OP_REPLACE,
    MAX_OPS_PER_ATTR,
    RTBULK_OP,
    RTN_UNICAST,
    RT_SCOPE_LINK,
    RT_SCOPE_NOWHERE,
    RT_SCOPE_UNIVERSE,
    RT_TABLE_MAIN,
    GenlSocket,
    apply_batch,
    BIRD_RTBULK_GENL_NAME,
)


ROUTER_RE = re.compile(r"(?:^|[-_])as(\d+)brd-r(\d+)")
IPV4_PREFIX_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){3}/\d{1,2})\s+unicast\b.*\*")
VIA_RE = re.compile(r"^\s+via\s+(\d{1,3}(?:\.\d{1,3}){3})\s+on\s+(\S+)")
DEV_RE = re.compile(r"^\s+dev\s+(\S+)")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
CLONE_NEWNET = 0x40000000
LIBC = ctypes.CDLL(None, use_errno=True)
RTPROT_BIRD = 12


def run_cmd(cmd: list[str], *, input_text: str | None = None, timeout: float = 120, check: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, input=input_text, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"{' '.join(cmd)} failed: {msg[-2000:]}")
    return proc


def safe_name(name: str) -> str:
    return SAFE_NAME_RE.sub("_", name).strip("_") or "container"


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


def parse_bird_primary_routes(text: str) -> tuple[list[dict], int]:
    routes: list[dict] = []
    current: dict | None = None
    prefixes_seen = 0
    for line in text.splitlines():
        route_match = IPV4_PREFIX_RE.match(line)
        if route_match:
            prefixes_seen += 1
            if current and current["nexthops"]:
                routes.append(current)
            current = {"prefix": route_match.group(1), "nexthops": []}
            continue
        if current is None:
            continue
        via_match = VIA_RE.match(line)
        if via_match:
            current["nexthops"].append({"via": via_match.group(1), "dev": via_match.group(2)})
            continue
        dev_match = DEV_RE.match(line)
        if dev_match:
            current["nexthops"].append({"dev": dev_match.group(1)})
    if current and current["nexthops"]:
        routes.append(current)
    return routes, prefixes_seen


def route_to_dump_line(route: dict, multipath_policy: str) -> tuple[str | None, str | None]:
    hops = route.get("nexthops") or []
    if not hops:
        return None, "no_nexthop"
    if len(hops) > 1:
        if multipath_policy == "skip":
            return None, "multipath"
        reason = "multipath_first"
    else:
        reason = None
    hop = hops[0]
    dev = hop.get("dev")
    if not dev:
        return None, "missing_dev"
    via = hop.get("via") or ""
    scope = RT_SCOPE_UNIVERSE if via else RT_SCOPE_LINK
    return f"{route['prefix']}\t{via}\t{dev}\t{scope}\n", reason


def dump_one(node: dict, args: argparse.Namespace) -> dict:
    started = time.monotonic()
    out_path = args.dump_dir / f"{safe_name(node['name'])}.routes.tsv"
    result = {
        "name": node["name"],
        "id": node["id"][:12],
        "pid": node["pid"],
        "as": node.get("as"),
        "r": node.get("r"),
        "dump_file": str(out_path),
        "success": False,
    }
    try:
        proc = run_cmd(
            ["nsenter", "-t", str(node["pid"]), "-m", "-n", "--", "birdc", "-s", "/run/bird/bird.ctl", "show", "route", "primary"],
            timeout=args.command_timeout,
        )
        result["birdc_returncode"] = proc.returncode
        if proc.returncode != 0:
            result["error"] = (proc.stderr or proc.stdout or "birdc show route primary failed")[-2000:]
            return result
        routes, prefixes_seen = parse_bird_primary_routes(proc.stdout)
        stats = {
            "prefixes_seen": prefixes_seen,
            "routes_with_nexthop": len(routes),
            "ops": 0,
            "skipped_no_nexthop": max(0, prefixes_seen - len(routes)),
            "skipped_multipath": 0,
            "multipath_first": 0,
            "skipped_missing_dev": 0,
            "parse_errors": 0,
        }
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for route in routes:
                line, reason = route_to_dump_line(route, args.multipath_policy)
                if reason == "multipath":
                    stats["skipped_multipath"] += 1
                elif reason == "multipath_first":
                    stats["multipath_first"] += 1
                elif reason == "missing_dev":
                    stats["skipped_missing_dev"] += 1
                if line:
                    fh.write(line)
                    stats["ops"] += 1
        result.update(stats)
        result["success"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        result["dump_seconds"] = time.monotonic() - started
    return result


def setns_path(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        rc = LIBC.setns(fd, CLONE_NEWNET)
        if rc != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), path)
    finally:
        os.close(fd)


def pack_op(prefix: str, via: str, dev: str, scope: int, ifindex_cache: dict[str, int], args: argparse.Namespace) -> bytes:
    network = ipaddress.IPv4Network(prefix, strict=False)
    if network.version != 4:
        raise ValueError(f"non-IPv4 prefix: {prefix}")
    ifindex = ifindex_cache.get(dev)
    if ifindex is None:
        ifindex = socket.if_nametoindex(dev)
        ifindex_cache[dev] = ifindex
    gateway = int(ipaddress.IPv4Address(via)).to_bytes(4, "big") if via else b"\0\0\0\0"
    op = {
        "replace": BIRD_RTBULK_OP_REPLACE,
        "add": BIRD_RTBULK_OP_ADD,
        "del": BIRD_RTBULK_OP_DEL,
    }[args.route_action]
    if op == BIRD_RTBULK_OP_DEL:
        scope = RT_SCOPE_NOWHERE
    return RTBULK_OP.pack(
        op,
        network.prefixlen,
        RTN_UNICAST,
        scope,
        args.protocol,
        b"\0\0\0",
        int(network.network_address).to_bytes(4, "big"),
        gateway,
        ifindex,
        args.table,
        args.metric,
        0,
    )


def route_info_key(via: str, dev: str, scope: int, args: argparse.Namespace) -> tuple:
    op = {
        "replace": BIRD_RTBULK_OP_REPLACE,
        "add": BIRD_RTBULK_OP_ADD,
        "del": BIRD_RTBULK_OP_DEL,
    }[args.route_action]
    if op == BIRD_RTBULK_OP_DEL:
        scope = RT_SCOPE_NOWHERE
    return (op, RTN_UNICAST, scope, via, dev, args.protocol, args.table, args.metric)


def record_batch_result(result: dict, stats: dict, batch_first_global: int) -> None:
    result["batches"] += 1
    result["success_count"] += stats["success_count"]
    result["fail_count"] += stats["fail_count"]
    result["kernel_apply_seconds"] += stats["apply_nsec"] / 1_000_000_000
    if stats["fail_count"] and result["first_fail"] is None:
        result["first_fail"] = {
            "batch": result["batches"],
            "batch_index": stats["first_fail_index"],
            "global_index": batch_first_global + stats["first_fail_index"],
            "errno": stats["first_fail_errno"],
        }


def apply_packed_ops(
    family: GenlSocket,
    family_id: int,
    packed_ops: list[bytes],
    args: argparse.Namespace,
    result: dict,
    *,
    global_start: int,
) -> None:
    for start in range(0, len(packed_ops), args.batch):
        chunk_ops = packed_ops[start : start + args.batch]
        stats = apply_batch(family, family_id, b"".join(chunk_ops), args.ops_per_attr)
        record_batch_result(result, stats, global_start + start)


def write_one(dump_result: dict, args: argparse.Namespace) -> dict:
    started = time.monotonic()
    result = {
        "name": dump_result.get("name"),
        "id": dump_result.get("id"),
        "pid": dump_result.get("pid"),
        "as": dump_result.get("as"),
        "r": dump_result.get("r"),
        "dump_file": dump_result.get("dump_file"),
        "success": False,
        "success_count": 0,
        "fail_count": 0,
        "batches": 0,
        "kernel_apply_seconds": 0.0,
        "first_fail": None,
    }
    if not dump_result.get("success"):
        result["error"] = dump_result.get("error", "dump failed")
        result["write_seconds"] = 0.0
        return result
    if int(dump_result.get("ops") or 0) == 0:
        result["success"] = True
        result["write_seconds"] = 0.0
        return result

    original_ns = os.open("/proc/self/ns/net", os.O_RDONLY)
    try:
        setns_path(f"/proc/{dump_result['pid']}/ns/net")
        family = GenlSocket()
        family_id = family.resolve_family(BIRD_RTBULK_GENL_NAME)
        result["family_id"] = family_id
        ifindex_cache: dict[str, int] = {}
        global_index = 0
        if args.group_fib_info:
            grouped_ops: OrderedDict[tuple, list[bytes]] = OrderedDict()
            group_starts: dict[tuple, int] = {}
            with Path(dump_result["dump_file"]).open("r", encoding="utf-8") as fh:
                for line in fh:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != 4:
                        raise ValueError(f"bad dump line in {dump_result['dump_file']}: {line[:200]!r}")
                    prefix, via, dev, scope_text = fields
                    scope = int(scope_text)
                    key = route_info_key(via, dev, scope, args)
                    if key not in grouped_ops:
                        grouped_ops[key] = []
                        group_starts[key] = global_index
                    grouped_ops[key].append(pack_op(prefix, via, dev, scope, ifindex_cache, args))
                    global_index += 1
            result["groups"] = len(grouped_ops)
            for key, packed_ops in grouped_ops.items():
                apply_packed_ops(
                    family, family_id, packed_ops, args, result, global_start=group_starts[key]
                )
        else:
            batch = bytearray()
            batch_first_global = 0
            with Path(dump_result["dump_file"]).open("r", encoding="utf-8") as fh:
                for line in fh:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != 4:
                        raise ValueError(f"bad dump line in {dump_result['dump_file']}: {line[:200]!r}")
                    prefix, via, dev, scope_text = fields
                    if not batch:
                        batch_first_global = global_index
                    batch.extend(pack_op(prefix, via, dev, int(scope_text), ifindex_cache, args))
                    global_index += 1
                    if len(batch) >= args.batch * RTBULK_OP.size:
                        stats = apply_batch(family, family_id, bytes(batch), args.ops_per_attr)
                        record_batch_result(result, stats, batch_first_global)
                        batch.clear()
            if batch:
                stats = apply_batch(family, family_id, bytes(batch), args.ops_per_attr)
                record_batch_result(result, stats, batch_first_global)
        result["ifindex_cache_size"] = len(ifindex_cache)
        result["success"] = result["fail_count"] == 0 and result["success_count"] == int(dump_result.get("ops") or 0)
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        try:
            rc = LIBC.setns(original_ns, CLONE_NEWNET)
            if rc != 0:
                err = ctypes.get_errno()
                result["restore_error"] = os.strerror(err)
        finally:
            os.close(original_ns)
        result["write_seconds"] = time.monotonic() - started
    return result


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


def aggregate(args: argparse.Namespace, nodes: list[dict], dump_rows: list[dict], write_rows: list[dict], started: float, dump_wall: float, write_wall: float) -> dict:
    dump_ok = [row for row in dump_rows if row.get("success")]
    dump_failed = [row for row in dump_rows if not row.get("success")]
    write_ok = [row for row in write_rows if row.get("success")]
    write_failed = [row for row in write_rows if not row.get("success")]
    return {
        "mode": "k1-real-bird-bulk-writer",
        "project_name": args.project_name,
        "jobs": args.jobs,
        "dump_jobs": args.dump_jobs,
        "batch_size": args.batch,
        "ops_per_attr": args.ops_per_attr,
        "route_action": args.route_action,
        "group_fib_info": args.group_fib_info,
        "metric": args.metric,
        "protocol": args.protocol,
        "table": args.table,
        "multipath_policy": args.multipath_policy,
        "containers": len(nodes),
        "dump_successes": len(dump_ok),
        "dump_failures": len(dump_failed),
        "write_successes": len(write_ok),
        "write_failures": len(write_failed),
        "success": len(nodes) > 0 and not dump_failed and not write_failed,
        "wall_seconds": time.monotonic() - started,
        "dump_wall_seconds": dump_wall,
        "write_wall_seconds": write_wall,
        "prefixes_seen": sum(int(row.get("prefixes_seen") or 0) for row in dump_ok),
        "routes_with_nexthop": sum(int(row.get("routes_with_nexthop") or 0) for row in dump_ok),
        "ops_requested": sum(int(row.get("ops") or 0) for row in dump_ok),
        "success_count": sum(int(row.get("success_count") or 0) for row in write_rows),
        "fail_count": sum(int(row.get("fail_count") or 0) for row in write_rows),
        "skipped_no_nexthop": sum(int(row.get("skipped_no_nexthop") or 0) for row in dump_ok),
        "skipped_multipath": sum(int(row.get("skipped_multipath") or 0) for row in dump_ok),
        "multipath_first": sum(int(row.get("multipath_first") or 0) for row in dump_ok),
        "skipped_missing_dev": sum(int(row.get("skipped_missing_dev") or 0) for row in dump_ok),
        "batches": sum(int(row.get("batches") or 0) for row in write_rows),
        "groups": sum(int(row.get("groups") or 0) for row in write_rows),
        "kernel_apply_seconds_sum": sum(float(row.get("kernel_apply_seconds") or 0.0) for row in write_rows),
        "dump_seconds": numeric_summary(dump_ok, "dump_seconds"),
        "write_seconds": numeric_summary(write_rows, "write_seconds"),
        "kernel_apply_seconds": numeric_summary(write_rows, "kernel_apply_seconds"),
        "failed_dump_samples": dump_failed[:20],
        "failed_write_samples": write_failed[:20],
        "dump_dir": str(args.dump_dir),
        "kept_dumps": args.keep_dumps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-name", default="topology-291e63c1")
    parser.add_argument("--metric", type=int, default=232)
    parser.add_argument("--protocol", type=int, default=RTPROT_BIRD)
    parser.add_argument("--table", type=int, default=RT_TABLE_MAIN)
    parser.add_argument("--route-action", choices=["replace", "add", "del"], default="replace")
    parser.add_argument("--multipath-policy", choices=["skip", "first"], default="skip")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--dump-jobs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=MAX_OPS_PER_ATTR)
    parser.add_argument("--ops-per-attr", type=int, default=MAX_OPS_PER_ATTR)
    parser.add_argument(
        "--group-fib-info",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Group replace/add ops by nexthop/device/metric/table before "
            "submitting. This lets K3c reuse one cached fib_info per attr."
        ),
    )
    parser.add_argument("--limit-containers", type=int, default=0)
    parser.add_argument("--command-timeout", type=float, default=180)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/k1-real-bird-bulk"))
    parser.add_argument("--keep-dumps", action="store_true")
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--details-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.geteuid() != 0:
        raise SystemExit("K1 bulk writer requires root for setns and GENL_ADMIN_PERM; run with sudo -n")
    if args.ops_per_attr <= 0 or args.ops_per_attr > MAX_OPS_PER_ATTR:
        raise SystemExit(f"--ops-per-attr must be in 1..{MAX_OPS_PER_ATTR}")
    if args.batch <= 0:
        raise SystemExit("--batch must be positive")
    args.work_dir = args.work_dir.expanduser().resolve()
    args.dump_dir = args.work_dir / "route_dumps"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    nodes = docker_brdnodes(args.project_name, args.limit_containers)
    if not nodes:
        raise SystemExit(f"no running brdnodes found for project {args.project_name}")

    started = time.monotonic()
    dump_started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.dump_jobs)) as executor:
        dump_rows = list(executor.map(partial(dump_one, args=args), nodes))
    dump_wall = time.monotonic() - dump_started

    write_started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        write_rows = list(executor.map(partial(write_one, args=args), dump_rows))
    write_wall = time.monotonic() - write_started

    summary = aggregate(args, nodes, dump_rows, write_rows, started, dump_wall, write_wall)
    if args.details_path:
        args.details_path.parent.mkdir(parents=True, exist_ok=True)
        args.details_path.write_text(
            json.dumps({"dump": dump_rows, "write": write_rows}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.keep_dumps:
        shutil.rmtree(args.dump_dir, ignore_errors=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["success"] and summary["fail_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
