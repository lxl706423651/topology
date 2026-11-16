#!/usr/bin/env python3
"""Write precomputed IPv4 route command files through BIRD_RT_BULK.

This is the K1 delta/reconvergence writer.  It consumes command files produced
by delta_fib_host_writer.py --dump-only and submits the IPv4 replace/delete
commands through the experimental BIRD_RT_BULK generic-netlink API.
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
import shlex
import socket
import statistics
import subprocess
import time

from kernel_bulk_writer import (
    BIRD_RTBULK_OP_DEL,
    BIRD_RTBULK_OP_REPLACE,
    BIRD_RTBULK_OP_ADD,
    MAX_OPS_PER_ATTR,
    RTBULK_OP,
    RT_SCOPE_LINK,
    RT_SCOPE_NOWHERE,
    RT_SCOPE_UNIVERSE,
    RT_TABLE_MAIN,
    RTN_BLACKHOLE,
    RTN_UNICAST,
    BIRD_RTBULK_GENL_NAME,
    GenlSocket,
    apply_batch,
)


ROUTER_RE = re.compile(r"(?:^|[-_])as(\d+)brd-r(\d+)")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
CLONE_NEWNET = 0x40000000
LIBC = ctypes.CDLL(None, use_errno=True)
RTPROT_BIRD = 12
RTN_UNREACHABLE = 7
RTN_PROHIBIT = 8
RTNH_F_ONLINK = 4
ESRCH = 3


def run_cmd(cmd: list[str], *, timeout: float = 120, check: bool = False) -> subprocess.CompletedProcess:
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
        rc = LIBC.setns(fd, CLONE_NEWNET)
        if rc != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), path)
    finally:
        os.close(fd)


def parse_route_command(line: str, forced_op: str, multipath_policy: str) -> dict | None:
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

    network = ipaddress.IPv4Network(prefix_token, strict=False)
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
            if multipath_policy == "skip":
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
            if multipath_policy == "first":
                break
        else:
            idx += 1

    if forced_op == "delete" or action in {"del", "delete"}:
        op = BIRD_RTBULK_OP_DEL
    elif action == "add":
        op = BIRD_RTBULK_OP_ADD
    elif action == "replace":
        op = BIRD_RTBULK_OP_REPLACE
    else:
        raise ValueError(f"unsupported route action in {line!r}")

    if route_type == RTN_UNICAST and op != BIRD_RTBULK_OP_DEL and not dev:
        return None

    scope = RT_SCOPE_UNIVERSE if via or route_type != RTN_UNICAST else RT_SCOPE_LINK
    if op == BIRD_RTBULK_OP_DEL:
        # fib_table_delete() treats RT_SCOPE_NOWHERE as a wildcard.  Delta
        # delete commands intentionally omit nexthop scope, so using LINK here
        # would miss existing via routes that were installed with UNIVERSE.
        scope = RT_SCOPE_NOWHERE

    return {
        "op": op,
        "prefix": network,
        "type": route_type,
        "via": via,
        "dev": dev,
        "scope": scope,
        "flags": flags,
        "multipath": multipath,
    }


def pack_parsed(parsed: dict, ifindex_cache: dict[str, int], args: argparse.Namespace) -> bytes:
    network = parsed["prefix"]
    ifindex = 0
    dev = parsed["dev"]
    if dev:
        ifindex = ifindex_cache.get(dev)
        if ifindex is None:
            ifindex = socket.if_nametoindex(dev)
            ifindex_cache[dev] = ifindex
    gateway = int(ipaddress.IPv4Address(parsed["via"])).to_bytes(4, "big") if parsed["via"] else b"\0\0\0\0"
    return RTBULK_OP.pack(
        parsed["op"],
        network.prefixlen,
        parsed["type"],
        parsed["scope"],
        args.protocol,
        b"\0\0\0",
        int(network.network_address).to_bytes(4, "big"),
        gateway,
        ifindex,
        args.table,
        args.metric,
        parsed["flags"],
    )


def fib_info_key(parsed: dict, args: argparse.Namespace) -> tuple:
    return (
        parsed["op"],
        parsed["type"],
        parsed["scope"],
        parsed["via"],
        parsed["dev"],
        parsed["flags"],
        args.protocol,
        args.table,
        args.metric,
    )


def group_parsed_ops(parsed_ops: list[dict], args: argparse.Namespace) -> list[list[dict]]:
    if not parsed_ops:
        return []
    if not args.group_fib_info or len(parsed_ops) == 1:
        return [parsed_ops]

    groups: OrderedDict[tuple, list[dict]] = OrderedDict()
    for item in parsed_ops:
        groups.setdefault(fib_info_key(item, args), []).append(item)
    return list(groups.values())


def apply_ops(
    family: GenlSocket,
    family_id: int,
    packed_ops: list[bytes],
    args: argparse.Namespace,
    *,
    delete_batch: bool,
) -> dict:
    out = {
        "batches": 0,
        "success_count": 0,
        "fail_count": 0,
        "ignored_delete_miss": 0,
        "kernel_apply_seconds": 0.0,
        "first_fail": None,
    }
    for start in range(0, len(packed_ops), args.batch):
        chunk = b"".join(packed_ops[start : start + args.batch])
        stats = apply_batch(family, family_id, chunk, args.ops_per_attr)
        out["batches"] += 1
        fail_count = int(stats["fail_count"])
        first_errno = int(stats["first_fail_errno"])
        ignored = bool(args.ignore_delete_miss and delete_batch and fail_count and first_errno == -ESRCH)
        out["success_count"] += int(stats["success_count"])
        if ignored:
            out["ignored_delete_miss"] += fail_count
        else:
            out["fail_count"] += fail_count
            if fail_count and out["first_fail"] is None:
                out["first_fail"] = {
                    "batch": out["batches"],
                    "batch_index": int(stats["first_fail_index"]),
                    "global_index": start + int(stats["first_fail_index"]),
                    "errno": first_errno,
                }
        out["kernel_apply_seconds"] += int(stats["apply_nsec"]) / 1_000_000_000
    return out


def merge_apply_result(total: dict, item: dict) -> None:
    batch_offset = total["batches"]
    total["batches"] += item["batches"]
    total["success_count"] += item["success_count"]
    total["fail_count"] += item["fail_count"]
    total["ignored_delete_miss"] += item["ignored_delete_miss"]
    total["kernel_apply_seconds"] += item["kernel_apply_seconds"]
    if item["first_fail"] and total["first_fail"] is None:
        first = dict(item["first_fail"])
        first["batch"] += batch_offset
        total["first_fail"] = first


def apply_parsed_groups(
    family: GenlSocket,
    family_id: int,
    parsed_ops: list[dict],
    ifindex_cache: dict[str, int],
    args: argparse.Namespace,
    *,
    delete_batch: bool,
) -> dict:
    out = {
        "batches": 0,
        "groups": 0,
        "success_count": 0,
        "fail_count": 0,
        "ignored_delete_miss": 0,
        "kernel_apply_seconds": 0.0,
        "first_fail": None,
    }
    for group in group_parsed_ops(parsed_ops, args):
        packed_ops = [pack_parsed(item, ifindex_cache, args) for item in group]
        result = apply_ops(family, family_id, packed_ops, args, delete_batch=delete_batch)
        out["groups"] += 1
        merge_apply_result(out, result)
    return out


def apply_mixed_ops(family: GenlSocket, family_id: int, packed_ops: list[bytes], args: argparse.Namespace) -> dict:
    out = {
        "batches": 0,
        "success_count": 0,
        "fail_count": 0,
        "ignored_delete_miss": 0,
        "kernel_apply_seconds": 0.0,
        "first_fail": None,
    }
    for start in range(0, len(packed_ops), args.batch):
        chunk = b"".join(packed_ops[start : start + args.batch])
        stats = apply_batch(family, family_id, chunk, args.ops_per_attr)
        out["batches"] += 1
        fail_count = int(stats["fail_count"])
        first_errno = int(stats["first_fail_errno"])
        ignored = bool(args.ignore_delete_miss and fail_count and first_errno == -ESRCH)
        out["success_count"] += int(stats["success_count"])
        if ignored:
            out["ignored_delete_miss"] += fail_count
        else:
            out["fail_count"] += fail_count
            if fail_count and out["first_fail"] is None:
                out["first_fail"] = {
                    "batch": out["batches"],
                    "batch_index": int(stats["first_fail_index"]),
                    "global_index": start + int(stats["first_fail_index"]),
                    "errno": first_errno,
                }
        out["kernel_apply_seconds"] += int(stats["apply_nsec"]) / 1_000_000_000
    return out


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
            item = parse_route_command(line, forced_op, args.multipath_policy)
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
        family = GenlSocket()
        family_id = family.resolve_family(BIRD_RTBULK_GENL_NAME)
        result["family_id"] = family_id
        ifindex_cache: dict[str, int] = {}
        if args.mixed_batches:
            replace_ops = [pack_parsed(item, ifindex_cache, args) for item in replace_cmds]
            delete_ops = [pack_parsed(item, ifindex_cache, args) for item in delete_cmds]
            mixed_result = apply_mixed_ops(family, family_id, replace_ops + delete_ops, args)
            result["mixed_result"] = mixed_result
            result["batches"] = mixed_result["batches"]
            result["success_count"] = mixed_result["success_count"]
            result["fail_count"] = mixed_result["fail_count"]
            result["ignored_delete_miss"] = mixed_result["ignored_delete_miss"]
            result["kernel_apply_seconds"] = mixed_result["kernel_apply_seconds"]
        else:
            replace_result = apply_parsed_groups(
                family, family_id, replace_cmds, ifindex_cache, args, delete_batch=False
            )
            delete_result = apply_parsed_groups(
                family, family_id, delete_cmds, ifindex_cache, args, delete_batch=True
            )
            result["replace_result"] = replace_result
            result["delete_result"] = delete_result
            result["batches"] = replace_result["batches"] + delete_result["batches"]
            result["groups"] = replace_result["groups"] + delete_result["groups"]
            result["success_count"] = replace_result["success_count"] + delete_result["success_count"]
            result["fail_count"] = replace_result["fail_count"] + delete_result["fail_count"]
            result["ignored_delete_miss"] = delete_result["ignored_delete_miss"]
            result["kernel_apply_seconds"] = replace_result["kernel_apply_seconds"] + delete_result["kernel_apply_seconds"]
        result["ifindex_cache_size"] = len(ifindex_cache)
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
        "mode": "k1-command-bulk-writer",
        "project_name": args.project_name,
        "commands_dir": str(args.commands_dir),
        "jobs": args.jobs,
        "batch_size": args.batch,
        "ops_per_attr": args.ops_per_attr,
        "metric": args.metric,
        "protocol": args.protocol,
        "table": args.table,
        "multipath_policy": args.multipath_policy,
        "ignore_delete_miss": args.ignore_delete_miss,
        "mixed_batches": args.mixed_batches,
        "group_fib_info": args.group_fib_info,
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
        "groups": sum(int(row.get("groups") or 0) for row in rows),
        "kernel_apply_seconds_sum": sum(float(row.get("kernel_apply_seconds") or 0.0) for row in rows),
        "write_seconds": numeric_summary(rows, "write_seconds"),
        "kernel_apply_seconds": numeric_summary(rows, "kernel_apply_seconds"),
        "failed_write_samples": failed[:20],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-name", default="topology-291e63c1")
    parser.add_argument("--commands-dir", type=Path, required=True)
    parser.add_argument("--metric", type=int, default=232)
    parser.add_argument("--protocol", type=int, default=RTPROT_BIRD)
    parser.add_argument("--table", type=int, default=RT_TABLE_MAIN)
    parser.add_argument("--multipath-policy", choices=["skip", "first"], default="first")
    parser.add_argument("--ignore-delete-miss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mixed-batches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Submit replace and delete ops in the same rtbulk batch for each "
            "container. This reduces sparse-delta batch count, but delete misses "
            "cannot be ignored independently inside a mixed batch."
        ),
    )
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jobs", type=int, default=8)
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
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--details-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.geteuid() != 0:
        raise SystemExit("K1 command bulk writer requires root for setns and GENL_ADMIN_PERM; run with sudo -n")
    if args.ops_per_attr <= 0 or args.ops_per_attr > MAX_OPS_PER_ATTR:
        raise SystemExit(f"--ops-per-attr must be in 1..{MAX_OPS_PER_ATTR}")
    if args.batch <= 0:
        raise SystemExit("--batch must be positive")
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
