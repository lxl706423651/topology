#!/usr/bin/env python3
"""Write BIRD source-delta events to container FIBs from the VM host.

This is the fast path for large experiments: it reads each container's event
file through /proc/<pid>/root, collapses to the final event per prefix, and
uses nsenter+ip batch to update the container network namespace.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
from functools import partial
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from delta_fib_agent import parse_route_event, route_command, route_order


ROUTER_RE = re.compile(r"^as\d+brd-r")
IPV4_PREFIX_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){3}/\d{1,2})\s+unicast\b.*\*")
VIA_RE = re.compile(r"^\s+via\s+(\d{1,3}(?:\.\d{1,3}){3})\s+on\s+(\S+)")
DEV_RE = re.compile(r"^\s+dev\s+(\S+)")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
CLONE_NEWNET = 0x40000000
LIBC = ctypes.CDLL(None, use_errno=True)


def run_cmd(
    cmd: list[str],
    *,
    input_text: str | None = None,
    timeout: float = 120,
    check: bool = False,
) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, input=input_text, text=True, capture_output=True, timeout=timeout)
    if check and proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "command failed"
        raise RuntimeError(f"{' '.join(cmd)}: {msg}")
    return proc


def delete_miss_only(stderr: str) -> bool:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return False
    return all(line == "RTNETLINK answers: No such process" or re.fullmatch(r"Command failed -:\d+", line) for line in lines)


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
        if running.lower() != "true" or not ROUTER_RE.match(name):
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid <= 0:
            continue
        nodes.append({"id": cid, "name": name, "pid": pid})
    nodes.sort(key=lambda item: item["name"])
    if limit > 0:
        nodes = nodes[:limit]
    return nodes


def event_file_for(pid: int, event_file: str) -> Path:
    rel = event_file.lstrip("/")
    return Path(f"/proc/{pid}/root") / rel


def offset_for(pid: int, offset_mark: str | None, explicit_offset: int) -> int:
    if explicit_offset > 0:
        return explicit_offset
    if not offset_mark:
        return 0
    mark_path = Path(f"/proc/{pid}/root") / offset_mark.lstrip("/")
    try:
        return int(mark_path.read_text(encoding="utf-8").strip() or "0")
    except FileNotFoundError:
        return 0


def collapse_events(path: Path, since_offset: int, max_events: int) -> tuple[dict, dict]:
    started = time.monotonic()
    pending_raw: dict[str, str] = {}
    stats = {
        "raw_route_events": 0,
        "parse_errors": 0,
        "final_parse_errors": 0,
        "start_byte_offset": since_offset,
        "end_byte_offset": since_offset,
    }

    with path.open("rb") as fh:
        if since_offset > 0:
            fh.seek(since_offset, os.SEEK_SET)
        stats["start_byte_offset"] = fh.tell()
        for raw in fh:
            stats["end_byte_offset"] = fh.tell()
            line = raw.decode("utf-8", errors="replace")
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t", 6)
            if len(fields) < 6 or fields[1] != "R":
                continue
            prefix = fields[5]
            if not prefix:
                stats["parse_errors"] += 1
                continue
            stats["raw_route_events"] += 1
            pending_raw[prefix] = line
            if max_events and stats["raw_route_events"] >= max_events:
                break

    pending = {}
    for line in pending_raw.values():
        try:
            event = parse_route_event(line)
        except Exception:
            stats["final_parse_errors"] += 1
            continue
        if event is not None:
            pending[event.prefix] = event

    stats["read_collapse_seconds"] = time.monotonic() - started
    stats["collapsed_events"] = len(pending)
    return pending, stats


def parse_bird_primary_routes(text: str) -> list[dict]:
    routes: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        route_match = IPV4_PREFIX_RE.match(line)
        if route_match:
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
    return routes


def bird_route_to_ip_batch_line(route: dict, metric: int, route_action: str) -> str | None:
    prefix = route["prefix"]
    nexthops = route["nexthops"]
    if not nexthops:
        return None
    op = "add" if route_action == "add" else "replace"
    if len(nexthops) == 1:
        hop = nexthops[0]
        if hop.get("via"):
            return f"route {op} {prefix} via {hop['via']} dev {hop['dev']} proto bird metric {metric}"
        return f"route {op} {prefix} dev {hop['dev']} proto bird metric {metric}"
    parts = [f"route {op} {prefix} proto bird metric {metric}"]
    for hop in nexthops:
        if hop.get("via"):
            parts.append(f"nexthop via {hop['via']} dev {hop['dev']}")
        else:
            parts.append(f"nexthop dev {hop['dev']}")
    return " ".join(parts)


def birdc_primary_commands(pid: int, args: argparse.Namespace) -> tuple[list[str], dict]:
    started = time.monotonic()
    proc = run_cmd(
        ["nsenter", "-t", str(pid), "-m", "-n", "--", "birdc", "-s", "/run/bird/bird.ctl", "show", "route", "primary"],
        timeout=args.command_timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "birdc show route primary failed")[-2000:])
    routes = parse_bird_primary_routes(proc.stdout)
    lines = []
    skipped = 0
    for route in routes:
        line = bird_route_to_ip_batch_line(route, args.metric, args.route_action)
        if line:
            lines.append(line)
        else:
            skipped += 1
    return lines, {
        "raw_route_events": len(routes),
        "collapsed_events": len(routes),
        "commands": len(lines),
        "parse_errors": 0,
        "final_parse_errors": 0,
        "read_collapse_seconds": time.monotonic() - started,
        "skipped_events": skipped,
    }


def safe_name(name: str) -> str:
    return SAFE_NAME_RE.sub("_", name).strip("_") or "container"


def command_file_paths(commands_dir: Path, name: str) -> dict[str, Path]:
    base = safe_name(name)
    return {
        "v4_replace": commands_dir / f"{base}.v4.replace",
        "v6_replace": commands_dir / f"{base}.v6.replace",
        "v4_delete": commands_dir / f"{base}.v4.delete",
        "v6_delete": commands_dir / f"{base}.v6.delete",
        "meta": commands_dir / f"{base}.meta.json",
    }


def write_command_files(
    commands_dir: Path,
    node: dict,
    v4_replace: list[str],
    v6_replace: list[str],
    v4_delete: list[str],
    v6_delete: list[str],
    stats: dict,
) -> dict:
    commands_dir.mkdir(parents=True, exist_ok=True)
    paths = command_file_paths(commands_dir, node["name"])
    command_sets = {
        "v4_replace": v4_replace,
        "v6_replace": v6_replace,
        "v4_delete": v4_delete,
        "v6_delete": v6_delete,
    }
    for key, lines in command_sets.items():
        paths[key].write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    meta = {
        "name": node["name"],
        "id": node["id"][:12],
        "pid": node["pid"],
        "commands": sum(len(lines) for lines in command_sets.values()),
        "command_counts": {key: len(lines) for key, lines in command_sets.items()},
        "stats": stats,
        "created_at_epoch": time.time(),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def read_lines_if_exists(path: Path) -> list[str]:
    try:
        return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except FileNotFoundError:
        return []


def command_file_commands(node: dict, args: argparse.Namespace) -> tuple[list[str], list[str], list[str], list[str], dict]:
    if not args.commands_dir:
        raise RuntimeError("--commands-dir is required with --source command-files")
    started = time.monotonic()
    paths = command_file_paths(args.commands_dir, node["name"])
    v4_replace = read_lines_if_exists(paths["v4_replace"])
    v6_replace = read_lines_if_exists(paths["v6_replace"])
    v4_delete = read_lines_if_exists(paths["v4_delete"])
    v6_delete = read_lines_if_exists(paths["v6_delete"])
    commands = len(v4_replace) + len(v6_replace) + len(v4_delete) + len(v6_delete)
    return (
        v4_replace,
        v6_replace,
        v4_delete,
        v6_delete,
        {
            "raw_route_events": commands,
            "collapsed_events": commands,
            "commands": commands,
            "parse_errors": 0,
            "final_parse_errors": 0,
            "read_collapse_seconds": time.monotonic() - started,
            "skipped_events": 0,
            "command_files": {key: str(path) for key, path in paths.items()},
        },
    )


def route_write_command(cmd: str, route_action: str, omit_onlink: bool) -> str:
    if omit_onlink:
        cmd = cmd.replace(" onlink", "")
    if route_action == "replace":
        return cmd
    if cmd.startswith("route replace "):
        return "route add " + cmd[len("route replace ") :]
    return cmd


def split_commands(
    events: dict,
    skip_deletes: bool,
    route_action: str,
    omit_onlink: bool,
) -> tuple[list[str], list[str], list[str], list[str], int]:
    v4_replace: list[str] = []
    v6_replace: list[str] = []
    v4_delete: list[str] = []
    v6_delete: list[str] = []
    skipped = 0
    for event in sorted(events.values(), key=route_order):
        cmd = route_command(event)
        if not cmd:
            skipped += 1
            continue
        if event.action == "delete":
            if skip_deletes:
                skipped += 1
                continue
            if event.family == 6:
                v6_delete.append(cmd)
            else:
                v4_delete.append(cmd)
        elif event.family == 6:
            v6_replace.append(route_write_command(cmd, route_action, omit_onlink))
        else:
            v4_replace.append(route_write_command(cmd, route_action, omit_onlink))
    return v4_replace, v6_replace, v4_delete, v6_delete, skipped


def setns_net(pid: int) -> None:
    ns_path = f"/proc/{pid}/ns/net"
    fd = os.open(ns_path, os.O_RDONLY)
    try:
        if hasattr(os, "setns"):
            os.setns(fd, CLONE_NEWNET)
        else:
            rc = LIBC.setns(fd, CLONE_NEWNET)
            if rc != 0:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err), ns_path)
    finally:
        os.close(fd)


def setns_ip_batch(pid: int, family: int, lines: list[str], timeout: float, ignore_delete_miss: bool) -> dict:
    if not lines:
        return {"family": family, "commands": 0, "returncode": 0, "stderr": ""}
    setns_net(pid)
    return run_ip_batch_chunks(["ip", f"-{family}", "-force", "-batch", "-"], family, lines, timeout, ignore_delete_miss, "setns-ip")


def chunk_lines(lines: list[str], batch_lines: int) -> list[list[str]]:
    if batch_lines <= 0 or len(lines) <= batch_lines:
        return [lines]
    return [lines[index : index + batch_lines] for index in range(0, len(lines), batch_lines)]


def run_ip_batch_chunks(
    cmd: list[str],
    family: int,
    lines: list[str],
    timeout: float,
    ignore_delete_miss: bool,
    write_mode: str,
    *,
    batch_lines: int = 0,
    batch_sleep_ms: float = 0.0,
) -> dict:
    chunks = chunk_lines(lines, batch_lines)
    errors: list[str] = []
    ignored_count = 0
    returncode = 0
    for index, chunk in enumerate(chunks, start=1):
        proc = run_cmd(cmd, input_text="\n".join(chunk) + "\n", timeout=timeout)
        raw = (proc.stderr or proc.stdout or "").strip()
        ignored = bool(ignore_delete_miss and proc.returncode != 0 and delete_miss_only(raw))
        if ignored:
            ignored_count += 1
        elif proc.returncode != 0:
            returncode = proc.returncode if returncode == 0 else returncode
            errors.append(f"chunk {index}/{len(chunks)} rc={proc.returncode}: {raw[-1000:]}")
        if batch_sleep_ms > 0 and index < len(chunks):
            time.sleep(batch_sleep_ms / 1000.0)
    return {
        "family": family,
        "commands": len(lines),
        "chunks": len(chunks),
        "returncode": returncode,
        "stderr": "\n".join(errors)[-4000:],
        "ignored_delete_miss": ignored_count > 0,
        "ignored_delete_miss_chunks": ignored_count,
        "batch_lines": batch_lines,
        "batch_sleep_ms": batch_sleep_ms,
        "write_mode": write_mode,
    }


def nsenter_ip_batch(pid: int, family: int, lines: list[str], timeout: float, ignore_delete_miss: bool) -> dict:
    if not lines:
        return {"family": family, "commands": 0, "returncode": 0, "stderr": ""}
    return run_ip_batch_chunks(
        ["nsenter", "-t", str(pid), "-n", "--", "ip", f"-{family}", "-force", "-batch", "-"],
        family,
        lines,
        timeout,
        ignore_delete_miss,
        "nsenter-ip",
    )


def setns_ip_flush(pid: int, family: int, metric: int, timeout: float) -> dict:
    setns_net(pid)
    proc = run_cmd(["ip", f"-{family}", "route", "flush", "proto", "bird", "metric", str(metric)], timeout=timeout)
    return {"family": family, "returncode": proc.returncode, "stderr": (proc.stderr or proc.stdout or "").strip()[-1000:], "write_mode": "setns-ip"}


def nsenter_ip_flush(pid: int, family: int, metric: int, timeout: float) -> dict:
    proc = run_cmd(
        ["nsenter", "-t", str(pid), "-n", "--", "ip", f"-{family}", "route", "flush", "proto", "bird", "metric", str(metric)],
        timeout=timeout,
    )
    return {"family": family, "returncode": proc.returncode, "stderr": (proc.stderr or proc.stdout or "").strip()[-1000:], "write_mode": "nsenter-ip"}


def ip_batch(pid: int, family: int, lines: list[str], args: argparse.Namespace, ignore_delete_miss: bool) -> dict:
    if args.write_mode == "setns-ip":
        if not lines:
            return {"family": family, "commands": 0, "returncode": 0, "stderr": ""}
        setns_net(pid)
        return run_ip_batch_chunks(
            ["ip", f"-{family}", "-force", "-batch", "-"],
            family,
            lines,
            args.command_timeout,
            ignore_delete_miss,
            "setns-ip",
            batch_lines=args.batch_lines,
            batch_sleep_ms=args.batch_sleep_ms,
        )
    return run_ip_batch_chunks(
        ["nsenter", "-t", str(pid), "-n", "--", "ip", f"-{family}", "-force", "-batch", "-"],
        family,
        lines,
        args.command_timeout,
        ignore_delete_miss,
        "nsenter-ip",
        batch_lines=args.batch_lines,
        batch_sleep_ms=args.batch_sleep_ms,
    )


def ip_flush(pid: int, family: int, args: argparse.Namespace) -> dict:
    if args.write_mode == "setns-ip":
        return setns_ip_flush(pid, family, args.metric, args.command_timeout)
    return nsenter_ip_flush(pid, family, args.metric, args.command_timeout)


def write_one(node: dict, args: argparse.Namespace) -> dict:
    started_epoch = time.time()
    started = time.monotonic()
    path = event_file_for(node["pid"], args.event_file)
    since_offset = offset_for(node["pid"], args.offset_mark, args.since_byte_offset)
    result = {
        "name": node["name"],
        "id": node["id"][:12],
        "pid": node["pid"],
        "event_file": str(path),
        "started_at_epoch": started_epoch,
        "ended_at_epoch": None,
        "success": False,
        "batch_errors": [],
    }
    try:
        if args.flush_only:
            write_started = time.monotonic()
            flush_results = [ip_flush(node["pid"], 4, args), ip_flush(node["pid"], 6, args)]
            result.update(
                {
                    "raw_route_events": 0,
                    "collapsed_events": 0,
                    "commands": 0,
                    "parse_errors": 0,
                    "read_collapse_seconds": 0.0,
                    "write_seconds": time.monotonic() - write_started,
                    "flush_results": flush_results,
                    "batch_results": [],
                    "batch_errors": [item for item in flush_results if item.get("returncode") != 0],
                    "success": all(item.get("returncode") == 0 for item in flush_results),
                }
            )
            return result

        if args.source == "birdc-primary":
            v4_replace, stats = birdc_primary_commands(node["pid"], args)
            v6_replace: list[str] = []
            v4_delete: list[str] = []
            v6_delete: list[str] = []
            result.update(stats)
        elif args.source == "command-files":
            v4_replace, v6_replace, v4_delete, v6_delete, stats = command_file_commands(node, args)
            result.update(stats)
        else:
            pending, stats = collapse_events(path, since_offset, args.max_events)
            result.update(stats)
            v4_replace, v6_replace, v4_delete, v6_delete, skipped = split_commands(
                pending,
                args.skip_deletes,
                args.route_action,
                args.omit_onlink,
            )
            result["skipped_events"] = skipped
            result["commands"] = len(v4_replace) + len(v6_replace) + len(v4_delete) + len(v6_delete)

        if args.dump_only:
            if not args.commands_dir:
                raise RuntimeError("--commands-dir is required with --dump-only")
            command_files = write_command_files(args.commands_dir, node, v4_replace, v6_replace, v4_delete, v6_delete, result)
            result.update(
                {
                    "write_seconds": 0.0,
                    "flush_results": [],
                    "batch_results": [],
                    "commands_dir": str(args.commands_dir),
                    "command_files": command_files,
                    "dump_only": True,
                    "success": True,
                }
            )
            return result

        write_started = time.monotonic()
        flush_results = []
        if args.flush_first:
            flush_results = [ip_flush(node["pid"], 4, args), ip_flush(node["pid"], 6, args)]
        batch_results = [
            ip_batch(node["pid"], 4, v4_replace, args, False),
            ip_batch(node["pid"], 6, v6_replace, args, False),
            ip_batch(node["pid"], 4, v4_delete, args, True),
            ip_batch(node["pid"], 6, v6_delete, args, True),
        ]
        result["write_seconds"] = time.monotonic() - write_started
        result["flush_results"] = flush_results
        result["batch_results"] = batch_results
        errors = [item for item in flush_results + batch_results if item.get("returncode") != 0]
        result["batch_errors"] = errors
        result["ignored_delete_miss_batches"] = sum(1 for item in batch_results if item.get("ignored_delete_miss"))
        result["success"] = not errors
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        result["ended_at_epoch"] = time.time()
        result["elapsed_seconds"] = time.monotonic() - started
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


def aggregate(nodes: list[dict], rows: list[dict], started: float, args: argparse.Namespace) -> dict:
    ok = [row for row in rows if row.get("success")]
    failed = [row for row in rows if not row.get("success")]
    return {
        "mode": "host-delta-writer",
        "source": args.source,
        "project_name": args.project_name,
        "metric": args.metric,
        "flush_first": args.flush_first,
        "flush_only": args.flush_only,
        "skip_deletes": args.skip_deletes,
        "route_action": args.route_action,
        "omit_onlink": args.omit_onlink,
        "jobs": args.jobs,
        "write_mode": args.write_mode,
        "dump_only": args.dump_only,
        "commands_dir": str(args.commands_dir) if args.commands_dir else None,
        "batch_lines": args.batch_lines,
        "batch_sleep_ms": args.batch_sleep_ms,
        "wall_seconds": time.monotonic() - started,
        "total_brdnodes": len(nodes),
        "successes": len(ok),
        "failures": len(failed),
        "failed_samples": failed[:20],
        "raw_route_events": sum(int(row.get("raw_route_events") or 0) for row in ok),
        "collapsed_events": sum(int(row.get("collapsed_events") or 0) for row in ok),
        "commands": sum(int(row.get("commands") or 0) for row in ok),
        "parse_errors": sum(int(row.get("parse_errors") or 0) for row in ok),
        "batch_errors_total": sum(len(row.get("batch_errors") or []) for row in ok),
        "elapsed_seconds": numeric_summary(ok, "elapsed_seconds"),
        "read_collapse_seconds": numeric_summary(ok, "read_collapse_seconds"),
        "write_seconds": numeric_summary(ok, "write_seconds"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-name", default="topology-291e63c1")
    parser.add_argument("--metric", type=int, default=232)
    parser.add_argument("--event-file", default="/run/bird/delta-events.tsv")
    parser.add_argument("--source", choices=["event", "birdc-primary", "command-files"], default="event")
    parser.add_argument("--dump-only", action="store_true", help="only dump generated ip batch commands into --commands-dir; do not write FIB")
    parser.add_argument("--commands-dir", type=Path, default=None, help="directory for dumped or precomputed ip batch command files")
    parser.add_argument("--offset-mark", default=None)
    parser.add_argument("--since-byte-offset", type=int, default=0)
    parser.add_argument("--flush-first", action="store_true")
    parser.add_argument("--flush-only", action="store_true")
    parser.add_argument("--skip-deletes", action="store_true", help="skip delete events; only valid when writing into an empty FIB")
    parser.add_argument(
        "--route-action",
        choices=["replace", "add"],
        default="replace",
        help="use route add for clean-FIB initial writes; reconvergence should keep replace",
    )
    parser.add_argument(
        "--omit-onlink",
        action="store_true",
        help="drop onlink from generated commands; useful to compare against the old F writer",
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--executor", choices=["process", "thread"], default="process")
    parser.add_argument("--write-mode", choices=["setns-ip", "nsenter-ip"], default="setns-ip")
    parser.add_argument("--batch-lines", type=int, default=0, help="split each ip -batch write into chunks of this many lines; 0 writes one batch")
    parser.add_argument("--batch-sleep-ms", type=float, default=0.0, help="sleep between ip -batch chunks")
    parser.add_argument("--limit-containers", type=int, default=0)
    parser.add_argument("--command-timeout", type=float, default=180)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--details-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.commands_dir:
        args.commands_dir = args.commands_dir.expanduser().resolve()
    if args.write_mode == "setns-ip" and os.geteuid() != 0:
        raise SystemExit("--write-mode setns-ip requires root; run with sudo or use --write-mode nsenter-ip")
    started = time.monotonic()
    nodes = docker_brdnodes(args.project_name, args.limit_containers)
    if not nodes:
        raise SystemExit(f"no running brdnodes found for project {args.project_name}")

    executor_class = concurrent.futures.ProcessPoolExecutor if args.executor == "process" else concurrent.futures.ThreadPoolExecutor
    with executor_class(max_workers=args.jobs) as executor:
        rows = list(executor.map(partial(write_one, args=args), nodes))

    summary = aggregate(nodes, rows, started, args)
    if args.details_path:
        args.details_path.parent.mkdir(parents=True, exist_ok=True)
        args.details_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.summary_path:
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["failures"] == 0 and summary["batch_errors_total"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
