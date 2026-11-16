#!/usr/bin/env python3
"""Large-scale RTNL/FIB benchmark for D/E/F/G solution prototypes.

This runner uses lightweight Linux network namespaces as simulated routers.
It writes synthetic blackhole routes into an isolated route table, so the
benchmark exercises kernel FIB/RTNL paths without changing the main table.
"""

import argparse
import concurrent.futures
import datetime as dt
import ipaddress
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


DEFAULT_NODES = 5000
DEFAULT_ROUTES_PER_NODE = 10000
DEFAULT_TABLE = 252
DEFAULT_PREFIX_BASE = "198.18.0.0"
DEFAULT_NETNS_PREFIX = "rtnl-scale"
DEFAULT_WORKERS = {
    "D": 1,
    "E": 8,
    "F": 30,
    "G": 1,
}


class BenchError(RuntimeError):
    pass


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run_cmd(cmd, *, input_text=None, timeout=None, check=True):
    proc = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "command failed"
        raise BenchError(f"{' '.join(cmd)}: {msg}")
    return proc


def privileged_prefix():
    if os.geteuid() == 0:
        return []
    if shutil.which("sudo"):
        check = subprocess.run(["sudo", "-n", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if check.returncode == 0:
            return ["sudo", "-n"]
    raise BenchError("root or passwordless sudo is required")


def ip_cmd(prefix, *args):
    return prefix + ["ip"] + list(args)


def netns_name(prefix, idx, width):
    return f"{prefix}-{idx:0{width}d}"


def target_netns_names(prefix, nodes):
    width = max(4, len(str(nodes - 1)))
    return [netns_name(prefix, idx, width) for idx in range(nodes)]


def list_netns(prefix):
    proc = run_cmd(ip_cmd(prefix, "netns", "list"), check=False)
    names = set()
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        names.add(line.split()[0])
    return names


def ensure_route_range(prefix_base, nodes, routes_per_node):
    base = int(ipaddress.IPv4Address(prefix_base))
    total = nodes * routes_per_node
    last = base + total
    if last > 0xFFFFFFFF:
        raise BenchError(f"route range exceeds IPv4 space: base={prefix_base}, total={total}")
    return total


def route_batch(prefix_base, node_idx, routes_per_node, table):
    base = int(ipaddress.IPv4Address(prefix_base)) + node_idx * routes_per_node
    lines = []
    for ridx in range(routes_per_node):
        route = ipaddress.IPv4Address(base + ridx + 1)
        lines.append(f"route replace blackhole {route}/32 table {table}")
    return "\n".join(lines) + "\n"


def load_sample():
    try:
        load1, load5, load15 = os.getloadavg()
        return {"t": time.time(), "load1": load1, "load5": load5, "load15": load15}
    except OSError:
        return {"t": time.time()}


def load_stats(samples):
    vals = [item["load1"] for item in samples if "load1" in item]
    if not vals:
        return {"samples": len(samples)}
    return {
        "samples": len(samples),
        "load1_min": min(vals),
        "load1_avg": statistics.fmean(vals),
        "load1_max": max(vals),
    }


class LoadMonitor:
    def __init__(self, interval):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(load_sample())
            self._stop.wait(self.interval)

    def stats(self):
        return load_stats(self.samples)


def write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(path)


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        json.dump(row, fh, ensure_ascii=False)
        fh.write("\n")


def default_work_dir(nodes, routes_per_node):
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.home() / "bird" / "runs" / f"rtnl-scale_{nodes}x{routes_per_node}_{stamp}"


def create_one_netns(prefix, name, timeout):
    proc = run_cmd(ip_cmd(prefix, "netns", "add", name), timeout=timeout, check=False)
    return {
        "name": name,
        "success": proc.returncode == 0,
        "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip()),
    }


def prepare_netns(prefix, names, workers, timeout, progress_cb):
    existing = list_netns(prefix)
    missing = [name for name in names if name not in existing]
    start = time.monotonic()
    failures = []
    completed = 0
    if not missing:
        progress_cb({"created": 0, "reused": len(names), "failures": 0})
        return {"created": 0, "reused": len(names), "failures": 0, "duration_seconds": 0.0, "errors": []}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(create_one_netns, prefix, name, timeout) for name in missing]
        for fut in concurrent.futures.as_completed(futures):
            item = fut.result()
            completed += 1
            if not item["success"]:
                failures.append(item)
            if completed % 100 == 0 or completed == len(missing):
                progress_cb({"created": completed, "reused": len(names) - len(missing), "failures": len(failures)})

    duration = time.monotonic() - start
    return {
        "created": completed - len(failures),
        "reused": len(names) - len(missing),
        "failures": len(failures),
        "duration_seconds": round(duration, 6),
        "errors": failures[:50],
    }


def flush_one(prefix, name, table, timeout):
    cmd = ip_cmd(prefix, "netns", "exec", name, "ip", "route", "flush", "table", str(table))
    start = time.monotonic()
    proc = run_cmd(cmd, timeout=timeout, check=False)
    output = proc.stderr.strip() or proc.stdout.strip()
    empty_table = proc.returncode != 0 and "FIB table does not exist" in output
    return {
        "name": name,
        "success": proc.returncode == 0 or empty_table,
        "duration_seconds": round(time.monotonic() - start, 6),
        "error": "" if proc.returncode == 0 or empty_table else output,
        "empty_table": empty_table,
    }


def flush_routes(prefix, names, table, workers, timeout, progress_cb):
    start = time.monotonic()
    failures = []
    completed = 0
    durations = []
    existing = list_netns(prefix)
    targets = [name for name in names if name in existing]
    if not targets:
        return aggregate_stage(0.0, 0, 0, [], [], workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(flush_one, prefix, name, table, timeout) for name in targets]
        for fut in concurrent.futures.as_completed(futures):
            item = fut.result()
            completed += 1
            durations.append(item["duration_seconds"])
            if not item["success"]:
                failures.append(item)
            if completed % 100 == 0 or completed == len(targets):
                progress_cb({"completed": completed, "failures": len(failures)})
    duration = time.monotonic() - start
    out = aggregate_stage(duration, len(targets), 0, durations, failures, workers)
    out["skipped_missing"] = len(names) - len(targets)
    return out


def delete_one_netns(prefix, name, timeout):
    proc = run_cmd(ip_cmd(prefix, "netns", "del", name), timeout=timeout, check=False)
    return {
        "name": name,
        "success": proc.returncode == 0,
        "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip()),
    }


def cleanup_netns(prefix, names, workers, timeout, progress_cb):
    start = time.monotonic()
    failures = []
    completed = 0
    existing = list_netns(prefix)
    targets = [name for name in names if name in existing]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(delete_one_netns, prefix, name, timeout) for name in targets]
        for fut in concurrent.futures.as_completed(futures):
            item = fut.result()
            completed += 1
            if not item["success"]:
                failures.append(item)
            if completed % 100 == 0 or completed == len(targets):
                progress_cb({"deleted": completed, "failures": len(failures)})
    return {
        "deleted": completed - len(failures),
        "skipped_missing": len(names) - len(targets),
        "failures": len(failures),
        "duration_seconds": round(time.monotonic() - start, 6),
        "errors": failures[:50],
    }


def write_node_routes(prefix, name, node_idx, args):
    batch = route_batch(args.prefix_base, node_idx, args.routes_per_node, args.table)
    cmd = ip_cmd(prefix, "netns", "exec", name, "ip", "-force", "-batch", "-")
    start = time.monotonic()
    proc = run_cmd(cmd, input_text=batch, timeout=args.node_timeout, check=False)
    duration = time.monotonic() - start
    return {
        "node": name,
        "node_index": node_idx,
        "routes": args.routes_per_node,
        "success": proc.returncode == 0,
        "duration_seconds": round(duration, 6),
        "routes_per_second": round(args.routes_per_node / duration, 3) if duration > 0 else None,
        "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip()),
    }


def aggregate_stage(duration, nodes, route_ops, durations, failures, workers):
    out = {
        "workers": workers,
        "nodes": nodes,
        "route_ops": route_ops,
        "duration_seconds": round(duration, 6),
        "route_ops_per_second": round(route_ops / duration, 3) if route_ops and duration > 0 else None,
        "completed_nodes": len(durations),
        "failures": len(failures),
        "errors": failures[:50],
    }
    if durations:
        out.update(
            {
                "node_duration_min": min(durations),
                "node_duration_avg": statistics.fmean(durations),
                "node_duration_max": max(durations),
            }
        )
    return out


def solution_label(name):
    labels = {
        "D": "D-centralized-host-fib-writer",
        "E": "E-per-netns-bulk-route-batch",
        "F": "F-sharded-per-netns-rtnl-proxy",
        "G": "G-sorted-multi-netns-transaction",
    }
    return labels[name]


def solution_workers(name, args):
    return {
        "D": args.d_workers,
        "E": args.e_workers,
        "F": args.f_shards,
        "G": args.g_workers,
    }[name]


def solution_order(name, indexed_names, args):
    if name == "G":
        return sorted(indexed_names, key=lambda item: item[1])
    if name == "F":
        shards = []
        shard_count = max(1, args.f_shards)
        for shard in range(shard_count):
            shards.extend(item for item in indexed_names if item[1] % shard_count == shard)
        return shards
    return indexed_names


def run_solution(prefix, solution, indexed_names, args, summary, summary_path):
    label = solution_label(solution)
    workers = solution_workers(solution, args)
    ordered = solution_order(solution, indexed_names, args)
    result_path = Path(summary["work_dir"]) / f"{solution}_{label}.jsonl"
    if result_path.exists():
        result_path.unlink()

    stage = {
        "label": label,
        "started_at": now_iso(),
        "ended_at": None,
        "success": None,
        "workers": workers,
        "nodes": len(ordered),
        "routes_per_node": args.routes_per_node,
        "route_ops": len(ordered) * args.routes_per_node,
        "completed_nodes": 0,
        "failed_nodes": 0,
        "result_jsonl": str(result_path),
    }
    summary["solutions"][solution] = stage
    write_json_atomic(summary_path, summary)

    start = time.monotonic()
    failures = []
    durations = []
    route_ops_done = 0
    with LoadMonitor(args.load_sample_interval) as monitor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [
                pool.submit(write_node_routes, prefix, name, node_idx, args)
                for name, node_idx in ordered
            ]
            for fut in concurrent.futures.as_completed(futures):
                item = fut.result()
                append_jsonl(result_path, item)
                durations.append(item["duration_seconds"])
                if not item["success"]:
                    failures.append(item)
                else:
                    route_ops_done += item["routes"]

                completed = len(durations)
                if completed % args.progress_interval == 0 or completed == len(ordered):
                    elapsed = time.monotonic() - start
                    stage.update(
                        {
                            "completed_nodes": completed,
                            "failed_nodes": len(failures),
                            "route_ops_completed": route_ops_done,
                            "elapsed_seconds": round(elapsed, 6),
                            "current_route_ops_per_second": round(route_ops_done / elapsed, 3) if elapsed > 0 else None,
                            "last_progress_at": now_iso(),
                            "load": monitor.stats(),
                        }
                    )
                    write_json_atomic(summary_path, summary)

    duration = time.monotonic() - start
    aggregate = aggregate_stage(duration, len(ordered), route_ops_done, durations, failures, workers)
    aggregate["load"] = monitor.stats()
    aggregate["ideal_all_netns_independent_seconds"] = round(max(durations), 6) if durations else None
    aggregate["actual_over_ideal_ratio"] = (
        round(duration / max(durations), 6) if durations and max(durations) > 0 else None
    )
    stage.update(aggregate)
    stage["success"] = len(failures) == 0
    stage["ended_at"] = now_iso()
    summary["solutions"][solution] = stage
    write_json_atomic(summary_path, summary)
    return stage


def parse_args():
    parser = argparse.ArgumentParser(description="Run a 5000x10000 RTNL scale benchmark in Linux netns.")
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    parser.add_argument("--routes-per-node", type=int, default=DEFAULT_ROUTES_PER_NODE)
    parser.add_argument("--solutions", default="D,E,F,G", help="Comma-separated subset of D,E,F,G")
    parser.add_argument("--table", type=int, default=DEFAULT_TABLE)
    parser.add_argument("--prefix-base", default=DEFAULT_PREFIX_BASE)
    parser.add_argument("--netns-prefix", default=DEFAULT_NETNS_PREFIX)
    parser.add_argument("--work-dir", help="Default: ~/bird/runs/rtnl-scale_<nodes>x<routes>_<timestamp>")
    parser.add_argument("--prepare-workers", type=int, default=16)
    parser.add_argument("--cleanup-workers", type=int, default=16)
    parser.add_argument("--d-workers", type=int, default=DEFAULT_WORKERS["D"])
    parser.add_argument("--e-workers", type=int, default=DEFAULT_WORKERS["E"])
    parser.add_argument("--f-shards", type=int, default=DEFAULT_WORKERS["F"])
    parser.add_argument("--g-workers", type=int, default=DEFAULT_WORKERS["G"])
    parser.add_argument("--node-timeout", type=float, default=3600)
    parser.add_argument("--netns-timeout", type=float, default=120)
    parser.add_argument("--flush-timeout", type=float, default=3600)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--load-sample-interval", type=float, default=5.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--cleanup-netns", action="store_true")
    parser.add_argument("--no-cleanup-before", action="store_true")
    parser.add_argument("--no-cleanup-between", action="store_true")
    parser.add_argument("--cleanup-after", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    selected = [item.strip().upper() for item in args.solutions.split(",") if item.strip()]
    bad = [item for item in selected if item not in {"D", "E", "F", "G"}]
    if bad:
        raise SystemExit(f"unknown solutions: {','.join(bad)}")
    total_route_ops = ensure_route_range(args.prefix_base, args.nodes, args.routes_per_node)

    prefix = privileged_prefix()
    work_dir = Path(args.work_dir).expanduser() if args.work_dir else default_work_dir(args.nodes, args.routes_per_node)
    summary_path = work_dir / "summary.json"
    names = target_netns_names(args.netns_prefix, args.nodes)
    indexed_names = list(zip(names, range(args.nodes)))

    summary = {
        "schema_version": 1,
        "started_at": now_iso(),
        "ended_at": None,
        "success": None,
        "error": None,
        "host": socket.gethostname(),
        "command": sys.argv,
        "work_dir": str(work_dir),
        "target": {
            "nodes": args.nodes,
            "routes_per_node": args.routes_per_node,
            "route_ops_per_solution": total_route_ops,
            "solutions": selected,
            "table": args.table,
            "prefix_base": args.prefix_base,
            "netns_prefix": args.netns_prefix,
        },
        "settings": vars(args),
        "prepare": {},
        "cleanup": {},
        "solutions": {},
    }
    write_json_atomic(summary_path, summary)

    def progress_update(section, data):
        summary.setdefault(section, {}).update(data)
        summary[section]["last_progress_at"] = now_iso()
        write_json_atomic(summary_path, summary)

    try:
        if args.cleanup_only:
            progress_update("cleanup", {"started_at": now_iso(), "mode": "routes"})
            summary["cleanup"]["routes"] = flush_routes(
                prefix,
                names,
                args.table,
                args.cleanup_workers,
                args.flush_timeout,
                lambda data: progress_update("cleanup", {"route_flush_progress": data}),
            )
            if args.cleanup_netns:
                summary["cleanup"]["netns"] = cleanup_netns(
                    prefix,
                    names,
                    args.cleanup_workers,
                    args.netns_timeout,
                    lambda data: progress_update("cleanup", {"netns_delete_progress": data}),
                )
            summary["cleanup"]["ended_at"] = now_iso()
            summary["success"] = True
            return 0

        progress_update("prepare", {"started_at": now_iso(), "nodes": args.nodes})
        summary["prepare"]["netns"] = prepare_netns(
            prefix,
            names,
            args.prepare_workers,
            args.netns_timeout,
            lambda data: progress_update("prepare", {"netns_progress": data}),
        )
        summary["prepare"]["ended_at"] = now_iso()
        if summary["prepare"]["netns"]["failures"]:
            raise BenchError("failed to create all netns")
        if args.prepare_only:
            summary["success"] = True
            return 0

        if not args.no_cleanup_before:
            progress_update("cleanup", {"pre_run_started_at": now_iso()})
            summary["cleanup"]["pre_run_routes"] = flush_routes(
                prefix,
                names,
                args.table,
                args.cleanup_workers,
                args.flush_timeout,
                lambda data: progress_update("cleanup", {"pre_run_flush_progress": data}),
            )

        for idx, solution in enumerate(selected):
            print(f"[{now_iso()}] running {solution} on {args.nodes} nodes x {args.routes_per_node} routes", flush=True)
            run_solution(prefix, solution, indexed_names, args, summary, summary_path)
            if idx != len(selected) - 1 and not args.no_cleanup_between:
                key = f"after_{solution}_routes"
                summary["cleanup"][key] = flush_routes(
                    prefix,
                    names,
                    args.table,
                    args.cleanup_workers,
                    args.flush_timeout,
                    lambda data, key=key: progress_update("cleanup", {f"{key}_progress": data}),
                )

        if args.cleanup_after:
            summary["cleanup"]["post_run_routes"] = flush_routes(
                prefix,
                names,
                args.table,
                args.cleanup_workers,
                args.flush_timeout,
                lambda data: progress_update("cleanup", {"post_run_flush_progress": data}),
            )
        if args.cleanup_netns:
            summary["cleanup"]["netns"] = cleanup_netns(
                prefix,
                names,
                args.cleanup_workers,
                args.netns_timeout,
                lambda data: progress_update("cleanup", {"netns_delete_progress": data}),
            )

        summary["success"] = all(summary["solutions"][s]["success"] for s in selected)
    except Exception as exc:
        summary["success"] = False
        summary["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
    finally:
        summary["ended_at"] = now_iso()
        write_json_atomic(summary_path, summary)

    print(f"Summary: {summary_path}", flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
