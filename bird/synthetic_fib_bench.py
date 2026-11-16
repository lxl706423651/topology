#!/usr/bin/env python3
"""Synthetic FIB route insertion benchmark for many router namespaces.

This script creates temporary Linux network namespaces, writes synthetic IPv4
/32 routes into each namespace's main FIB using one of the configured writers,
records timing and load samples, then optionally deletes the namespaces to clean
all routes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run_cmd(cmd: list[str], *, timeout: float | None = None, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    if check and proc.returncode != 0:
        output = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"{' '.join(cmd)} failed rc={proc.returncode}: {output[-1000:]}")
    return proc


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def ns_name(prefix: str, index: int) -> str:
    return f"{prefix}{index:03d}"


def list_bench_namespaces(prefix: str) -> list[str]:
    proc = run_cmd(["ip", "netns", "list"], check=False)
    names = []
    for line in proc.stdout.splitlines():
        name = line.split()[0] if line.split() else ""
        if name.startswith(prefix):
            names.append(name)
    return sorted(names)


def cleanup_namespaces(prefix: str) -> dict:
    started = time.monotonic()
    names = list_bench_namespaces(prefix)
    deleted = 0
    failures = []
    for name in names:
        proc = run_cmd(["ip", "netns", "delete", name], check=False)
        if proc.returncode == 0:
            deleted += 1
        else:
            failures.append({"name": name, "stderr": (proc.stderr or proc.stdout or "").strip()[-500:]})
    return {
        "requested_prefix": prefix,
        "found": len(names),
        "deleted": deleted,
        "failures": failures,
        "seconds": time.monotonic() - started,
    }


def setup_namespaces(prefix: str, routers: int) -> dict:
    started = time.monotonic()
    run_cmd(["mkdir", "-p", "/run/netns"], check=False)
    created = 0
    reused = 0
    failures = []
    existing = set(list_bench_namespaces(prefix))
    for index in range(routers):
        name = ns_name(prefix, index)
        if name in existing:
            reused += 1
        else:
            proc = run_cmd(["ip", "netns", "add", name], check=False)
            if proc.returncode != 0:
                failures.append({"name": name, "stderr": (proc.stderr or proc.stdout or "").strip()[-500:]})
                continue
            created += 1
        proc = run_cmd(["ip", "-n", name, "link", "set", "lo", "up"], check=False)
        if proc.returncode != 0:
            failures.append({"name": name, "stderr": (proc.stderr or proc.stdout or "").strip()[-500:]})
    if failures:
        raise RuntimeError(f"namespace setup failed: {failures[:3]}")
    return {"created": created, "reused": reused, "seconds": time.monotonic() - started}


def prefix_for(index: int, base_a: int, base_b: int) -> str:
    # 10.<base_b + high>.<mid>.<low>/32 gives each namespace the same synthetic
    # route set; namespaces are independent, so prefixes do not need to differ.
    high = index >> 16
    mid = (index >> 8) & 0xFF
    low = index & 0xFF
    return f"{base_a}.{base_b + high}.{mid}.{low}/32"


def write_routes_one(args: tuple[str, int, int, int, int, int]) -> dict:
    name, routes_per_router, proto, metric, base_a, base_b = args
    started = time.monotonic()
    proc = subprocess.Popen(
        ["ip", "-n", name, "-4", "-force", "-batch", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    written = 0
    error = None
    try:
        assert proc.stdin is not None
        for route_index in range(routes_per_router):
            prefix = prefix_for(route_index, base_a, base_b)
            proc.stdin.write(f"route replace {prefix} dev lo proto {proto} metric {metric}\n")
            written += 1
        proc.stdin.close()
        stdout = proc.stdout.read() if proc.stdout else ""
        stderr = proc.stderr.read() if proc.stderr else ""
        returncode = proc.wait()
    except Exception as exc:
        error = repr(exc)
        try:
            proc.kill()
        except Exception:
            pass
        stdout = ""
        stderr = ""
        returncode = proc.wait()
    ended = time.monotonic()
    return {
        "name": name,
        "routes_requested": routes_per_router,
        "routes_written_to_stdin": written,
        "returncode": returncode,
        "success": returncode == 0 and error is None,
        "seconds": ended - started,
        "routes_per_second": written / (ended - started) if ended > started else None,
        "error": error,
        "stdout_tail": stdout[-1000:],
        "stderr_tail": stderr[-2000:],
    }


def write_routes_one_rtnl(args: tuple[str, int, int, int, int, int, str, int, str]) -> dict:
    name, routes_per_router, proto, metric, base_a, base_b, writer, batch, ack_mode = args
    started = time.monotonic()
    base = f"{base_a}.{base_b}.0.0"
    proc = run_cmd(
        [
            "python3",
            writer,
            "--netns-path",
            f"/run/netns/{name}",
            "--synthetic-routes",
            str(routes_per_router),
            "--synthetic-dev",
            "lo",
            "--synthetic-base",
            base,
            "--metric",
            str(metric),
            "--protocol",
            str(proto),
            "--batch",
            str(batch),
            "--ack-mode",
            ack_mode,
        ],
        timeout=None,
        check=False,
    )
    ended = time.monotonic()
    try:
        summary = json.loads(proc.stdout)
    except Exception:
        summary = {}
    success_count = int(summary.get("success_count") or 0)
    return {
        "name": name,
        "routes_requested": routes_per_router,
        "routes_written_to_stdin": success_count,
        "returncode": proc.returncode,
        "success": proc.returncode == 0 and summary.get("success") is True,
        "seconds": ended - started,
        "routes_per_second": success_count / (ended - started) if ended > started else None,
        "error": None if proc.returncode == 0 else (proc.stderr or proc.stdout or "")[-2000:],
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-2000:],
        "writer_summary": summary,
    }


def write_routes_one_bulk(args: tuple[str, int, int, int, int, int, str, int, int, str, str]) -> dict:
    name, routes_per_router, proto, metric, base_a, base_b, writer, batch, ops_per_attr, route_type, op = args
    started = time.monotonic()
    base = f"{base_a}.{base_b}.0.0"
    proc = run_cmd(
        [
            "python3",
            writer,
            "--netns-path",
            f"/run/netns/{name}",
            "--routes",
            str(routes_per_router),
            "--route-type",
            route_type,
            "--op",
            op,
            "--dev",
            "lo",
            "--base",
            base,
            "--metric",
            str(metric),
            "--protocol",
            str(proto),
            "--batch",
            str(batch),
            "--ops-per-attr",
            str(ops_per_attr),
        ],
        timeout=None,
        check=False,
    )
    ended = time.monotonic()
    try:
        summary = json.loads(proc.stdout)
    except Exception:
        summary = {}
    success_count = int(summary.get("success_count") or 0)
    fail_count = int(summary.get("fail_count") or 0)
    return {
        "name": name,
        "routes_requested": routes_per_router,
        "routes_written_to_stdin": success_count,
        "returncode": proc.returncode,
        "success": proc.returncode == 0 and success_count == routes_per_router and fail_count == 0,
        "seconds": ended - started,
        "routes_per_second": success_count / (ended - started) if ended > started else None,
        "error": None if proc.returncode == 0 else (proc.stderr or proc.stdout or "")[-2000:],
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-2000:],
        "writer_summary": summary,
    }


def read_cpu() -> tuple[int, int]:
    parts = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    vals = [int(item) for item in parts[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return sum(vals), idle


def sample_system(prev_cpu: tuple[int, int] | None) -> tuple[dict, tuple[int, int]]:
    load_parts = Path("/proc/loadavg").read_text(encoding="utf-8").split()
    cpu = read_cpu()
    sample = {
        "at": now_iso(),
        "load1": float(load_parts[0]),
        "load5": float(load_parts[1]),
        "load15": float(load_parts[2]),
    }
    if prev_cpu:
        total_delta = cpu[0] - prev_cpu[0]
        idle_delta = cpu[1] - prev_cpu[1]
        if total_delta > 0:
            sample["cpu_util_percent"] = round(100.0 * (1.0 - idle_delta / total_delta), 2)
    return sample, cpu


def run_benchmark(args: argparse.Namespace) -> dict:
    if os.geteuid() != 0:
        raise SystemExit("synthetic_fib_bench.py must run as root")
    summary = {
        "mode": "synthetic-fib-bench",
        "started_at": now_iso(),
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "host": os.uname().nodename,
        "success": False,
    }
    summary_path = args.summary_path
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    interrupted = False

    def handle_signal(signum, _frame):
        nonlocal interrupted
        interrupted = True
        raise KeyboardInterrupt(f"signal {signum}")

    old_int = signal.signal(signal.SIGINT, handle_signal)
    old_term = signal.signal(signal.SIGTERM, handle_signal)
    try:
        if args.cleanup_before:
            summary["cleanup_before"] = cleanup_namespaces(args.prefix)
        if args.cleanup_only:
            summary["cleanup_only"] = cleanup_namespaces(args.prefix)
            summary["success"] = not summary["cleanup_only"]["failures"]
            return summary

        summary["setup"] = setup_namespaces(args.prefix, args.routers)
        names = [ns_name(args.prefix, index) for index in range(args.routers)]
        if args.writer == "rtnl":
            writer = str(args.rtnl_writer.expanduser().resolve())
            work = [
                (name, args.routes_per_router, args.proto, args.metric, args.base_a, args.base_b, writer, args.rtnl_batch, args.rtnl_ack_mode)
                for name in names
            ]
            worker_fn = write_routes_one_rtnl
        elif args.writer == "bulk":
            writer = str(args.bulk_writer.expanduser().resolve())
            work = [
                (
                    name,
                    args.routes_per_router,
                    args.proto,
                    args.metric,
                    args.base_a,
                    args.base_b,
                    writer,
                    args.bulk_batch,
                    args.bulk_ops_per_attr,
                    args.bulk_route_type,
                    args.bulk_op,
                )
                for name in names
            ]
            worker_fn = write_routes_one_bulk
        else:
            work = [(name, args.routes_per_router, args.proto, args.metric, args.base_a, args.base_b) for name in names]
            worker_fn = write_routes_one

        started = time.monotonic()
        samples = []
        rows = []
        prev_cpu = None
        next_sample = 0.0
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(worker_fn, item): item[0] for item in work}
            while futures:
                if interrupted:
                    raise KeyboardInterrupt("interrupted")
                now = time.monotonic()
                if now >= next_sample:
                    sample, prev_cpu = sample_system(prev_cpu)
                    sample["completed_routers"] = len(rows)
                    sample["remaining_routers"] = len(futures)
                    samples.append(sample)
                    next_sample = now + args.sample_interval
                done = [future for future in futures if future.done()]
                if not done:
                    time.sleep(0.5)
                    continue
                for future in done:
                    name = futures.pop(future)
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {"name": name, "success": False, "error": repr(exc)}
                    rows.append(row)
                    print(
                        f"{now_iso()} completed {len(rows)}/{args.routers} {name} "
                        f"success={row.get('success')} seconds={row.get('seconds')}",
                        flush=True,
                    )

        wall = time.monotonic() - started
        ok = [row for row in rows if row.get("success")]
        failed = [row for row in rows if not row.get("success")]
        summary["write"] = {
            "wall_seconds": wall,
            "routers": args.routers,
            "routes_per_router": args.routes_per_router,
            "total_routes_requested": args.routers * args.routes_per_router,
            "successful_routers": len(ok),
            "failed_routers": len(failed),
            "total_routes_written_to_stdin": sum(int(row.get("routes_written_to_stdin") or 0) for row in rows),
            "aggregate_routes_per_second": (args.routers * args.routes_per_router) / wall if wall > 0 else None,
            "per_router_seconds": {
                "min": min((row.get("seconds") for row in ok), default=None),
                "max": max((row.get("seconds") for row in ok), default=None),
                "avg": (sum(float(row.get("seconds") or 0) for row in ok) / len(ok)) if ok else None,
            },
            "failed_samples": failed[:10],
        }
        summary["samples"] = samples
        summary["rows"] = rows if args.include_rows else rows[:20]
        summary["success"] = not failed
    except KeyboardInterrupt as exc:
        summary["success"] = False
        summary["interrupted"] = repr(exc)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        if args.cleanup_after:
            summary["cleanup_after"] = cleanup_namespaces(args.prefix)
        summary["ended_at"] = now_iso()
        if summary_path:
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routers", type=int, default=100)
    parser.add_argument("--routes-per-router", type=int, default=600_000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--prefix", default="fibbench-r")
    parser.add_argument("--proto", type=int, default=232)
    parser.add_argument("--metric", type=int, default=232)
    parser.add_argument("--writer", choices=["ip", "rtnl", "bulk"], default="ip")
    parser.add_argument("--rtnl-writer", type=Path, default=Path("bird/kernel/tools/rtnetlink_route_writer.py"))
    parser.add_argument("--rtnl-batch", type=int, default=512)
    parser.add_argument("--rtnl-ack-mode", choices=["each", "none"], default="each")
    parser.add_argument("--bulk-writer", type=Path, default=Path("bird/kernel/tools/kernel_bulk_writer.py"))
    parser.add_argument("--bulk-batch", type=int, default=16_000)
    parser.add_argument("--bulk-ops-per-attr", type=int, default=2_000)
    parser.add_argument("--bulk-route-type", choices=["dev", "blackhole"], default="dev")
    parser.add_argument("--bulk-op", choices=["replace", "add", "del"], default="replace")
    parser.add_argument("--base-a", type=int, default=10)
    parser.add_argument("--base-b", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=15.0)
    parser.add_argument("--summary-path", type=Path, default=Path("/tmp/synthetic_fib_bench_summary.json"))
    parser.add_argument("--cleanup-before", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cleanup-after", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--include-rows", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_benchmark(args)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, ensure_ascii=False))
    return 0 if summary.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
