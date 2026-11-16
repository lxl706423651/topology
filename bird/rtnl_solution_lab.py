#!/usr/bin/env python3
"""Run executable RTNL solution experiments inside the BIRD/RTNL VM.

The workloads use Docker labels to find SeedEMU router containers, enter each
container network namespace with nsenter, and write synthetic blackhole routes
into an isolated routing table. This exercises Linux FIB/RTNL paths without
touching the default routing table used by the emulation.

With --real-bird, the script instead starts BIRD with kernel export disabled,
reads selected IPv4 routes from BIRD's real RIB, and writes those routes to the
default FIB through controlled ip -batch writers.
"""

import argparse
import concurrent.futures
import datetime as dt
import ipaddress
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


DEFAULT_PROJECT = "topology-291e63c1"
DEFAULT_ROUTE_TABLE = 252
DEFAULT_PREFIX_BASE = "198.18.0.0"
ROUTER_RE = re.compile(r"^as\d+brd-r")
AS_RE = re.compile(r"^as(\d+)brd-r")
IPV4_PREFIX_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){3}/\d{1,2})\s+unicast\b.*\*")
VIA_RE = re.compile(r"^\s+via\s+(\d{1,3}(?:\.\d{1,3}){3})\s+on\s+(\S+)")
DEV_RE = re.compile(r"^\s+dev\s+(\S+)")

KERNEL_EXPORT_NONE = """protocol kernel {
    merge paths on;
    persist;
    scan time 60000;
    ipv4 {
        import none;
        export none;
    };
}
"""


class LabError(RuntimeError):
    pass


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def run_cmd(cmd, *, input_text=None, timeout=120, check=True):
    proc = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "command failed"
        raise LabError(f"{' '.join(cmd)}: {msg}")
    return proc


def nsenter_prefix():
    if os.geteuid() == 0:
        return ["nsenter"]
    if shutil.which("sudo"):
        check = subprocess.run(["sudo", "-n", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if check.returncode == 0:
            return ["sudo", "-n", "nsenter"]
    raise LabError("root or passwordless sudo is required for nsenter")


def docker_router_containers(project, limit):
    ps = run_cmd(
        [
            "docker",
            "ps",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.ID}}",
        ]
    )
    ids = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    if not ids:
        raise LabError(f"no running containers found for compose project {project}")

    inspect = run_cmd(
        ["docker", "inspect", "-f", "{{.Name}} {{.State.Pid}} {{.State.Running}}"] + ids
    )
    routers = []
    for line in inspect.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[0].lstrip("/")
        try:
            pid = int(parts[1])
        except ValueError:
            pid = 0
        if parts[2].lower() == "true" and pid > 0 and ROUTER_RE.match(name):
            as_match = AS_RE.match(name)
            routers.append(
                {
                    "name": name,
                    "pid": pid,
                    "as": int(as_match.group(1)) if as_match else None,
                }
            )
    routers.sort(key=lambda item: item["name"])
    if limit:
        routers = routers[:limit]
    if not routers:
        raise LabError(f"no running router containers matched project {project}")
    return routers


def route_prefix(prefix_base, container_index, route_index, routes_per_netns):
    base = int(ipaddress.IPv4Address(prefix_base))
    addr = ipaddress.IPv4Address(base + container_index * routes_per_netns + route_index + 1)
    return f"{addr}/32"


def route_batch(prefix_base, container_index, route_count, table):
    lines = []
    for idx in range(route_count):
        prefix = route_prefix(prefix_base, container_index, idx, route_count)
        lines.append(f"route replace blackhole {prefix} table {table}")
    return "\n".join(lines) + "\n"


def nsenter_batch(prefix, pid, batch, timeout):
    cmd = prefix + ["-t", str(pid), "-n", "-m", "--", "ip", "-force", "-batch", "-"]
    return run_cmd(cmd, input_text=batch, timeout=timeout, check=False)


def nsenter_input(prefix, pid, shell_cmd, input_text, timeout):
    cmd = prefix + ["-t", str(pid), "-n", "-m", "--", "sh", "-c", shell_cmd]
    return run_cmd(cmd, input_text=input_text, timeout=timeout, check=False)


def nsenter_shell(prefix, pid, shell_cmd, timeout):
    cmd = prefix + ["-t", str(pid), "-n", "-m", "--", "sh", "-c", shell_cmd]
    return run_cmd(cmd, timeout=timeout, check=False)


def cleanup_table(prefix, containers, table, timeout):
    errors = []
    for item in containers:
        proc = nsenter_shell(prefix, item["pid"], f"ip route flush table {table} >/dev/null 2>&1 || true", timeout)
        if proc.returncode != 0:
            errors.append({"name": item["name"], "error": proc.stderr.strip() or proc.stdout.strip()})
    return errors


def cleanup_main_proto_bird(prefix, containers, timeout, workers):
    def task(item):
        proc = nsenter_shell(prefix, item["pid"], "ip route flush proto bird >/dev/null 2>&1 || true", timeout)
        return {
            "name": item["name"],
            "success": proc.returncode == 0,
            "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip()),
        }

    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for row in pool.map(task, containers):
            if not row["success"]:
                errors.append(row)
    return errors[:20]


def load_sample():
    load1, load5, load15 = os.getloadavg()
    return {"time": now_iso(), "load1": load1, "load5": load5, "load15": load15}


def wait_load_below(threshold, stable_seconds, timeout, interval):
    started = time.monotonic()
    stable_started = None
    samples = []
    while True:
        sample = load_sample()
        samples.append(sample)
        if sample["load1"] < threshold:
            if stable_started is None:
                stable_started = time.monotonic()
            stable_for = time.monotonic() - stable_started
        else:
            stable_started = None
            stable_for = 0.0
        info = {
            "threshold": threshold,
            "stable_seconds_required": stable_seconds,
            "stable_for_seconds": round(stable_for, 3),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "latest": sample,
            "samples": samples[-60:],
        }
        if stable_for >= stable_seconds:
            info["success"] = True
            return info
        if time.monotonic() - started > timeout:
            info["success"] = False
            return info
        time.sleep(interval)


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
            try:
                load1, load5, load15 = os.getloadavg()
                self.samples.append({"t": time.time(), "load1": load1, "load5": load5, "load15": load15})
            except OSError:
                pass
            self._stop.wait(self.interval)

    def stats(self):
        if not self.samples:
            return {"samples": 0}
        vals = [item["load1"] for item in self.samples]
        return {
            "samples": len(vals),
            "load1_min": min(vals),
            "load1_avg": statistics.fmean(vals),
            "load1_max": max(vals),
        }


def task_summary(results):
    ok = [item["duration_seconds"] for item in results if item["success"]]
    failed = [item for item in results if not item["success"]]
    out = {
        "tasks": len(results),
        "successes": len(ok),
        "failures": len(failed),
        "errors": failed[:20],
    }
    if ok:
        out.update(
            {
                "task_duration_min": min(ok),
                "task_duration_avg": statistics.fmean(ok),
                "task_duration_max": max(ok),
            }
        )
    return out


def parse_network_count(text):
    matches = re.findall(r"(\d+)\s+networks?", text)
    if matches:
        return int(matches[-1])
    matches = re.findall(r"network[s]?\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    if matches:
        return int(matches[-1])
    return None


def parse_bird_primary_routes(text):
    routes = []
    current = None
    for line in text.splitlines():
        route_match = IPV4_PREFIX_RE.match(line)
        if route_match:
            if current and current["nexthops"]:
                routes.append(current)
            current = {"prefix": route_match.group(1), "nexthops": [], "raw": [line]}
            continue
        if current is None:
            continue
        current["raw"].append(line)
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


def route_to_ip_batch_line(route):
    prefix = route["prefix"]
    nexthops = route["nexthops"]
    if not nexthops:
        return None
    if len(nexthops) == 1:
        hop = nexthops[0]
        if hop.get("via"):
            return f"route replace {prefix} via {hop['via']} dev {hop['dev']} proto bird metric 32"
        return f"route replace {prefix} dev {hop['dev']} proto bird metric 32"
    parts = [f"route replace {prefix} proto bird metric 32"]
    for hop in nexthops:
        if hop.get("via"):
            parts.append(f"nexthop via {hop['via']} dev {hop['dev']}")
        else:
            parts.append(f"nexthop dev {hop['dev']}")
    return " ".join(parts)


def real_bird_route_batch(prefix, container, args):
    dump_started = time.monotonic()
    proc = nsenter_shell(prefix, container["pid"], "birdc show route primary", args.timeout)
    dump_duration = time.monotonic() - dump_started
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        return {
            "name": container["name"],
            "pid": container["pid"],
            "success": False,
            "stage": "bird_dump",
            "duration_seconds": round(dump_duration, 6),
            "dump_duration_seconds": round(dump_duration, 6),
            "error": text[-2000:],
        }
    routes = parse_bird_primary_routes(text)
    lines = [line for route in routes for line in [route_to_ip_batch_line(route)] if line]
    return {
        "name": container["name"],
        "pid": container["pid"],
        "success": True,
        "routes": len(routes),
        "batch_lines": len(lines),
        "duration_seconds": round(dump_duration, 6),
        "dump_duration_seconds": round(dump_duration, 6),
        "batch": "\n".join(lines) + ("\n" if lines else ""),
    }


def real_bird_write_batch(prefix, batch_info, args):
    started = time.monotonic()
    if not batch_info.get("success"):
        return batch_info
    if not batch_info.get("batch"):
        out = dict(batch_info)
        out.update({"success": True, "write_duration_seconds": 0.0, "duration_seconds": batch_info.get("dump_duration_seconds", 0.0)})
        out.pop("batch", None)
        return out
    proc = nsenter_batch(prefix, batch_info["pid"], batch_info["batch"], args.timeout)
    out = dict(batch_info)
    out["write_duration_seconds"] = round(time.monotonic() - started, 6)
    out["duration_seconds"] = round(out.get("dump_duration_seconds", 0.0) + out["write_duration_seconds"], 6)
    out["success"] = proc.returncode == 0
    out["error"] = "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip())[-2000:]
    out.pop("batch", None)
    return out


def real_bird_dump_and_write_task(prefix, container, args):
    return real_bird_write_batch(prefix, real_bird_route_batch(prefix, container, args), args)


def prepare_real_bird_task(prefix, container, args):
    started = time.monotonic()
    stop = nsenter_shell(
        prefix,
        container["pid"],
        "birdc down >/dev/null 2>&1 || true; pkill -x bird >/dev/null 2>&1 || true; sleep 0.2; rm -f /run/bird/bird.pid /run/bird/bird.ctl",
        args.timeout,
    )
    conf = nsenter_input(
        prefix,
        container["pid"],
        "cat > /etc/bird/conf/kernel.conf && bird -p -c /etc/bird/bird.conf",
        KERNEL_EXPORT_NONE,
        args.timeout,
    )
    if conf.returncode == 0:
        start = nsenter_shell(prefix, container["pid"], "bird", args.timeout)
    else:
        start = conf
    success = stop.returncode == 0 and conf.returncode == 0 and start.returncode == 0
    return {
        "name": container["name"],
        "pid": container["pid"],
        "success": success,
        "duration_seconds": round(time.monotonic() - started, 6),
        "error": "" if success else ((conf.stderr or conf.stdout or start.stderr or start.stdout or "").strip())[-2000:],
    }


def run_real_parallel(label, prefix, containers, args, workers, *, sort_by_pid=False):
    ordered = sorted(containers, key=lambda item: item["pid"] if sort_by_pid else item["name"])
    start = time.monotonic()
    results = []
    with LoadMonitor(args.load_sample_interval) as monitor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {}
            for item in ordered:
                submitted = time.monotonic()
                futures[pool.submit(real_bird_dump_and_write_task, prefix, item, args)] = (item, submitted)
            for fut in concurrent.futures.as_completed(futures):
                item, submitted = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {"name": item["name"], "pid": item["pid"], "success": False, "error": str(exc)}
                result["duration_seconds"] = round(time.monotonic() - submitted, 6)
                results.append(result)
    duration = time.monotonic() - start
    route_ops = sum(item.get("batch_lines", 0) for item in results if item.get("success"))
    return {
        "label": label,
        "mode": "real-bird-main-fib",
        "workers": workers,
        "containers": len(ordered),
        "route_ops": route_ops,
        "duration_seconds": round(duration, 6),
        "route_ops_per_second": round(route_ops / duration, 3) if duration > 0 else None,
        "load": monitor.stats(),
        "task_summary": task_summary(results),
        "results": results,
    }


def run_real_transaction(label, prefix, containers, args, write_workers, *, sort_by_pid=True):
    ordered = sorted(containers, key=lambda item: item["pid"] if sort_by_pid else item["name"])
    start = time.monotonic()
    dump_results = []
    write_results = []
    with LoadMonitor(args.load_sample_interval) as monitor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.real_dump_workers)) as pool:
            futures = {pool.submit(real_bird_route_batch, prefix, item, args): item for item in ordered}
            for fut in concurrent.futures.as_completed(futures):
                item = futures[fut]
                try:
                    dump_results.append(fut.result())
                except Exception as exc:
                    dump_results.append({"name": item["name"], "pid": item["pid"], "success": False, "error": str(exc)})
        dump_results.sort(key=lambda item: item.get("pid", 0))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, write_workers)) as pool:
            futures = {pool.submit(real_bird_write_batch, prefix, item, args): item for item in dump_results}
            for fut in concurrent.futures.as_completed(futures):
                item = futures[fut]
                try:
                    write_results.append(fut.result())
                except Exception as exc:
                    write_results.append({"name": item.get("name"), "pid": item.get("pid"), "success": False, "error": str(exc)})
    duration = time.monotonic() - start
    route_ops = sum(item.get("batch_lines", 0) for item in write_results if item.get("success"))
    return {
        "label": label,
        "mode": "real-bird-main-fib",
        "dump_workers": args.real_dump_workers,
        "write_workers": write_workers,
        "containers": len(ordered),
        "route_ops": route_ops,
        "duration_seconds": round(duration, 6),
        "route_ops_per_second": round(route_ops / duration, 3) if duration > 0 else None,
        "load": monitor.stats(),
        "dump_summary": task_summary(dump_results),
        "task_summary": task_summary(write_results),
        "results": write_results,
    }


def sample_different_names(containers, count):
    selected = []
    seen_asns = set()
    ordered = sorted(containers, key=lambda item: (item.get("as") is None, item.get("as") or 0, item["name"]))
    for item in ordered:
        asn = item.get("as")
        if asn is not None and asn in seen_asns:
            continue
        selected.append(item)
        if asn is not None:
            seen_asns.add(asn)
        if len(selected) >= count:
            return selected

    selected_names = {item["name"] for item in selected}
    for item in sorted(containers, key=lambda item: item["name"]):
        if item["name"] in selected_names:
            continue
        selected.append(item)
        if len(selected) >= count:
            break
    return selected


def sample_bird_network_counts(prefix, containers, args):
    samples = []
    count = args.rib_sample_count or args.sample_count
    for item in sample_different_names(containers, count):
        proc = nsenter_shell(prefix, item["pid"], "birdc show route primary count", args.timeout)
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        networks = parse_network_count(text)
        samples.append(
            {
                "name": item["name"],
                "pid": item["pid"],
                "as": item.get("as"),
                "success": proc.returncode == 0 and networks is not None,
                "bird_networks": networks,
                "returncode": proc.returncode,
                "raw_tail": text[-1000:],
            }
        )
    return {
        "time": now_iso(),
        "success": bool(samples) and all(row["success"] for row in samples),
        "counts": [row["bird_networks"] for row in samples],
        "total_networks": sum(row["bird_networks"] or 0 for row in samples),
        "samples": samples,
    }


def wait_rib_stable(prefix, containers, args):
    started = time.monotonic()
    stable_started = None
    last_counts = None
    history = []
    while True:
        sample = sample_bird_network_counts(prefix, containers, args)
        sample["load"] = load_sample()
        history.append(sample)
        counts = tuple(sample["counts"]) if sample["success"] else None
        if counts is not None and counts == last_counts:
            if stable_started is None:
                stable_started = time.monotonic()
            stable_for = time.monotonic() - stable_started
        else:
            stable_started = None
            stable_for = 0.0
            last_counts = counts

        info = {
            "success": False,
            "threshold_seconds": args.rib_stable_seconds,
            "stable_for_seconds": round(stable_for, 3),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "sample_count": args.rib_sample_count or args.sample_count,
            "latest": sample,
            "history": history[-60:],
        }
        if counts is not None and stable_for >= args.rib_stable_seconds:
            info["success"] = True
            return info
        if time.monotonic() - started > args.rib_stable_timeout:
            return info
        time.sleep(args.rib_check_interval)


def verify_real_fib_rib(prefix, containers, args):
    samples = []
    for item in sample_different_names(containers, args.sample_count):
        proc = nsenter_shell(
            prefix,
            item["pid"],
            "printf 'IP_ROUTE_LINES='; ip route | wc -l; birdc show route count",
            args.timeout,
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        ip_match = re.search(r"IP_ROUTE_LINES=(\d+)", text)
        ip_routes = int(ip_match.group(1)) if ip_match else None
        networks = parse_network_count(text)
        passed = (
            proc.returncode == 0
            and ip_routes is not None
            and networks is not None
            and ip_routes >= networks + args.fib_rib_min_delta
        )
        samples.append(
            {
                "name": item["name"],
                "pid": item["pid"],
                "as": item.get("as"),
                "success": passed,
                "ip_route_lines": ip_routes,
                "bird_networks": networks,
                "returncode": proc.returncode,
                "raw_tail": text[-1000:],
            }
        )
    return {
        "sample_count": len(samples),
        "successes": sum(1 for row in samples if row["success"]),
        "failures": sum(1 for row in samples if not row["success"]),
        "samples": samples,
        "success": bool(samples) and all(row["success"] for row in samples),
        "condition": f"ip route | wc -l >= birdc show route count networks + {args.fib_rib_min_delta}",
    }


def real_solution_d(prefix, containers, args):
    cleanup = cleanup_main_proto_bird(prefix, containers, args.timeout, args.cleanup_workers) if args.pre_cleanup else []
    controlled = run_real_parallel("D-real-centralized-bounded-host-writer", prefix, containers, args, args.d_workers)
    return {"description": "Real BIRD RIB to main FIB through a bounded centralized host writer.", "controlled": controlled, "pre_cleanup_errors": cleanup}


def real_solution_e(prefix, containers, args):
    cleanup = cleanup_main_proto_bird(prefix, containers, args.timeout, args.cleanup_workers) if args.pre_cleanup else []
    bulk = run_real_parallel("E-real-per-router-ip-batch-writer", prefix, containers, args, args.e_workers)
    return {"description": "Real BIRD RIB to main FIB using one ip -batch submission per router.", "bulk_batch": bulk, "pre_cleanup_errors": cleanup}


def real_solution_f(prefix, containers, args):
    cleanup = cleanup_main_proto_bird(prefix, containers, args.timeout, args.cleanup_workers) if args.pre_cleanup else []
    sharded = run_real_parallel("F-real-sharded-netns-writers", prefix, containers, args, args.f_shards)
    return {"description": "Real BIRD RIB to main FIB with sharded namespace writers.", "sharded_current_kernel": sharded, "pre_cleanup_errors": cleanup}


def real_solution_g(prefix, containers, args):
    cleanup = cleanup_main_proto_bird(prefix, containers, args.timeout, args.cleanup_workers) if args.pre_cleanup else []
    transaction = run_real_transaction("G-real-ordered-dump-then-write-transaction", prefix, containers, args, args.g_shards)
    return {
        "description": "Real BIRD RIB to main FIB by dumping route batches first, sorting by target netns PID, then applying an ordered transaction plan.",
        "netns_lock_order": "ascending target container init PID",
        "transaction": transaction,
        "pre_cleanup_errors": cleanup,
    }


def validate_real_solution(prefix, containers, args, result):
    result["final_load_gate"] = wait_load_below(
        args.success_load_threshold,
        args.success_stable_seconds,
        args.success_load_timeout,
        args.load_sample_interval,
    )
    result["fib_rib_sample"] = verify_real_fib_rib(prefix, containers, args)
    result["success"] = bool(result["final_load_gate"].get("success")) and bool(result["fib_rib_sample"].get("success"))
    return result


def prepare_real_bird(prefix, containers, args):
    start = time.monotonic()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.prepare_workers)) as pool:
        futures = {pool.submit(prepare_real_bird_task, prefix, item, args): item for item in containers}
        for fut in concurrent.futures.as_completed(futures):
            item = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({"name": item["name"], "pid": item["pid"], "success": False, "error": str(exc)})
    time.sleep(args.control_plane_sleep)
    return {
        "duration_seconds": round(time.monotonic() - start, 3),
        "task_summary": task_summary(results),
        "results": results,
        "control_plane_sleep": args.control_plane_sleep,
        "post_bird_load_gate": wait_load_below(
            args.post_bird_load_threshold,
            args.post_bird_load_stable_seconds,
            args.post_bird_load_timeout,
            args.load_sample_interval,
        ),
    }


def batched_workload(label, prefix, containers, args, workers, *, sort_by_pid=False):
    ordered = sorted(containers, key=lambda item: item["pid"] if sort_by_pid else item["name"])
    start = time.monotonic()
    results = []
    with LoadMonitor(args.load_sample_interval) as monitor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {}
            for cidx, item in enumerate(ordered):
                batch = route_batch(args.prefix_base, cidx, args.routes_per_netns, args.table)
                submitted = time.monotonic()
                fut = pool.submit(nsenter_batch, prefix, item["pid"], batch, args.timeout)
                futures[fut] = (item, submitted)
            for fut in concurrent.futures.as_completed(futures):
                item, submitted = futures[fut]
                try:
                    proc = fut.result()
                    success = proc.returncode == 0
                    error = proc.stderr.strip() or proc.stdout.strip()
                except Exception as exc:
                    success = False
                    error = str(exc)
                results.append(
                    {
                        "name": item["name"],
                        "pid": item["pid"],
                        "success": success,
                        "duration_seconds": round(time.monotonic() - submitted, 6),
                        "error": "" if success else error,
                    }
                )
    duration = time.monotonic() - start
    route_ops = len(ordered) * args.routes_per_netns
    return {
        "label": label,
        "workers": workers,
        "containers": len(ordered),
        "routes_per_netns": args.routes_per_netns,
        "route_ops": route_ops,
        "duration_seconds": round(duration, 6),
        "route_ops_per_second": round(route_ops / duration, 3) if duration > 0 else None,
        "load": monitor.stats(),
        "task_summary": task_summary(results),
        "results": results,
    }


def per_route_workload(label, prefix, container, args):
    start = time.monotonic()
    results = []
    with LoadMonitor(args.load_sample_interval) as monitor:
        for ridx in range(args.routes_per_netns):
            route = route_prefix(args.prefix_base, 0, ridx, args.routes_per_netns)
            submitted = time.monotonic()
            proc = nsenter_shell(prefix, container["pid"], f"ip route replace blackhole {route} table {args.table}", args.timeout)
            results.append(
                {
                    "name": f"route-{ridx}",
                    "success": proc.returncode == 0,
                    "duration_seconds": round(time.monotonic() - submitted, 6),
                    "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip()),
                }
            )
    duration = time.monotonic() - start
    return {
        "label": label,
        "container": container["name"],
        "routes": args.routes_per_netns,
        "duration_seconds": round(duration, 6),
        "route_ops_per_second": round(args.routes_per_netns / duration, 3) if duration > 0 else None,
        "load": monitor.stats(),
        "task_summary": task_summary(results),
    }


def solution_d(prefix, containers, args):
    cleanup_table(prefix, containers, args.table, args.timeout)
    baseline_workers = args.d_baseline_workers or len(containers)
    baseline = batched_workload("D-baseline-many-concurrent-writers", prefix, containers, args, baseline_workers)
    cleanup_table(prefix, containers, args.table, args.timeout)
    controlled = batched_workload("D-centralized-bounded-host-writer", prefix, containers, args, args.d_workers)
    return {
        "description": "Host centralized FIB writer: compare many concurrent namespace writers with a bounded host-controlled writer.",
        "baseline": baseline,
        "controlled": controlled,
        "cleanup_errors": cleanup_table(prefix, containers, args.table, args.timeout) if args.cleanup_after else [],
    }


def solution_e(prefix, containers, args):
    one = containers[0]
    cleanup_table(prefix, [one], args.table, args.timeout)
    per_route = per_route_workload("E-per-route-current-api-analogue", prefix, one, args)
    cleanup_table(prefix, [one], args.table, args.timeout)
    bulk = batched_workload("E-single-netns-bulk-batch-analogue", prefix, [one], args, 1)
    speedup = per_route["duration_seconds"] / bulk["duration_seconds"] if bulk["duration_seconds"] else None
    return {
        "description": "Single-netns bulk API analogue: compare one route per command with one ip -batch submission.",
        "per_route": per_route,
        "bulk_batch": bulk,
        "speedup_bulk_over_per_route": round(speedup, 6) if speedup is not None else None,
        "cleanup_errors": cleanup_table(prefix, [one], args.table, args.timeout) if args.cleanup_after else [],
    }


def solution_f(prefix, containers, args):
    cleanup_table(prefix, containers, args.table, args.timeout)
    serial = batched_workload("F-current-global-rtnl-serial-writer", prefix, containers, args, 1)
    cleanup_table(prefix, containers, args.table, args.timeout)
    sharded = batched_workload("F-sharded-per-netns-rtnl-proxy", prefix, containers, args, args.f_shards)

    serial_ok = [item["duration_seconds"] for item in serial["results"] if item["success"]]
    shard_sums = [0.0 for _ in range(max(1, args.f_shards))]
    for idx, value in enumerate(serial_ok):
        shard_sums[idx % len(shard_sums)] += value
    ideal = max(shard_sums) if shard_sums else None
    return {
        "description": "Per-netns/sharded RTNL experiment: execute sharded namespace writers and estimate ideal route-only per-netns RTNL wall time.",
        "serial_current_kernel": serial,
        "sharded_current_kernel": sharded,
        "ideal_per_netns_rtnl_seconds": round(ideal, 6) if ideal is not None else None,
        "actual_over_ideal_ratio": round(sharded["duration_seconds"] / ideal, 6) if ideal else None,
        "cleanup_errors": cleanup_table(prefix, containers, args.table, args.timeout) if args.cleanup_after else [],
    }


def solution_g(prefix, containers, args):
    cleanup_table(prefix, containers, args.table, args.timeout)
    transaction = batched_workload(
        "G-multi-netns-sorted-transaction-analogue",
        prefix,
        containers,
        args,
        args.g_shards,
        sort_by_pid=True,
    )
    return {
        "description": "Multi-netns bulk writer analogue: sort target netns by PID and execute an ordered route-batch transaction plan.",
        "netns_lock_order": "ascending target container init PID",
        "transaction": transaction,
        "cleanup_errors": cleanup_table(prefix, containers, args.table, args.timeout) if args.cleanup_after else [],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run D/E/F/G RTNL solution experiments in the BIRD/RTNL VM.")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--solution", choices=["D", "E", "F", "G", "all"], default="all")
    parser.add_argument("--limit-containers", type=int, default=0)
    parser.add_argument("--routes-per-netns", type=int, default=32)
    parser.add_argument("--table", type=int, default=DEFAULT_ROUTE_TABLE)
    parser.add_argument("--prefix-base", default=DEFAULT_PREFIX_BASE)
    parser.add_argument("--work-dir", help="Default: ~/bird/runs/rtnl-solutions_<timestamp>")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--load-sample-interval", type=float, default=0.2)
    parser.add_argument("--d-workers", type=int, default=1)
    parser.add_argument("--d-baseline-workers", type=int, default=0)
    parser.add_argument("--e-workers", type=int, default=1)
    parser.add_argument("--f-shards", type=int, default=8)
    parser.add_argument("--g-shards", type=int, default=1)
    parser.add_argument("--cleanup-after", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--real-bird", action="store_true", help="Use real BIRD RIB routes and write them to the default FIB instead of synthetic table routes.")
    parser.add_argument("--real-prepare-bird", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-workers", type=int, default=16)
    parser.add_argument("--cleanup-workers", type=int, default=16)
    parser.add_argument("--real-dump-workers", type=int, default=16)
    parser.add_argument("--pre-cleanup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control-plane-sleep", type=float, default=60)
    parser.add_argument("--post-bird-load-threshold", type=float, default=40)
    parser.add_argument("--post-bird-load-stable-seconds", type=float, default=1)
    parser.add_argument("--post-bird-load-timeout", type=float, default=1800)
    parser.add_argument("--pre-write-rib-stable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rib-stable-seconds", type=float, default=120)
    parser.add_argument("--rib-stable-timeout", type=float, default=1800)
    parser.add_argument("--rib-check-interval", type=float, default=10)
    parser.add_argument("--rib-sample-count", type=int, default=10)
    parser.add_argument("--success-load-threshold", type=float, default=50)
    parser.add_argument("--success-stable-seconds", type=float, default=120)
    parser.add_argument("--success-load-timeout", type=float, default=7200)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--fib-rib-min-delta", type=int, default=1)
    return parser.parse_args()


def write_summary(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(path)


def default_work_dir():
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.home() / "bird" / "runs" / f"rtnl-solutions_{stamp}"


def main():
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser() if args.work_dir else default_work_dir()
    summary_path = work_dir / "summary.json"
    summary = {
        "schema_version": 1,
        "started_at": now_iso(),
        "ended_at": None,
        "success": None,
        "error": None,
        "host": socket.gethostname(),
        "command": sys.argv,
        "project": args.project,
        "solution": args.solution,
        "settings": vars(args),
        "work_dir": str(work_dir),
        "containers": [],
        "results": {},
    }
    write_summary(summary_path, summary)

    try:
        prefix = nsenter_prefix()
        containers = docker_router_containers(args.project, args.limit_containers)
        summary["containers"] = containers
        write_summary(summary_path, summary)

        if args.real_bird:
            if args.cleanup_only:
                summary["results"]["cleanup"] = {
                    "cleanup_errors": cleanup_main_proto_bird(prefix, containers, args.timeout, args.cleanup_workers),
                    "containers": len(containers),
                    "mode": "real-bird-main-fib",
                }
            else:
                if args.real_prepare_bird:
                    print(f"[{now_iso()}] preparing BIRD export-none on {len(containers)} containers", flush=True)
                    summary["results"]["prepare_real_bird"] = prepare_real_bird(prefix, containers, args)
                    write_summary(summary_path, summary)
                    prep = summary["results"]["prepare_real_bird"]
                    if prep["task_summary"]["failures"]:
                        raise LabError("real BIRD preparation failed")
                    if not prep["post_bird_load_gate"].get("success"):
                        raise LabError("post-BIRD load gate failed during real BIRD preparation")

                if args.pre_write_rib_stable:
                    print(f"[{now_iso()}] waiting for sampled BIRD RIB counts to stabilize", flush=True)
                    summary["results"]["pre_write_rib_stability"] = wait_rib_stable(prefix, containers, args)
                    write_summary(summary_path, summary)
                    if not summary["results"]["pre_write_rib_stability"].get("success"):
                        raise LabError("pre-write BIRD RIB stability gate failed")

                runners = {
                    "D": real_solution_d,
                    "E": real_solution_e,
                    "F": real_solution_f,
                    "G": real_solution_g,
                }
                names = ["D", "E", "F", "G"] if args.solution == "all" else [args.solution]
                for name in names:
                    print(f"[{now_iso()}] running real solution {name} on {len(containers)} containers", flush=True)
                    result = runners[name](prefix, containers, args)
                    summary["results"][name] = validate_real_solution(prefix, containers, args, result)
                    write_summary(summary_path, summary)
                    if not summary["results"][name]["success"]:
                        raise LabError(f"real solution {name} failed validation")

                if args.cleanup_after:
                    summary["results"]["final_cleanup"] = {
                        "cleanup_errors": cleanup_main_proto_bird(prefix, containers, args.timeout, args.cleanup_workers),
                        "containers": len(containers),
                        "mode": "real-bird-main-fib",
                    }
        elif args.cleanup_only:
            summary["results"]["cleanup"] = {
                "cleanup_errors": cleanup_table(prefix, containers, args.table, args.timeout),
                "containers": len(containers),
            }
        else:
            runners = {"D": solution_d, "E": solution_e, "F": solution_f, "G": solution_g}
            names = ["D", "E", "F", "G"] if args.solution == "all" else [args.solution]
            for name in names:
                print(f"[{now_iso()}] running solution {name} on {len(containers)} containers", flush=True)
                summary["results"][name] = runners[name](prefix, containers, args)
                write_summary(summary_path, summary)

        summary["success"] = True
    except Exception as exc:
        summary["success"] = False
        summary["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr)
    finally:
        summary["ended_at"] = now_iso()
        write_summary(summary_path, summary)

    print(f"Summary: {summary_path}", flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
