#!/usr/bin/env python3
"""Run RTNL/FIB convergence experiments for SeedEMU router containers.

Scheme A is the real BIRD workflow: start BIRD with kernel export disabled,
wait for control-plane quiet time, then enable kernel export through the
legacy 0318 per-router docker-exec flow or the controlled batch flow, and
verify data-plane convergence. Schemes D/E/F/G are delegated to
rtnl_solution_lab.py as executable table-252 prototypes.
"""

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import random
import re
import shlex
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_PROJECT = "topology-291e63c1"
DEFAULT_BATCH_SIZE = 10
DEFAULT_KERNEL_JOBS = 1
DEFAULT_START_JOBS = 16
DEFAULT_SLEEP_SECONDS = 60
DEFAULT_POST_BIRD_LOAD_THRESHOLD = 40.0
DEFAULT_SUCCESS_LOAD_THRESHOLD = 50.0
DEFAULT_SUCCESS_STABLE_SECONDS = 120
DEFAULT_LOAD_GATE_TIMEOUT = 1800
DEFAULT_KERNEL_METHOD = "legacy0318"
DEFAULT_KERNEL_LOAD_THRESHOLD = 50.0
DEFAULT_KERNEL_PER_ROUTER_DELAY = 0.5
DEFAULT_SAMPLE_COUNT = 10
ROUTER_RE = re.compile(r"^as(\d+)brd-r")

KERNEL_EXPORT_NONE = """protocol kernel kernel1 {
    merge paths on;
    persist;
    scan time 60000;
    ipv4 {
        import none;
        export none;
    };
}
"""

KERNEL_EXPORT_ALL = """protocol kernel kernel1 {
    merge paths on;
    persist;
    scan time 60000;
    ipv4 {
        import none;
        export all;
    };
}
"""

LEGACY0318_KERNEL_EXPORT_NONE = """protocol kernel {
    merge paths on;
    persist;
    scan time 60000;
    ipv4 {
        import none;
        export none;
    };
}
"""

LEGACY0318_KERNEL_EXPORT_ALL = """protocol kernel {{
    merge paths on;
    persist;
    scan time {interval};
    ipv4 {{
        import none;
        export all;
    }};
}}
"""


class ExperimentError(RuntimeError):
    pass


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_cmd(cmd, *, input_text=None, timeout=120, check=True, cwd=None):
    proc = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        cwd=cwd,
    )
    if check and proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "command failed"
        raise ExperimentError(f"{shlex.join(cmd)}: {msg}")
    return proc


def docker_router_containers(project, limit=None):
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
        raise ExperimentError(f"no running containers found for compose project {project}")
    inspect = run_cmd(["docker", "inspect", "-f", "{{.Id}} {{.Name}} {{.State.Pid}} {{.State.Running}}"] + ids)
    routers = []
    for line in inspect.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[1].lstrip("/")
        match = ROUTER_RE.match(name)
        if not match or parts[3].lower() != "true":
            continue
        routers.append(
            {
                "id": parts[0],
                "short_id": parts[0][:12],
                "name": name,
                "as": int(match.group(1)),
                "pid": int(parts[2]) if parts[2].isdigit() else 0,
            }
        )
    routers.sort(key=lambda item: (item["as"], item["name"]))
    if limit:
        routers = routers[:limit]
    if not routers:
        raise ExperimentError(f"no running router containers matched project {project}")
    return routers


def order_by_as_round_robin(containers):
    groups = {}
    for item in containers:
        groups.setdefault(item["as"], []).append(item)
    ordered = []
    keys = sorted(groups)
    while keys:
        next_keys = []
        for key in keys:
            group = groups[key]
            ordered.append(group.pop(0))
            if group:
                next_keys.append(key)
        keys = next_keys
    return ordered


def docker_exec(container, shell_cmd, *, input_text=None, detach=False, timeout=120):
    cmd = ["docker", "exec"]
    if detach:
        cmd.append("-d")
    elif input_text is not None:
        cmd.append("-i")
    cmd += [container["short_id"], "sh", "-c", shell_cmd]
    return run_cmd(cmd, input_text=input_text, timeout=timeout, check=False)


def run_parallel(containers, func, workers, *, submit_delay=0.0, progress_every=200):
    results = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {}
        for item in containers:
            futures[pool.submit(func, item)] = item
            if submit_delay > 0:
                time.sleep(submit_delay)
        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            completed += 1
            item = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = {"success": False, "error": str(exc)}
            result.setdefault("name", item["name"])
            result.setdefault("as", item["as"])
            results.append(result)
            if progress_every and (completed % progress_every == 0 or completed == len(containers)):
                print(f"[{now_iso()}] completed {completed}/{len(containers)}", flush=True)
    ok_durations = [row.get("duration_seconds", 0) for row in results if row.get("success")]
    return {
        "items": len(containers),
        "successes": sum(1 for row in results if row.get("success")),
        "failures": sum(1 for row in results if not row.get("success")),
        "duration_seconds": round(time.monotonic() - started, 3),
        "task_duration_min": min(ok_durations) if ok_durations else None,
        "task_duration_avg": statistics.fmean(ok_durations) if ok_durations else None,
        "task_duration_max": max(ok_durations) if ok_durations else None,
        "errors": [row for row in results if not row.get("success")][:20],
    }


def load_sample():
    load1, load5, load15 = os.getloadavg()
    return {"time": now_iso(), "load1": load1, "load5": load5, "load15": load15}


def wait_load_below(threshold, stable_seconds, timeout, interval, summary_cb=None):
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
        if summary_cb:
            summary_cb(info)
        if stable_for >= stable_seconds:
            info["success"] = True
            return info
        if time.monotonic() - started > timeout:
            info["success"] = False
            return info
        time.sleep(interval)


def write_kernel_conf_task(content, configure=False, timeout=180):
    shell = "cat > /etc/bird/conf/kernel.conf && bird -p -c /etc/bird/bird.conf"
    if configure:
        shell += " && birdc configure"

    def task(container):
        started = time.monotonic()
        proc = docker_exec(container, shell, input_text=content, timeout=timeout)
        return {
            "name": container["name"],
            "as": container["as"],
            "success": proc.returncode == 0,
            "duration_seconds": round(time.monotonic() - started, 6),
            "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip()),
        }

    return task


def legacy0318_kernel_export_all():
    interval = 6000 + random.randint(0, 1200)
    return LEGACY0318_KERNEL_EXPORT_ALL.format(interval=interval), interval


def start_bird_task(command, timeout=60):
    shell = f"command -v bird >/dev/null 2>&1 && test -f /etc/bird/bird.conf && exec {command}"

    def task(container):
        started = time.monotonic()
        proc = docker_exec(container, shell, detach=True, timeout=timeout)
        return {
            "name": container["name"],
            "as": container["as"],
            "success": proc.returncode == 0,
            "duration_seconds": round(time.monotonic() - started, 6),
            "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip()),
        }

    return task


def kill_bird_task(timeout=60):
    def task(container):
        started = time.monotonic()
        proc = docker_exec(container, "pkill -x bird >/dev/null 2>&1 || true", timeout=timeout)
        return {
            "name": container["name"],
            "as": container["as"],
            "success": proc.returncode == 0,
            "duration_seconds": round(time.monotonic() - started, 6),
            "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip()),
        }

    return task


def parse_network_count(text):
    matches = re.findall(r"(\d+)\s+networks?", text)
    if matches:
        return int(matches[-1])
    matches = re.findall(r"network[s]?\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    if matches:
        return int(matches[-1])
    return None


def sample_different_as(containers, count):
    selected = []
    seen = set()
    for item in sorted(containers, key=lambda row: (row["as"], row["name"])):
        if item["as"] in seen:
            continue
        selected.append(item)
        seen.add(item["as"])
        if len(selected) >= count:
            break
    return selected


def verify_fib_rib(containers, sample_count, timeout):
    samples = []
    for item in sample_different_as(containers, sample_count):
        proc = docker_exec(
            item,
            "printf 'IP_ROUTE_LINES='; ip route | wc -l; birdc show route count",
            timeout=timeout,
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        ip_match = re.search(r"IP_ROUTE_LINES=(\d+)", text)
        ip_routes = int(ip_match.group(1)) if ip_match else None
        networks = parse_network_count(text)
        passed = proc.returncode == 0 and ip_routes is not None and networks is not None and ip_routes > networks
        samples.append(
            {
                "name": item["name"],
                "as": item["as"],
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
        "condition": "ip route | wc -l > birdc show route count networks",
    }


def run_legacy0318_kernel_export(ordered, args, result, summary, summary_path):
    groups = {}
    for item in sorted(ordered, key=lambda row: (row["as"], row["name"])):
        groups.setdefault(item["as"], []).append(item)

    started = time.monotonic()
    info = {
        "method": "legacy0318",
        "description": "start_bird_kernel0318.py style per-router docker exec update without birdc reload kernel",
        "config_template": "protocol kernel { import none; export all; }",
        "config_check_cmd": "bird -p -c /etc/bird/bird.conf",
        "config_apply_cmd": "birdc configure",
        "removed_cmd": "birdc reload kernel",
        "load_threshold": args.kernel_load_threshold,
        "load_check_interval": args.load_check_interval,
        "load_timeout": args.kernel_load_timeout,
        "per_router_delay": args.kernel_per_router_delay,
        "as_groups": {asn: len(items) for asn, items in sorted(groups.items())},
        "total": len(ordered),
        "processed": 0,
        "failed": 0,
        "skipped_no_kernel_conf": 0,
        "failed_containers": [],
        "last_container": None,
        "last_gate": None,
        "duration_seconds": 0.0,
    }
    result["stages"]["kernel_export"] = info
    summary["results"]["A"] = result
    write_json(summary_path, summary)

    for asn in sorted(groups):
        print(f"[{now_iso()}] legacy0318 kernel export AS {asn} ({len(groups[asn])} routers)", flush=True)
        for container in groups[asn]:
            info["last_container"] = container["name"]
            read_proc = docker_exec(
                container,
                "test -f /etc/bird/conf/kernel.conf && cat /etc/bird/conf/kernel.conf",
                timeout=args.command_timeout,
            )
            if read_proc.returncode != 0:
                info["skipped_no_kernel_conf"] += 1
                info["duration_seconds"] = round(time.monotonic() - started, 3)
                summary["results"]["A"] = result
                write_json(summary_path, summary)
                continue

            content, interval = legacy0318_kernel_export_all()
            shell = "cat > /etc/bird/conf/kernel.conf && bird -p -c /etc/bird/bird.conf && birdc configure"
            proc = docker_exec(container, shell, input_text=content, timeout=args.command_timeout)
            if proc.returncode == 0:
                info["processed"] += 1
                print(
                    f"[{now_iso()}] legacy0318 kernel export updated {info['processed']}/{info['total']} "
                    f"({container['name']}, scan_time={interval})",
                    flush=True,
                )
            else:
                info["failed"] += 1
                info["failed_containers"].append(
                    {
                        "name": container["name"],
                        "as": container["as"],
                        "stage": "configure",
                        "error": (proc.stderr.strip() or proc.stdout.strip())[-2000:],
                    }
                )

            info["duration_seconds"] = round(time.monotonic() - started, 3)
            summary["results"]["A"] = result
            write_json(summary_path, summary)

            gate = wait_load_below(
                args.kernel_load_threshold,
                0.0,
                args.kernel_load_timeout,
                args.load_check_interval,
            )
            info["last_gate"] = gate
            info["duration_seconds"] = round(time.monotonic() - started, 3)
            summary["results"]["A"] = result
            write_json(summary_path, summary)
            if not gate["success"]:
                raise ExperimentError(f"legacy0318 load gate failed after {container['name']}")

            if args.kernel_per_router_delay > 0:
                time.sleep(args.kernel_per_router_delay)

    info["duration_seconds"] = round(time.monotonic() - started, 3)
    summary["results"]["A"] = result
    write_json(summary_path, summary)
    if info["failed"] > 0:
        raise ExperimentError(f"legacy0318 kernel export failed for {info['failed']} containers")
    if info["processed"] == 0:
        raise ExperimentError("legacy0318 kernel export did not update any router containers")
    return info


def run_batch_kernel_export(ordered, args, result, summary, summary_path):
    batches = list(range(0, len(ordered), args.kernel_batch_size))
    kernel_started = time.monotonic()
    batch_results = []
    for batch_index, start in enumerate(batches, 1):
        batch = ordered[start : start + args.kernel_batch_size]
        print(f"[{now_iso()}] scheme A kernel batch {batch_index}/{len(batches)} size={len(batch)}", flush=True)
        if args.kernel_jitter_max > 0:
            random.shuffle(batch)
        batch_result = run_parallel(
            batch,
            write_kernel_conf_task(KERNEL_EXPORT_ALL, configure=True, timeout=args.command_timeout),
            args.kernel_jobs,
            submit_delay=args.kernel_submit_delay,
            progress_every=0,
        )
        batch_result["batch_index"] = batch_index
        batch_results.append(batch_result)
        result["stages"]["kernel_export"] = {
            "method": "batch",
            "batches_total": len(batches),
            "batches_completed": batch_index,
            "duration_seconds": round(time.monotonic() - kernel_started, 3),
            "batch_results": batch_results[-20:],
            "failures": sum(item["failures"] for item in batch_results),
        }
        summary["results"]["A"] = result
        write_json(summary_path, summary)
        if batch_result["failures"]:
            raise ExperimentError(f"kernel export batch {batch_index} failed")
        if batch_index < len(batches):
            gate = wait_load_below(
                args.kernel_load_threshold,
                args.kernel_load_stable_seconds,
                args.kernel_load_timeout,
                args.load_check_interval,
            )
            batch_result["post_batch_load_gate"] = gate
            if not gate["success"]:
                raise ExperimentError(f"load gate failed after kernel batch {batch_index}")
    result["stages"]["kernel_export"]["duration_seconds"] = round(time.monotonic() - kernel_started, 3)
    summary["results"]["A"] = result
    write_json(summary_path, summary)
    return result["stages"]["kernel_export"]


def run_scheme_a(args, summary, summary_path):
    containers = docker_router_containers(args.project, args.limit_containers)
    ordered = order_by_as_round_robin(containers)
    summary["containers"] = {
        "routers": len(containers),
        "first": containers[0]["name"] if containers else None,
        "limit": args.limit_containers,
    }
    write_json(summary_path, summary)

    result = {"description": "Scheme A: staged kernel export after BIRD control-plane quiet period.", "stages": {}}

    if args.force_restart_bird:
        print(f"[{now_iso()}] killing existing bird processes", flush=True)
        result["stages"]["kill_bird"] = run_parallel(ordered, kill_bird_task(args.command_timeout), args.start_jobs)
        summary["results"]["A"] = result
        write_json(summary_path, summary)

    initial_kernel_conf = LEGACY0318_KERNEL_EXPORT_NONE if args.kernel_method == "legacy0318" else KERNEL_EXPORT_NONE
    print(f"[{now_iso()}] writing kernel export none to {len(ordered)} routers", flush=True)
    result["stages"]["kernel_export_none"] = run_parallel(
        ordered,
        write_kernel_conf_task(initial_kernel_conf, configure=False, timeout=args.command_timeout),
        args.start_jobs,
    )
    summary["results"]["A"] = result
    write_json(summary_path, summary)
    if result["stages"]["kernel_export_none"]["failures"]:
        raise ExperimentError("kernel_export_none failed")

    print(f"[{now_iso()}] starting BIRD on {len(ordered)} routers", flush=True)
    result["stages"]["start_bird"] = run_parallel(
        ordered,
        start_bird_task(args.bird_start_command, timeout=args.command_timeout),
        args.start_jobs,
        submit_delay=args.start_bird_delay,
    )
    summary["results"]["A"] = result
    write_json(summary_path, summary)
    if result["stages"]["start_bird"]["failures"]:
        raise ExperimentError("start_bird failed")

    print(f"[{now_iso()}] sleeping {args.control_plane_sleep}s before kernel export", flush=True)
    sleep_started = time.monotonic()
    time.sleep(args.control_plane_sleep)
    result["stages"]["control_plane_sleep"] = {
        "duration_seconds": round(time.monotonic() - sleep_started, 3),
        "load_after_sleep": load_sample(),
    }
    summary["results"]["A"] = result
    write_json(summary_path, summary)

    print(f"[{now_iso()}] waiting for post-BIRD load < {args.post_bird_load_threshold}", flush=True)
    result["stages"]["post_bird_load_gate"] = wait_load_below(
        args.post_bird_load_threshold,
        args.post_bird_load_stable_seconds,
        args.post_bird_load_timeout,
        args.load_check_interval,
        lambda info: (result["stages"].__setitem__("post_bird_load_gate", info), summary["results"].__setitem__("A", result), write_json(summary_path, summary)),
    )
    if not result["stages"]["post_bird_load_gate"]["success"]:
        raise ExperimentError("post-BIRD load did not fall below threshold")

    if args.kernel_method == "legacy0318":
        run_legacy0318_kernel_export(ordered, args, result, summary, summary_path)
    else:
        run_batch_kernel_export(ordered, args, result, summary, summary_path)

    print(f"[{now_iso()}] waiting for final load < {args.success_load_threshold} for {args.success_stable_seconds}s", flush=True)
    result["stages"]["final_load_stable"] = wait_load_below(
        args.success_load_threshold,
        args.success_stable_seconds,
        args.success_load_timeout,
        args.load_check_interval,
        lambda info: (result["stages"].__setitem__("final_load_stable", info), summary["results"].__setitem__("A", result), write_json(summary_path, summary)),
    )
    if not result["stages"]["final_load_stable"]["success"]:
        raise ExperimentError("final load did not stay below success threshold")

    print(f"[{now_iso()}] verifying FIB/RIB samples", flush=True)
    result["stages"]["fib_rib_sample"] = verify_fib_rib(ordered, args.sample_count, args.command_timeout)
    result["success"] = result["stages"]["fib_rib_sample"]["success"]
    result["duration_seconds"] = round(
        result["stages"]["start_bird"]["duration_seconds"]
        + result["stages"]["control_plane_sleep"]["duration_seconds"]
        + result["stages"]["kernel_export"]["duration_seconds"],
        3,
    )
    summary["results"]["A"] = result
    write_json(summary_path, summary)
    if not result["success"]:
        raise ExperimentError("FIB/RIB sample convergence check failed")
    return result


def extract_prototype_duration(name, data):
    if name == "D":
        return data.get("controlled", {}).get("duration_seconds")
    if name == "E":
        return data.get("bulk_batch", {}).get("duration_seconds")
    if name == "F":
        return data.get("sharded_current_kernel", {}).get("duration_seconds")
    if name == "G":
        return data.get("transaction", {}).get("duration_seconds")
    return None


def run_prototype_solution(name, args, work_dir):
    script = Path(__file__).resolve().with_name("rtnl_solution_lab.py")
    if not script.is_file():
        raise ExperimentError(f"missing prototype script: {script}")
    subdir = work_dir / f"prototype_{name}"
    cmd = [
        sys.executable,
        str(script),
        "--project",
        args.project,
        "--solution",
        name,
        "--limit-containers",
        str(args.prototype_limit_containers),
        "--routes-per-netns",
        str(args.prototype_routes_per_netns),
        "--work-dir",
        str(subdir),
        "--d-workers",
        str(args.d_workers),
        "--d-baseline-workers",
        str(args.d_baseline_workers),
        "--f-shards",
        str(args.f_shards),
        "--g-shards",
        str(args.g_shards),
    ]
    if not args.prototype_cleanup_after:
        cmd.append("--no-cleanup-after")
    started = time.monotonic()
    proc = run_cmd(cmd, timeout=args.prototype_timeout, check=False)
    summary_path = subdir / "summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    return {
        "success": proc.returncode == 0 and bool(data and data.get("success")),
        "duration_seconds": round(time.monotonic() - started, 3),
        "command": cmd,
        "summary_path": str(summary_path),
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
        "summary": data,
    }


def build_comparison(summary):
    a = summary.get("results", {}).get("A", {})
    a_kernel = a.get("stages", {}).get("kernel_export", {}).get("duration_seconds")
    a_total = a.get("duration_seconds")
    out = {
        "note": "D/E/F/G are table-252 executable prototypes; speedups are indicative, not main-table BIRD convergence equivalence.",
        "scheme_a_kernel_export_seconds": a_kernel,
        "scheme_a_start_to_kernel_seconds": a_total,
        "prototype_speedups": {},
    }
    for name in ["D", "E", "F", "G"]:
        proto = summary.get("results", {}).get(name)
        if not proto or not proto.get("summary"):
            continue
        data = proto["summary"].get("results", {}).get(name, {})
        duration = extract_prototype_duration(name, data)
        out["prototype_speedups"][name] = {
            "prototype_duration_seconds": duration,
            "speedup_over_a_kernel_export": round(a_kernel / duration, 6) if a_kernel and duration else None,
            "speedup_over_a_start_to_kernel": round(a_total / duration, 6) if a_total and duration else None,
        }
    return out


def default_work_dir():
    return Path.home() / "bird" / "runs" / f"rtnl-experiment_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--solution", choices=["A", "D", "E", "F", "G", "all"], default="A")
    parser.add_argument("--work-dir", help="Default in VM/local mode: ~/bird/runs/rtnl-experiment_<timestamp>")
    parser.add_argument("--limit-containers", type=int, help="Limit scheme A to first N router containers.")
    parser.add_argument("--force-restart-bird", action="store_true")
    parser.add_argument("--bird-start-command", default="bird")
    parser.add_argument("--start-jobs", type=int, default=DEFAULT_START_JOBS)
    parser.add_argument("--start-bird-delay", type=float, default=0.02)
    parser.add_argument("--control-plane-sleep", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--post-bird-load-threshold", type=float, default=DEFAULT_POST_BIRD_LOAD_THRESHOLD)
    parser.add_argument("--post-bird-load-stable-seconds", type=float, default=1.0)
    parser.add_argument("--post-bird-load-timeout", type=float, default=DEFAULT_LOAD_GATE_TIMEOUT)
    parser.add_argument(
        "--kernel-method",
        choices=["legacy0318", "batch"],
        default=DEFAULT_KERNEL_METHOD,
        help="legacy0318 uses the start_bird_kernel0318.py-style per-router docker exec flow; batch keeps the previous grouped flow.",
    )
    parser.add_argument("--kernel-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--kernel-jobs", type=int, default=DEFAULT_KERNEL_JOBS)
    parser.add_argument("--kernel-submit-delay", type=float, default=0.0)
    parser.add_argument("--kernel-jitter-max", type=float, default=0.0)
    parser.add_argument("--kernel-load-threshold", type=float, default=DEFAULT_KERNEL_LOAD_THRESHOLD)
    parser.add_argument("--kernel-load-stable-seconds", type=float, default=10.0)
    parser.add_argument("--kernel-load-timeout", type=float, default=DEFAULT_LOAD_GATE_TIMEOUT)
    parser.add_argument("--kernel-per-router-delay", type=float, default=DEFAULT_KERNEL_PER_ROUTER_DELAY)
    parser.add_argument("--success-load-threshold", type=float, default=DEFAULT_SUCCESS_LOAD_THRESHOLD)
    parser.add_argument("--success-stable-seconds", type=float, default=DEFAULT_SUCCESS_STABLE_SECONDS)
    parser.add_argument("--success-load-timeout", type=float, default=7200)
    parser.add_argument("--load-check-interval", type=float, default=5.0)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--command-timeout", type=float, default=180)
    parser.add_argument("--prototype-limit-containers", type=int, default=128)
    parser.add_argument("--prototype-routes-per-netns", type=int, default=8)
    parser.add_argument("--prototype-timeout", type=float, default=3600)
    parser.add_argument("--prototype-cleanup-after", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--d-workers", type=int, default=1)
    parser.add_argument("--d-baseline-workers", type=int, default=0)
    parser.add_argument("--f-shards", type=int, default=8)
    parser.add_argument("--g-shards", type=int, default=1)
    parser.add_argument("--vm", help="Run this experiment on a VM over SSH.")
    parser.add_argument("--vm-user", default="lxl")
    parser.add_argument("--vm-runner", help="Default: /home/<vm-user>/bird/rtnl_experiment.py")
    return parser.parse_args()


def ssh_base(user, host):
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-o",
        "ConnectTimeout=10",
        f"{user}@{host}",
    ]


def rsync_base(user, host):
    ssh_cmd = "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    return ["rsync", "-a", "--delete", "-e", ssh_cmd]


def run_streamed(cmd, check=True):
    print(f"$ {shlex.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True)
    if check and proc.returncode != 0:
        raise ExperimentError(f"Command failed with exit code {proc.returncode}: {shlex.join(cmd)}")
    return proc.returncode


def remote_args(args, remote_runner, remote_work_dir):
    cmd = [str(remote_runner)]
    for key, value in vars(args).items():
        if key in {"vm", "vm_user", "vm_runner", "work_dir"}:
            continue
        opt = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(opt)
        elif value is not None:
            cmd += [opt, str(value)]
    cmd += ["--work-dir", str(remote_work_dir)]
    return cmd


def run_on_vm(args):
    vm_host = args.vm
    vm_user = args.vm_user
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    local_dir = Path(__file__).resolve().parent / "runs" / f"rtnl-experiment_{stamp}"
    remote_dir = Path(f"/home/{vm_user}/bird/runs/rtnl-experiment_{stamp}")
    stable_runner = Path(args.vm_runner or f"/home/{vm_user}/bird/rtnl_experiment.py")
    remote_runner = remote_dir / stable_runner.name
    lab_script = Path(__file__).resolve().with_name("rtnl_solution_lab.py")
    stable_lab = stable_runner.with_name("rtnl_solution_lab.py")
    remote_lab = remote_dir / "rtnl_solution_lab.py"

    run_streamed(ssh_base(vm_user, vm_host) + ["mkdir", "-p", str(remote_dir), str(stable_runner.parent)])
    for src, dst in [
        (Path(__file__).resolve(), stable_runner),
        (Path(__file__).resolve(), remote_runner),
        (lab_script, stable_lab),
        (lab_script, remote_lab),
    ]:
        run_streamed(rsync_base(vm_user, vm_host) + [str(src), f"{vm_user}@{vm_host}:{dst}"])
    run_streamed(ssh_base(vm_user, vm_host) + ["chmod", "+x", str(stable_runner), str(remote_runner), str(stable_lab), str(remote_lab)])
    cmd = " ".join(shlex.quote(part) for part in remote_args(args, remote_runner, remote_dir))
    rc = run_streamed(ssh_base(vm_user, vm_host) + [cmd], check=False)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    run_streamed(rsync_base(vm_user, vm_host) + [f"{vm_user}@{vm_host}:{remote_dir}/", f"{local_dir}/"], check=False)
    print(f"Local experiment dir: {local_dir}", flush=True)
    return rc


def run_local(args):
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
        "results": {},
        "comparison": {},
    }
    write_json(summary_path, summary)
    try:
        names = ["A", "D", "E", "F", "G"] if args.solution == "all" else [args.solution]
        if "A" in names:
            run_scheme_a(args, summary, summary_path)
        for name in [item for item in names if item != "A"]:
            print(f"[{now_iso()}] running prototype solution {name}", flush=True)
            summary["results"][name] = run_prototype_solution(name, args, work_dir)
            write_json(summary_path, summary)
            if not summary["results"][name]["success"]:
                raise ExperimentError(f"prototype solution {name} failed")
        summary["comparison"] = build_comparison(summary)
        summary["success"] = True
    except Exception as exc:
        summary["success"] = False
        summary["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr)
    finally:
        summary["ended_at"] = now_iso()
        write_json(summary_path, summary)
    print(f"Summary: {summary_path}", flush=True)
    return 0 if summary["success"] else 1


def main():
    args = parse_args()
    if args.vm:
        return run_on_vm(args)
    return run_local(args)


if __name__ == "__main__":
    sys.exit(main())
