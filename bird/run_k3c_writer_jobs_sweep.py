#!/usr/bin/env python3
"""Measure K3c cold full-FIB install time across writer counts.

The topology and BIRD control plane must already be running. Before every
measured write, the script removes the metric-specific routes with a fixed
number of cleanup workers. This keeps every measured writer run on the same
empty target route set and separates BIRD dump time from kernel write time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
SSH_BASE = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
]


def parse_jobs(value: str) -> list[int]:
    jobs = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not jobs or any(item <= 0 for item in jobs):
        raise argparse.ArgumentTypeError("jobs must be a comma-separated list of positive integers")
    return jobs


def quote(parts: list[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def ssh_target(args: argparse.Namespace) -> str:
    return f"{args.vm_user}@{args.vm}"


def ssh_cmd(args: argparse.Namespace, remote: str) -> list[str]:
    return ["ssh", *SSH_BASE, "-o", f"ConnectTimeout={args.ssh_connect_timeout}", ssh_target(args), remote]


def run(cmd: list[str], *, timeout: float | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    print("$", shlex.join(cmd), flush=True)
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        timeout=timeout,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def remote_output(args: argparse.Namespace, command: str, timeout: float = 60) -> str:
    return (run(ssh_cmd(args, command), timeout=timeout, capture=True).stdout or "").strip()


def remote_load1(args: argparse.Namespace) -> float:
    return float(remote_output(args, "awk '{print $1}' /proc/loadavg", timeout=30).splitlines()[-1])


def wait_load(args: argparse.Namespace, stage: str) -> dict:
    started = time.monotonic()
    samples = []
    while True:
        load1 = remote_load1(args)
        samples.append({"elapsed_seconds": round(time.monotonic() - started, 3), "load1": load1})
        print(f"{stage}: load1={load1:.2f}", flush=True)
        if load1 < args.load_threshold:
            return {
                "duration_seconds": round(time.monotonic() - started, 3),
                "final_load1": load1,
                "samples": samples,
            }
        if time.monotonic() - started >= args.load_timeout:
            raise TimeoutError(f"{stage}: load1 did not fall below {args.load_threshold}")
        time.sleep(args.load_poll_interval)


def parse_gnu_time(path: Path) -> dict:
    result: dict[str, float | int | str] = {}
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("Percent of CPU this job got:"):
            result["cpu_percent"] = float(line.rsplit(":", 1)[1].strip().rstrip("%"))
        elif line.startswith("Maximum resident set size (kbytes):"):
            result["max_rss_kb"] = int(line.rsplit(":", 1)[1].strip())
        elif line.startswith("User time (seconds):"):
            result["user_seconds"] = float(line.rsplit(":", 1)[1].strip())
        elif line.startswith("System time (seconds):"):
            result["system_seconds"] = float(line.rsplit(":", 1)[1].strip())
    return result


def fetch(args: argparse.Namespace, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run([
        "rsync", "-a", "-e",
        "ssh " + " ".join(shlex.quote(item) for item in SSH_BASE),
        f"{ssh_target(args)}:{remote_path}", str(local_path),
    ], timeout=300)


def run_writer(
    args: argparse.Namespace,
    *,
    action: str,
    jobs: int,
    repeat: int,
    tag: str,
    local_run_dir: Path,
    remote_run_dir: str,
    remote_asset_dir: str,
) -> dict:
    local_stage = local_run_dir / "runs" / tag
    local_stage.mkdir(parents=True, exist_ok=True)
    remote_stage = f"{remote_run_dir}/runs/{tag}"
    remote_summary = f"{remote_stage}/summary.json"
    remote_details = f"{remote_stage}/details.json"
    remote_time = f"{remote_stage}/time.txt"
    remote_work = f"{remote_stage}/work"
    local_log = local_stage / "writer.log"
    local_summary = local_stage / "summary.json"
    local_time = local_stage / "time.txt"

    run(ssh_cmd(args, f"mkdir -p {shlex.quote(remote_stage)}"), timeout=60)
    writer = [
        "sudo", "-n", "python3", f"{remote_asset_dir}/kernel_bulk_birdc_writer.py",
        "--project-name", args.project_name,
        "--metric", args.metric,
        "--protocol", args.protocol,
        "--table", args.table,
        "--route-action", action,
        "--multipath-policy", "first",
        "--jobs", jobs,
        "--dump-jobs", args.dump_jobs,
        "--batch", args.batch,
        "--ops-per-attr", args.ops_per_attr,
        "--group-fib-info",
        "--command-timeout", args.command_timeout,
        "--work-dir", remote_work,
        "--summary-path", remote_summary,
        "--details-path", remote_details,
    ]
    remote = quote(["/usr/bin/time", "-v", "-o", remote_time, *writer])
    started = time.monotonic()
    with local_log.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(
            ssh_cmd(args, remote),
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            timeout=args.writer_timeout,
            check=False,
        )
    elapsed = time.monotonic() - started
    fetch(args, remote_summary, local_summary)
    fetch(args, remote_time, local_time)
    summary = json.loads(local_summary.read_text(encoding="utf-8"))
    result = {
        "tag": tag,
        "action": action,
        "jobs": jobs,
        "repeat": repeat,
        "returncode": proc.returncode,
        "host_elapsed_seconds": round(elapsed, 6),
        "writer": summary,
        "resource": parse_gnu_time(local_time),
        "local_log": str(local_log),
        "local_summary": str(local_summary),
    }
    result["success"] = bool(
        proc.returncode == 0
        and summary.get("success")
        and int(summary.get("fail_count") or 0) == 0
        and int(summary.get("write_failures") or 0) == 0
        and int(summary.get("dump_failures") or 0) == 0
    )
    if not result["success"]:
        raise RuntimeError(f"{tag} failed; see {local_log}")
    return result


def p95(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20, method="inclusive")[18]


def aggregate(results: list[dict]) -> list[dict]:
    rows = []
    for jobs in sorted({int(item["jobs"]) for item in results}):
        subset = [item for item in results if int(item["jobs"]) == jobs]
        writes = [float(item["writer"]["write_wall_seconds"]) for item in subset]
        totals = [float(item["writer"]["wall_seconds"]) for item in subset]
        dumps = [float(item["writer"]["dump_wall_seconds"]) for item in subset]
        throughput = [
            float(item["writer"]["success_count"]) / float(item["writer"]["write_wall_seconds"])
            for item in subset
        ]
        cpu = [float(item["resource"]["cpu_percent"]) for item in subset if "cpu_percent" in item["resource"]]
        rows.append({
            "jobs": jobs,
            "repeats": len(subset),
            "write_seconds_median": statistics.median(writes),
            "write_seconds_p95": p95(writes),
            "write_seconds_min": min(writes),
            "write_seconds_max": max(writes),
            "dump_seconds_median": statistics.median(dumps),
            "total_seconds_median": statistics.median(totals),
            "ops_per_second_median": statistics.median(throughput),
            "cpu_percent_median": statistics.median(cpu) if cpu else None,
            "fail_count": sum(int(item["writer"].get("fail_count") or 0) for item in subset),
        })
    return rows


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vm", default="192.168.122.60")
    parser.add_argument("--vm-user", default="lxl")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--jobs", type=parse_jobs, default=parse_jobs("1,2,4,8,12,16,24,32"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cleanup-jobs", type=int, default=16)
    parser.add_argument("--dump-jobs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=16000)
    parser.add_argument("--ops-per-attr", type=int, default=2000)
    parser.add_argument("--metric", type=int, default=232)
    parser.add_argument("--protocol", type=int, default=12)
    parser.add_argument("--table", type=int, default=254)
    parser.add_argument("--command-timeout", type=float, default=900)
    parser.add_argument("--writer-timeout", type=float, default=1800)
    parser.add_argument("--load-threshold", type=float, default=50)
    parser.add_argument("--load-timeout", type=float, default=7200)
    parser.add_argument("--load-poll-interval", type=float, default=10)
    parser.add_argument("--ssh-connect-timeout", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    local_run_dir = (args.run_dir or SCRIPT_DIR / "runs" / f"k3c_writer_jobs_sweep_3083_{timestamp}").resolve()
    remote_run_dir = f"/home/{args.vm_user}/bird/runs/{local_run_dir.name}"
    remote_asset_dir = f"{remote_run_dir}/assets"
    summary_path = local_run_dir / "summary.json"
    local_run_dir.mkdir(parents=True, exist_ok=True)

    running = int(remote_output(
        args,
        "docker ps --filter " + shlex.quote(f"label=com.docker.compose.project={args.project_name}") + " -q | wc -l",
    ))
    if running < 100:
        raise RuntimeError(f"project {args.project_name} has only {running} running containers")
    run(ssh_cmd(args, f"mkdir -p {shlex.quote(remote_asset_dir)}"), timeout=60)
    for name in ("kernel_bulk_writer.py", "kernel_bulk_birdc_writer.py"):
        run([
            "rsync", "-a", "-e", "ssh " + " ".join(shlex.quote(item) for item in SSH_BASE),
            str(SCRIPT_DIR / "kernel" / "tools" / name),
            f"{ssh_target(args)}:{remote_asset_dir}/{name}",
        ], timeout=300)

    payload = {
        "mode": "k3c-cold-full-writer-jobs-sweep",
        "started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": sys.argv,
        "project_name": args.project_name,
        "vm": args.vm,
        "settings": vars(args) | {"run_dir": str(local_run_dir)},
        "running_containers_before": running,
        "results": [],
        "success": None,
    }
    write_summary(summary_path, payload)

    jobs_orders = []
    for repeat in range(1, args.repeats + 1):
        offset = (repeat - 1) % len(args.jobs)
        jobs_orders.append(args.jobs[offset:] + args.jobs[:offset])

    for repeat, order in enumerate(jobs_orders, start=1):
        for jobs in order:
            wait_load(args, f"repeat={repeat} jobs={jobs} pre-clean")
            cleanup = run_writer(
                args,
                action="del",
                jobs=args.cleanup_jobs,
                repeat=repeat,
                tag=f"r{repeat}_j{jobs}_cleanup",
                local_run_dir=local_run_dir,
                remote_run_dir=remote_run_dir,
                remote_asset_dir=remote_asset_dir,
            )
            wait_load(args, f"repeat={repeat} jobs={jobs} pre-write")
            measured = run_writer(
                args,
                action="replace",
                jobs=jobs,
                repeat=repeat,
                tag=f"r{repeat}_j{jobs}_write",
                local_run_dir=local_run_dir,
                remote_run_dir=remote_run_dir,
                remote_asset_dir=remote_asset_dir,
            )
            measured["cleanup"] = {
                "jobs": cleanup["jobs"],
                "write_wall_seconds": cleanup["writer"].get("write_wall_seconds"),
                "success_count": cleanup["writer"].get("success_count"),
                "fail_count": cleanup["writer"].get("fail_count"),
            }
            payload["results"].append(measured)
            payload["aggregate"] = aggregate(payload["results"])
            write_summary(summary_path, payload)

    payload["ended_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    payload["success"] = all(item.get("success") for item in payload["results"])
    payload["aggregate"] = aggregate(payload["results"])
    write_summary(summary_path, payload)
    print(json.dumps(payload["aggregate"], indent=2), flush=True)
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
