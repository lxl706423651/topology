#!/usr/bin/env python3
"""Sweep post-fault full FIB writer jobs on the BIRD/RTNL VM.

This is intentionally separate from run_strict_delta_full_experiment.py. It sets
up only the full arm, dumps full birdc-primary commands once after a victim
failure, then replays the same command files with different writer job counts
while sampling VM load and CPU utilization.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

from run_strict_delta_full_experiment import (
    DEFAULT_DOMAIN,
    DEFAULT_USER,
    DEFAULT_VM,
    REPO_DIR,
    SCRIPT_DIR,
    Summary,
    choose_victims,
    compose_service_count,
    ensure_output_dir,
    fetch_remote_file,
    jsonable_settings,
    q,
    reset_vm,
    rsync_cmd,
    run,
    run_host_writer,
    run_setup,
    ssh,
    ssh_cmd,
    stop_victims,
    wait_load_below,
)


def parse_jobs(value: str) -> list[int]:
    jobs = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        item = int(part)
        if item <= 0:
            raise argparse.ArgumentTypeError("jobs must be positive")
        jobs.append(item)
    if not jobs:
        raise argparse.ArgumentTypeError("empty jobs list")
    return jobs


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def remote_quote_cmd(parts: list[str | Path | int | float]) -> str:
    return " ".join(q(str(part)) for part in parts)


def parse_proc_stat(line: str) -> tuple[int, int]:
    parts = line.split()
    if not parts or parts[0] != "cpu":
        return 0, 0
    vals = [int(item) for item in parts[1:]]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    total = sum(vals)
    return total, idle


def sample_vm(args: argparse.Namespace) -> dict:
    script = r"""
python3 - <<'PY'
import json
import os
import subprocess

def read_first(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.readline().strip()

load_parts = read_first("/proc/loadavg").split()
cpu_line = read_first("/proc/stat")
mem = {}
with open("/proc/meminfo", "r", encoding="utf-8") as fh:
    for line in fh:
        if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:")):
            key, value, *_ = line.split()
            mem[key.rstrip(":")] = int(value)
try:
    writer_count = int(subprocess.check_output(["pgrep", "-fc", "delta_fib_host_writer.py"], text=True).strip())
except Exception:
    writer_count = 0
try:
    ip_batch_count = int(subprocess.check_output(["pgrep", "-fc", "ip .*batch|-batch"], text=True).strip())
except Exception:
    ip_batch_count = 0
print(json.dumps({
    "load1": float(load_parts[0]),
    "load5": float(load_parts[1]),
    "load15": float(load_parts[2]),
    "cpu_line": cpu_line,
    "nproc": os.cpu_count(),
    "mem": mem,
    "writer_processes": writer_count,
    "ip_batch_processes": ip_batch_count,
}))
PY
"""
    started = time.monotonic()
    try:
        proc = ssh(args, script, capture=True, timeout=args.sample_timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        output = ""
        if exc.stdout:
            output = exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode(errors="replace")
        return {
            "at": now_iso(),
            "success": False,
            "ssh_seconds": round(elapsed, 3),
            "returncode": "timeout",
            "output": output[-1000:],
        }
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        return {
            "at": now_iso(),
            "success": False,
            "ssh_seconds": round(elapsed, 3),
            "returncode": proc.returncode,
            "output": (proc.stdout or "")[-1000:],
        }
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {
            "at": now_iso(),
            "success": False,
            "ssh_seconds": round(elapsed, 3),
            "returncode": proc.returncode,
            "output": (proc.stdout or "")[-1000:],
        }
    data["at"] = now_iso()
    data["success"] = True
    data["ssh_seconds"] = round(elapsed, 3)
    total, idle = parse_proc_stat(data.get("cpu_line", ""))
    data["cpu_total"] = total
    data["cpu_idle"] = idle
    return data


def add_cpu_util(samples: list[dict]) -> None:
    prev = None
    for sample in samples:
        if not sample.get("success"):
            continue
        if prev:
            total_delta = sample.get("cpu_total", 0) - prev.get("cpu_total", 0)
            idle_delta = sample.get("cpu_idle", 0) - prev.get("cpu_idle", 0)
            if total_delta > 0:
                sample["cpu_util_percent"] = round(100.0 * (1.0 - idle_delta / total_delta), 2)
        prev = sample


def sample_summary(samples: list[dict]) -> dict:
    ok = [item for item in samples if item.get("success")]
    add_cpu_util(ok)
    cpu_vals = [item["cpu_util_percent"] for item in ok if "cpu_util_percent" in item]
    loads = [float(item.get("load1") or 0.0) for item in ok]
    ssh = [float(item.get("ssh_seconds") or 0.0) for item in samples]
    mem_available = [item.get("mem", {}).get("MemAvailable") for item in ok if item.get("mem", {}).get("MemAvailable") is not None]
    return {
        "samples": len(samples),
        "successful_samples": len(ok),
        "failed_samples": len(samples) - len(ok),
        "max_load1": max(loads) if loads else None,
        "avg_load1": (sum(loads) / len(loads)) if loads else None,
        "max_cpu_util_percent": max(cpu_vals) if cpu_vals else None,
        "avg_cpu_util_percent": (sum(cpu_vals) / len(cpu_vals)) if cpu_vals else None,
        "max_ssh_seconds": max(ssh) if ssh else None,
        "min_mem_available_kb": min(mem_available) if mem_available else None,
        "tail": samples[-10:],
    }


def run_monitored_writer(
    args: argparse.Namespace,
    project: str,
    remote_asset_dir: Path,
    local_stage_dir: Path,
    remote_stage_dir: Path,
    commands_dir: Path,
    jobs: int,
) -> dict:
    local_stage_dir.mkdir(parents=True, exist_ok=True)
    remote_stage_dir_q = q(remote_stage_dir)
    remote_summary = remote_stage_dir / f"command-files_write_jobs{jobs}_summary.json"
    remote_details = remote_stage_dir / f"command-files_write_jobs{jobs}_details.json"
    remote_log = remote_stage_dir / f"command-files_write_jobs{jobs}.log"
    remote_cmd = [
        "sudo",
        "-n",
        "python3",
        remote_asset_dir / "delta_fib_host_writer.py",
        "--project-name",
        project,
        "--metric",
        args.metric,
        "--source",
        "command-files",
        "--jobs",
        jobs,
        "--executor",
        "process",
        "--write-mode",
        "setns-ip",
        "--command-timeout",
        args.command_timeout,
        "--summary-path",
        remote_summary,
        "--details-path",
        remote_details,
        "--commands-dir",
        commands_dir,
        "--batch-lines",
        args.batch_lines,
        "--batch-sleep-ms",
        args.batch_sleep_ms,
    ]
    ssh(args, f"mkdir -p {remote_stage_dir_q}", timeout=60)
    ssh(args, f"rm -f {q(remote_summary)} {q(remote_details)} {q(remote_log)}", timeout=60)
    command = remote_quote_cmd(remote_cmd) + f" > {q(remote_log)} 2>&1"
    proc = subprocess.Popen(
        ssh_cmd(args.vm_user, args.vm, command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    started = time.monotonic()
    samples: list[dict] = []
    high_load_hits = 0
    timed_out = False
    while proc.poll() is None:
        sample = sample_vm(args)
        samples.append(sample)
        load1 = sample.get("load1") if sample.get("success") else None
        if load1 is not None and load1 >= args.abort_load_threshold:
            high_load_hits += 1
        else:
            high_load_hits = 0
        if high_load_hits >= args.abort_load_samples:
            proc.terminate()
            raise RuntimeError(f"jobs={jobs} reached abort load threshold {args.abort_load_threshold}")
        if time.monotonic() - started > args.writer_timeout:
            proc.terminate()
            timed_out = True
            break
        time.sleep(args.sample_interval)

    output = ""
    try:
        output, _ = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()

    local_log = local_stage_dir / remote_log.name
    local_summary = local_stage_dir / remote_summary.name
    local_details = local_stage_dir / remote_details.name
    local_samples = local_stage_dir / f"command-files_write_jobs{jobs}_samples.json"
    fetch_remote_file(args, remote_log, local_log, required=False)
    fetch_remote_file(args, remote_summary, local_summary, required=False)
    fetch_remote_file(args, remote_details, local_details, required=False)
    local_samples.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    writer = None
    if local_summary.exists() and local_summary.stat().st_size > 0:
        writer = json.loads(local_summary.read_text(encoding="utf-8"))
    result = {
        "jobs": jobs,
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "local_summary": str(local_summary) if local_summary.exists() else None,
        "local_details": str(local_details) if local_details.exists() else None,
        "local_log": str(local_log) if local_log.exists() else None,
        "ssh_output_tail": (output or "")[-1000:],
        "monitor": sample_summary(samples),
        "writer": writer,
    }
    if timed_out:
        result["success"] = False
        result["error"] = "writer timeout"
    elif proc.returncode != 0:
        result["success"] = False
        result["error"] = f"writer ssh command exited {proc.returncode}"
    elif not writer:
        result["success"] = False
        result["error"] = "missing writer summary"
    else:
        result["success"] = not writer.get("failures") and not writer.get("batch_errors_total")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, default=3083)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--vm", default=DEFAULT_VM)
    parser.add_argument("--vm-user", default=DEFAULT_USER)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--reuse-remote-run-dir",
        type=Path,
        default=None,
        help="Reuse an existing VM run directory and only execute the jobs sweep against its jobs_sweep/commands.",
    )
    parser.add_argument(
        "--reuse-project-name",
        default=None,
        help="Compose project name to use with --reuse-remote-run-dir. Defaults to topology-<scale>-sweep.",
    )
    parser.add_argument("--metric", type=int, default=232)
    parser.add_argument("--down-count", type=int, default=10)
    parser.add_argument("--load-threshold", type=float, default=40.0)
    parser.add_argument("--max-wait-seconds", type=int, default=7200)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--restart-delay", type=float, default=0.1)
    parser.add_argument("--initial-write-jobs", type=int, default=8)
    parser.add_argument("--full-dump-jobs", type=int, default=4)
    parser.add_argument("--jobs", type=parse_jobs, default=parse_jobs("4,8,12,16"))
    parser.add_argument("--extra-jobs", type=parse_jobs, default=parse_jobs("24,32"))
    parser.add_argument("--continue-after-16-load-threshold", type=float, default=250.0)
    parser.add_argument("--batch-lines", type=int, default=1000)
    parser.add_argument("--batch-sleep-ms", type=float, default=5.0)
    parser.add_argument("--command-timeout", type=float, default=600)
    parser.add_argument("--setup-timeout", type=float, default=21600)
    parser.add_argument("--writer-timeout", type=float, default=7200)
    parser.add_argument("--reset-timeout", type=float, default=1800)
    parser.add_argument("--import-timeout", type=float, default=1800)
    parser.add_argument("--generate-timeout", type=float, default=1800)
    parser.add_argument("--sample-interval", type=float, default=15.0)
    parser.add_argument("--sample-timeout", type=float, default=15.0)
    parser.add_argument("--abort-load-threshold", type=float, default=250.0)
    parser.add_argument("--abort-load-samples", type=int, default=120)
    parser.add_argument("--skip-reset", action="store_true")
    parser.add_argument("--skip-seedemu-rebuild", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--import-seedemu-images", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (args.run_dir or (SCRIPT_DIR / "runs" / f"full_write_jobs_sweep_{args.scale}_{timestamp}")).expanduser().resolve()
    summary = Summary(
        run_dir / "summary.json",
        {
            "mode": "full-write-jobs-sweep",
            "command": sys.argv,
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "scale": args.scale,
            "vm": args.vm,
            "domain": args.domain,
            "settings": jsonable_settings(args),
        },
    )
    try:
        project = args.reuse_project_name or f"topology-{args.scale}-sweep"
        if args.reuse_remote_run_dir:
            remote_run_dir = args.reuse_remote_run_dir
            remote_asset_dir = remote_run_dir / "delta-assets"
            remote_stage_dir = remote_run_dir / "jobs_sweep"
            commands_dir = remote_stage_dir / "commands"
            summary.update(run_dir=str(run_dir), project_name=project, remote_run_dir=str(remote_run_dir), reused_remote_run=True)
            summary.stage("setup_reused", True, remote_run_dir=str(remote_run_dir), commands_dir=str(commands_dir))
        else:
            output_dir = ensure_output_dir(args, run_dir)
            summary.update(output_dir=str(output_dir), topology_services=compose_service_count(output_dir), run_dir=str(run_dir))
            if not args.skip_reset:
                reset_vm(args, summary, "reset_vm")
            setup = run_setup(args, output_dir, run_dir, "full", project, enable_delta_after_initial=False)
            remote_run_dir = Path(setup["remote_run_dir"])
            remote_asset_dir = remote_run_dir / "delta-assets"
            remote_stage_dir = remote_run_dir / "jobs_sweep"
            commands_dir = remote_stage_dir / "commands"
            summary.stage("setup", True, summary_path=str(run_dir / "full" / "setup_summary.json"), remote_run_dir=str(remote_run_dir))

            victims = choose_victims(args, project, args.down_count)
            (run_dir / "victims.json").write_text(json.dumps(victims, indent=2) + "\n", encoding="utf-8")
            stopped = stop_victims(args, victims)
            summary.stage("stop_victims", True, **{k: v for k, v in stopped.items() if k != "success"})
            wait_load_below(args, summary, "wait_load_after_fault")

            dump = run_host_writer(
                args,
                project,
                remote_asset_dir,
                run_dir / "full_dump",
                remote_stage_dir,
                source="birdc-primary",
                jobs=args.full_dump_jobs,
                dump_only=True,
                commands_dir=commands_dir,
            )
            summary.stage("full_dump_birdc_primary", True, writer=dump)

        sweep_jobs = list(args.jobs)
        results = []
        ran_16 = False
        continue_extra = False
        for jobs in sweep_jobs:
            result = run_monitored_writer(
                args,
                project,
                remote_asset_dir,
                run_dir / "jobs" / f"jobs_{jobs}",
                remote_stage_dir,
                commands_dir,
                jobs,
            )
            results.append(result)
            summary.stage(f"write_jobs_{jobs}", bool(result.get("success")), **result)
            if not result.get("success"):
                break
            if jobs == 16:
                ran_16 = True
                max_load = (result.get("monitor") or {}).get("max_load1") or 0.0
                continue_extra = max_load < args.continue_after_16_load_threshold
        if ran_16 and continue_extra:
            for jobs in args.extra_jobs:
                result = run_monitored_writer(
                    args,
                    project,
                    remote_asset_dir,
                    run_dir / "jobs" / f"jobs_{jobs}",
                    remote_stage_dir,
                    commands_dir,
                    jobs,
                )
                results.append(result)
                summary.stage(f"write_jobs_{jobs}", bool(result.get("success")), **result)
                if not result.get("success"):
                    break
        else:
            summary.stage(
                "extra_jobs_skipped",
                True,
                ran_16=ran_16,
                continue_extra=continue_extra,
                threshold=args.continue_after_16_load_threshold,
            )
        summary.update(results=results)
        summary.finalize(all(item.get("success") for item in results), None if all(item.get("success") for item in results) else "one or more jobs failed")
        return 0 if all(item.get("success") for item in results) else 1
    except Exception as exc:
        summary.finalize(False, repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
