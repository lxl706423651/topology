#!/usr/bin/env python3
"""Run the standalone BIRD source-delta FIB experiment on the VM.

This runner is intentionally separate from run_all.py and writeFIB_F. It can
build/up the topology, start the patched BIRD delta protocol, write the initial
main FIB through the delta agent, start resident agents, and validate samples.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_VM = "192.168.122.60"
DEFAULT_USER = "lxl"
DEFAULT_PROJECT = "topology-291e63c1"


class Summary:
    def __init__(self, path: Path, initial: dict) -> None:
        self.path = path
        self.data = {
            **initial,
            "started_at": now_iso(),
            "ended_at": None,
            "duration_seconds": None,
            "success": None,
            "error": None,
            "stages": {},
        }
        self._start = time.monotonic()
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def start_stage(self, name: str, **info) -> None:
        self.data["stages"][name] = {"started_at": now_iso(), "success": None, **info}
        self.write()

    def finish_stage(self, name: str, success: bool, **info) -> None:
        stage = self.data["stages"].setdefault(name, {})
        stage.update(info)
        stage["success"] = success
        stage["ended_at"] = now_iso()
        if "started_monotonic" in stage:
            stage["duration_seconds"] = round(time.monotonic() - stage.pop("started_monotonic"), 3)
        self.write()

    def update_stage(self, name: str, **info) -> None:
        self.data["stages"].setdefault(name, {}).update(info)
        self.write()

    def skip_stage(self, name: str, reason: str, **info) -> None:
        self.data["stages"][name] = {
            "started_at": now_iso(),
            "ended_at": now_iso(),
            "success": True,
            "skipped": True,
            "reason": reason,
            **info,
        }
        self.write()

    def finalize(self, success: bool, error: str | None = None) -> None:
        self.data["success"] = success
        self.data["error"] = error
        self.data["ended_at"] = now_iso()
        self.data["duration_seconds"] = round(time.monotonic() - self._start, 3)
        self.write()


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    log("$ " + shlex.join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
    )
    if check and proc.returncode != 0:
        output = f"\n{proc.stdout}" if capture and proc.stdout else ""
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {shlex.join(cmd)}{output}")
    return proc


def ssh_cmd(user: str, host: str, remote: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-o",
        "ConnectTimeout=30",
        f"{user}@{host}",
        remote,
    ]


def ssh_capture_retry(
    args: argparse.Namespace,
    remote: str,
    *,
    input_text: str | None = None,
    attempts: int = 5,
    delay_seconds: int = 10,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    last_proc: subprocess.CompletedProcess | None = None
    for attempt in range(1, attempts + 1):
        cmd = ssh_cmd(args.vm_user, args.vm, remote)
        log("$ " + shlex.join(cmd))
        proc = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        if proc.returncode == 0:
            return proc
        last_proc = proc
        msg = (proc.stdout or "").strip().replace("\n", " ")
        log(f"SSH attempt {attempt}/{attempts} failed with exit code {proc.returncode}: {msg[-500:]}")
        if attempt < attempts:
            time.sleep(delay_seconds)
    output = (last_proc.stdout or "").strip() if last_proc else ""
    raise RuntimeError(f"ssh command failed after {attempts} attempts: {remote}\n{output}")


def rsync_cmd(user: str, host: str) -> list[str]:
    return [
        "rsync",
        "-a",
        "--delete",
        "-e",
        "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR",
    ]


def remote_quote(path: Path | str) -> str:
    return shlex.quote(str(path))


def is_local_vm_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def read_vm_load1(args: argparse.Namespace) -> float:
    if is_local_vm_host(args.vm):
        return float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])
    load_proc = ssh_capture_retry(args, "awk '{print $1}' /proc/loadavg", timeout=60)
    return float((load_proc.stdout or "999").strip().split()[0])


def compose_service_count(output_dir: Path) -> int:
    compose = output_dir / "docker-compose.yml"
    data = yaml.safe_load(compose.read_text(encoding="utf-8"))
    return len(data.get("services", {}))


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in text).strip("-") or "project"


def parse_route_count(text: str) -> int | None:
    for line in text.splitlines():
        if "routes for" in line and "networks in table master4" in line:
            parts = line.split()
            try:
                return int(parts[5])
            except (IndexError, ValueError):
                return None
    return None


def local_shell_env(env: dict[str, str], cmd: list[str]) -> list[str]:
    return ["env"] + [f"{key}={value}" for key, value in env.items()] + cmd


def sync_stage(args: argparse.Namespace, summary: Summary, remote_run_dir: Path, remote_output: Path, remote_asset_dir: Path) -> None:
    summary.start_stage("sync", started_monotonic=time.monotonic(), remote_run_dir=str(remote_run_dir))
    run(ssh_cmd(args.vm_user, args.vm, f"mkdir -p {remote_quote(remote_run_dir)} {remote_quote(remote_output)} {remote_quote(remote_asset_dir)}"))

    if args.skip_build_up:
        summary.update_stage("sync", output_sync_skipped=True)
    else:
        run(
            rsync_cmd(args.vm_user, args.vm)
            + [
                "--exclude",
                "run_logs/",
                "--exclude",
                "summary.json",
                f"{args.output_dir}/",
                f"{args.vm_user}@{args.vm}:{remote_output}/",
            ]
        )

    script_files = [
        SCRIPT_DIR / "run.py",
        SCRIPT_DIR / "run_all.py",
        SCRIPT_DIR / "install_bird_delta_vm.sh",
        SCRIPT_DIR / "delta_fib_agent.py",
        SCRIPT_DIR / "delta_fib_host_writer.py",
        SCRIPT_DIR / "verify_bird_fib_vm.py",
        SCRIPT_DIR / "build_delta_fast_images.py",
        REPO_DIR / "1buildFastDockerImages.py",
    ]
    for path in script_files:
        run(rsync_cmd(args.vm_user, args.vm) + [str(path), f"{args.vm_user}@{args.vm}:{remote_run_dir / path.name}"])
    run(rsync_cmd(args.vm_user, args.vm) + [str(SCRIPT_DIR / "delta_fib_agent.py"), f"{args.vm_user}@{args.vm}:{remote_asset_dir / 'delta_fib_agent.py'}"])
    run(rsync_cmd(args.vm_user, args.vm) + [str(SCRIPT_DIR / "delta_fib_host_writer.py"), f"{args.vm_user}@{args.vm}:{remote_asset_dir / 'delta_fib_host_writer.py'}"])
    run(rsync_cmd(args.vm_user, args.vm) + [str(SCRIPT_DIR / "kernel" / "tools" / "kernel_bulk_writer.py"), f"{args.vm_user}@{args.vm}:{remote_asset_dir / 'kernel_bulk_writer.py'}"])
    run(rsync_cmd(args.vm_user, args.vm) + [str(SCRIPT_DIR / "kernel" / "tools" / "kernel_bulk_birdc_writer.py"), f"{args.vm_user}@{args.vm}:{remote_asset_dir / 'kernel_bulk_birdc_writer.py'}"])
    run(rsync_cmd(args.vm_user, args.vm) + [str(SCRIPT_DIR / "kernel" / "tools" / "kernel_bulk_command_writer.py"), f"{args.vm_user}@{args.vm}:{remote_asset_dir / 'kernel_bulk_command_writer.py'}"])
    run(rsync_cmd(args.vm_user, args.vm) + [str(SCRIPT_DIR / "kernel" / "tools" / "rtnetlink_route_writer.py"), f"{args.vm_user}@{args.vm}:{remote_asset_dir / 'rtnetlink_route_writer.py'}"])
    run(rsync_cmd(args.vm_user, args.vm) + [str(args.bird_binary), f"{args.vm_user}@{args.vm}:{remote_asset_dir / 'bird'}"])
    run(ssh_cmd(args.vm_user, args.vm, f"chmod +x {remote_quote(remote_run_dir)}/*.py {remote_quote(remote_run_dir / 'install_bird_delta_vm.sh')} {remote_quote(remote_asset_dir / 'bird')} {remote_quote(remote_asset_dir)}/*.py"))
    summary.finish_stage("sync", True)


def build_up_stage(args: argparse.Namespace, summary: Summary, remote_run_dir: Path, remote_output: Path, remote_asset_dir: Path) -> None:
    if args.skip_build_up:
        summary.skip_stage("build", "--skip-build-up")
        summary.skip_stage("up", "--skip-build-up")
        return

    if args.build_mode == "delta-fast":
        summary.start_stage("build", started_monotonic=time.monotonic(), mode=args.build_mode)
        remote_cmd = [
            "python3",
            str(remote_run_dir / "build_delta_fast_images.py"),
            str(remote_output / "docker-compose.yml"),
            "--fast-builder",
            str(remote_run_dir / "1buildFastDockerImages.py"),
            "--bird-binary",
            str(remote_asset_dir / "bird"),
            "--agent",
            str(remote_asset_dir / "delta_fib_agent.py"),
            "--work-dir",
            str(remote_output / ".deltaFastDockerBuild"),
            "--project-name",
            args.project_name,
            "--metric",
            str(args.metric),
            "--skip-service",
            "seedemu_internet_map",
        ]
        delta_fast_skip_load = os.environ.get("DELTA_FAST_SKIP_LOAD") == "1"
        if delta_fast_skip_load:
            remote_cmd.append("--skip-load")
        run(ssh_cmd(args.vm_user, args.vm, " ".join(shlex.quote(part) for part in remote_cmd)))
        summary.finish_stage(
            "build",
            True,
            mode=args.build_mode,
            fast_summary=str(remote_output / ".deltaFastDockerBuild" / "summary.json"),
            delta_fast_skip_load=delta_fast_skip_load,
        )
    elif args.build_mode == "fast":
        summary.start_stage("build", started_monotonic=time.monotonic(), mode=args.build_mode)
        remote_cmd = [
            "python3",
            str(remote_run_dir / "run.py"),
            str(remote_output),
            "--project-name",
            args.project_name,
            "--build-mode",
            "fast",
            "--no-up",
            "--summary-path",
            str(remote_run_dir / "build_summary.json"),
        ]
        run(ssh_cmd(args.vm_user, args.vm, " ".join(shlex.quote(part) for part in remote_cmd)))
        summary.finish_stage("build", True, mode=args.build_mode, remote_summary=str(remote_run_dir / "build_summary.json"))
    elif args.build_mode == "reuse":
        summary.skip_stage("build", "--build-mode reuse")
    else:
        raise RuntimeError(f"unknown build mode: {args.build_mode}")

    summary.start_stage("up", started_monotonic=time.monotonic(), down_first=args.down_first)
    remote_cmd = [
        "python3",
        str(remote_run_dir / "run.py"),
        str(remote_output),
        "--project-name",
        args.project_name,
        "--no-build",
        "--summary-path",
        str(remote_run_dir / "up_summary.json"),
    ]
    if os.environ.get("UP_BATCH"):
        remote_cmd += ["--up-batch", os.environ["UP_BATCH"]]
    if os.environ.get("UP_JOBS"):
        remote_cmd += ["--up-jobs", os.environ["UP_JOBS"]]
    if args.down_first:
        remote_cmd.append("--down-first")
    run(ssh_cmd(args.vm_user, args.vm, " ".join(shlex.quote(part) for part in remote_cmd)))
    summary.finish_stage("up", True, remote_summary=str(remote_run_dir / "up_summary.json"))


def run_delta_installer(
    args: argparse.Namespace,
    phase: str,
    jobs: int,
    *,
    check: bool = True,
    capture: bool = False,
    enable_on_restart: bool | None = None,
    serial_delay: float | None = None,
    ignore_initial_feed: bool | None = None,
) -> subprocess.CompletedProcess:
    env = {
        "VM_HOST": args.vm,
        "VM_USER": args.vm_user,
        "PROJECT_NAME": args.project_name,
        "LIMIT_CONTAINERS": str(args.limit_containers),
        "DELTA_FIB_METRIC": str(args.metric),
        "DELTA_JOBS": str(jobs),
    }
    if enable_on_restart is not None:
        env["DELTA_ENABLE_ON_RESTART"] = "1" if enable_on_restart else "0"
    if serial_delay is not None:
        env["DELTA_SERIAL_DELAY"] = str(serial_delay)
    if ignore_initial_feed is not None:
        env["DELTA_IGNORE_INITIAL_FEED"] = "1" if ignore_initial_feed else "0"
    if phase in {"restart-bird", "start-clean-bird"}:
        env["DELTA_RESTART_LOAD_GATE_EVERY"] = str(args.restart_load_gate_every)
        env["DELTA_RESTART_LOAD_THRESHOLD"] = str(args.restart_load_threshold)
        env["DELTA_RESTART_LOAD_POLL_INTERVAL"] = str(args.restart_load_poll_interval)
        env["DELTA_RESTART_LOAD_MAX_WAIT"] = str(args.restart_load_max_wait)
    return run(
        local_shell_env(env, [str(SCRIPT_DIR / "install_bird_delta_vm.sh"), phase]),
        cwd=REPO_DIR,
        check=check,
        capture=capture,
    )


def collect_container_agent_stats(args: argparse.Namespace, stats_file: str) -> dict:
    script = r"""
import concurrent.futures
import json
import statistics
import subprocess
import sys

project = sys.argv[1]
stats_file = sys.argv[2]

ps = subprocess.check_output(
    ["docker", "ps", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.ID}} {{.Names}}"],
    text=True,
)
containers = [
    line.split()[0]
    for line in ps.splitlines()
    if len(line.split()) >= 2 and line.split()[1].startswith("as") and "brd-r" in line.split()[1]
]

def read_one(cid):
    try:
        raw = subprocess.check_output(["docker", "exec", cid, "cat", stats_file], text=True, timeout=10)
        data = json.loads(raw)
        started = float(data.get("started_at_epoch") or 0)
        ended = float(data.get("ended_at_epoch") or 0)
        return {
            "ok": True,
            "elapsed_seconds": max(0.0, ended - started),
            "raw_route_events": int(data.get("raw_route_events") or 0),
            "collapsed_events": int(data.get("collapsed_events") or 0),
            "commands": int(data.get("commands") or 0),
            "flushes": int(data.get("flushes") or 0),
            "read_collapse_seconds": float(data.get("read_collapse_seconds") or 0.0),
            "write_seconds": float(data.get("write_seconds") or 0.0),
            "batch_errors": len(data.get("batch_errors") or []),
            "mode": data.get("mode"),
            "final_collapse": bool(data.get("final_collapse")),
            "start_byte_offset": int(data.get("start_byte_offset") or 0),
            "end_byte_offset": int(data.get("end_byte_offset") or 0),
        }
    except Exception as exc:
        return {"ok": False, "container": cid, "error": repr(exc)}

with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    rows = list(executor.map(read_one, containers))

ok = [row for row in rows if row.get("ok")]
failed = [row for row in rows if not row.get("ok")]

def numeric_summary(key):
    vals = [float(row.get(key) or 0.0) for row in ok]
    if not vals:
        return {"sum": 0.0}
    vals_sorted = sorted(vals)
    return {
        "sum": sum(vals),
        "avg": statistics.fmean(vals),
        "p50": statistics.median(vals),
        "p95": vals_sorted[min(len(vals_sorted) - 1, int(len(vals_sorted) * 0.95))],
        "max": max(vals),
    }

print(json.dumps({
    "total_brdnodes": len(containers),
    "stats_ok": len(ok),
    "stats_failed": len(failed),
    "failed_samples": failed[:20],
    "batch_errors_total": sum(int(row.get("batch_errors") or 0) for row in ok),
    "raw_route_events": sum(int(row.get("raw_route_events") or 0) for row in ok),
    "collapsed_events": sum(int(row.get("collapsed_events") or 0) for row in ok),
    "commands": sum(int(row.get("commands") or 0) for row in ok),
    "flushes": sum(int(row.get("flushes") or 0) for row in ok),
    "final_collapse_all": all(bool(row.get("final_collapse")) for row in ok) if ok else False,
    "elapsed_seconds": numeric_summary("elapsed_seconds"),
    "read_collapse_seconds": numeric_summary("read_collapse_seconds"),
    "write_seconds": numeric_summary("write_seconds"),
}, indent=2))
"""
    remote = f"python3 - {shlex.quote(args.project_name)} {shlex.quote(stats_file)}"
    proc = subprocess.run(
        ssh_cmd(args.vm_user, args.vm, remote),
        input=script,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"collect stats failed: {proc.stdout}")
    return json.loads(proc.stdout)


def run_remote_host_writer(
    args: argparse.Namespace,
    remote_asset_dir: Path,
    summary_path: Path,
    details_path: Path,
    extra: list[str] | None = None,
) -> dict:
    remote_cmd = [
        "sudo",
        "-n",
        "python3",
        str(remote_asset_dir / "delta_fib_host_writer.py"),
        "--project-name",
        args.project_name,
        "--metric",
        str(args.metric),
        "--jobs",
        str(args.write_jobs),
        "--executor",
        "process",
        "--write-mode",
        "setns-ip",
        "--summary-path",
        str(summary_path),
        "--details-path",
        str(details_path),
    ]
    if args.limit_containers:
        remote_cmd.extend(["--limit-containers", str(args.limit_containers)])
    if extra:
        remote_cmd.extend(extra)
    run(ssh_cmd(args.vm_user, args.vm, " ".join(shlex.quote(part) for part in remote_cmd)))
    proc = run(ssh_cmd(args.vm_user, args.vm, f"sudo -n cat {remote_quote(summary_path)}"), capture=True)
    return json.loads(proc.stdout or "{}")


def run_remote_k1_birdc_writer(
    args: argparse.Namespace,
    remote_asset_dir: Path,
    remote_run_dir: Path,
    summary_path: Path,
    details_path: Path,
) -> dict:
    work_dir = remote_run_dir / "k1_bulk_initial"
    remote_cmd = [
        "sudo",
        "-n",
        "python3",
        str(remote_asset_dir / "kernel_bulk_birdc_writer.py"),
        "--project-name",
        args.project_name,
        "--metric",
        str(args.metric),
        "--protocol",
        str(args.k1_protocol),
        "--jobs",
        str(args.write_jobs),
        "--dump-jobs",
        str(args.k1_dump_jobs),
        "--batch",
        str(args.k1_batch),
        "--ops-per-attr",
        str(args.k1_ops_per_attr),
        "--route-action",
        args.k1_route_action,
        "--multipath-policy",
        args.k1_multipath_policy,
        "--work-dir",
        str(work_dir),
        "--command-timeout",
        str(args.k1_command_timeout),
        "--summary-path",
        str(summary_path),
        "--details-path",
        str(details_path),
    ]
    if args.limit_containers:
        remote_cmd.extend(["--limit-containers", str(args.limit_containers)])
    if args.k1_keep_dumps:
        remote_cmd.append("--keep-dumps")
    run(ssh_cmd(args.vm_user, args.vm, " ".join(shlex.quote(part) for part in remote_cmd)))
    proc = run(ssh_cmd(args.vm_user, args.vm, f"sudo -n cat {remote_quote(summary_path)}"), capture=True)
    return json.loads(proc.stdout or "{}")


def run_remote_k3a_rtnl_birdc_writer(
    args: argparse.Namespace,
    remote_asset_dir: Path,
    remote_run_dir: Path,
    summary_path: Path,
    details_path: Path,
) -> dict:
    commands_dir = remote_run_dir / "k3a_rtnl_initial_commands"
    dump_summary_path = remote_run_dir / "k3a_rtnl_initial_dump_summary.json"
    dump_details_path = remote_run_dir / "k3a_rtnl_initial_dump_details.json"
    dump_cmd = [
        "sudo",
        "-n",
        "python3",
        str(remote_asset_dir / "delta_fib_host_writer.py"),
        "--project-name",
        args.project_name,
        "--metric",
        str(args.metric),
        "--jobs",
        str(args.write_jobs),
        "--executor",
        "process",
        "--write-mode",
        "setns-ip",
        "--source",
        args.initial_source,
        "--route-action",
        args.initial_route_action,
        "--dump-only",
        "--commands-dir",
        str(commands_dir),
        "--summary-path",
        str(dump_summary_path),
        "--details-path",
        str(dump_details_path),
    ]
    if args.limit_containers:
        dump_cmd.extend(["--limit-containers", str(args.limit_containers)])
    if args.initial_skip_deletes:
        dump_cmd.append("--skip-deletes")
    if args.initial_omit_onlink:
        dump_cmd.append("--omit-onlink")
    run(ssh_cmd(args.vm_user, args.vm, " ".join(shlex.quote(part) for part in dump_cmd)))

    write_cmd = [
        "sudo",
        "-n",
        "python3",
        str(remote_asset_dir / "rtnetlink_route_writer.py"),
        "--project-name",
        args.project_name,
        "--commands-dir",
        str(commands_dir),
        "--metric",
        str(args.metric),
        "--protocol",
        str(args.k3a_protocol),
        "--jobs",
        str(args.write_jobs),
        "--batch",
        str(args.k3a_batch),
        "--ack-mode",
        args.k3a_ack_mode,
        "--multipath-policy",
        args.k3a_multipath_policy,
        "--summary-path",
        str(summary_path),
        "--details-path",
        str(details_path),
    ]
    if args.limit_containers:
        write_cmd.extend(["--limit-containers", str(args.limit_containers)])
    run(ssh_cmd(args.vm_user, args.vm, " ".join(shlex.quote(part) for part in write_cmd)))
    proc = run(ssh_cmd(args.vm_user, args.vm, f"sudo -n cat {remote_quote(summary_path)}"), capture=True)
    write_summary = json.loads(proc.stdout or "{}")
    try:
        dump_summary = json.loads(run(ssh_cmd(args.vm_user, args.vm, f"sudo -n cat {remote_quote(dump_summary_path)}"), capture=True).stdout or "{}")
    except Exception:
        dump_summary = {}
    write_summary["dump_summary"] = dump_summary
    write_summary["commands_dir"] = str(commands_dir)
    return write_summary


def probe_bird_state_stage(args: argparse.Namespace, summary: Summary) -> dict:
    summary.start_stage("probe_bird_state", started_monotonic=time.monotonic(), sample_method="random")
    proc = run_delta_installer(args, "probe-bird-state", 1, capture=True)
    result = None
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        result = json.loads(line)
        break
    if result is None:
        raise RuntimeError(f"probe-bird-state did not return JSON: {(proc.stdout or '').strip()[-1000:]}")
    summary.finish_stage("probe_bird_state", True, result=result)
    return result


def delta_start_stage(args: argparse.Namespace, summary: Summary, bird_state: dict | None = None) -> None:
    if args.build_mode != "delta-fast" or args.force_install:
        summary.start_stage("install_delta", started_monotonic=time.monotonic(), jobs=args.delta_jobs)
        run_delta_installer(args, "install", args.delta_jobs)
        summary.finish_stage("install_delta", True)
    else:
        summary.skip_stage("install_delta", "patched BIRD and delta config were baked into delta-fast images")

    bird_state_name = (bird_state or {}).get("state")
    if bird_state_name == "no_brd_containers":
        raise RuntimeError("no brdnode containers found after up; cannot start BIRD")
    if bird_state_name == "bird_not_started":
        start_phase = "start-clean-bird"
        start_mode = "clean-start"
    else:
        start_phase = "restart-bird"
        start_mode = "restart-existing-or-unknown"

    summary.start_stage(
        "restart_bird",
        started_monotonic=time.monotonic(),
        phase=start_phase,
        start_mode=start_mode,
        probed_state=bird_state_name,
        jobs=args.restart_jobs,
        restart_delay=args.restart_delay,
        delta_enabled=not args.defer_delta,
        load_gate_every=args.restart_load_gate_every,
        load_gate_threshold=args.restart_load_threshold,
        load_gate_poll_interval=args.restart_load_poll_interval,
        load_gate_max_wait=args.restart_load_max_wait,
    )
    run_delta_installer(
        args,
        start_phase,
        args.restart_jobs,
        enable_on_restart=not args.defer_delta,
        serial_delay=args.restart_delay,
    )
    summary.finish_stage("restart_bird", True)


def remote_sample_counts(args: argparse.Namespace, sample_size: int) -> dict:
    script = r"""
set -euo pipefail
project="$1"
sample="$2"
docker ps --filter "label=com.docker.compose.project=${project}" --format '{{.ID}} {{.Names}}' \
  | awk -v limit="${sample}" '$2 ~ /^as[0-9]+brd-r/ {count++; if (count <= limit) print $1}' \
  | while read -r cid; do
      docker exec "$cid" sh -lc 'printf "%s " "$(hostname)"; birdc -s /run/bird/bird.ctl show route primary count 2>/dev/null | awk "/master4/ {print \$6}"' || echo "$cid ERR"
    done
"""
    cmd = f"bash -s -- {shlex.quote(args.project_name)} {sample_size}"
    proc = ssh_capture_retry(args, cmd, input_text=script, timeout=180)
    counts = {}
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1].isdigit():
            counts[fields[0]] = int(fields[1])
    return counts


def wait_stable_stage(args: argparse.Namespace, summary: Summary) -> None:
    summary.start_stage(
        "wait_rib_stable",
        started_monotonic=time.monotonic(),
        load_threshold=args.load_threshold,
        stable_seconds=args.stable_seconds,
        max_wait_seconds=args.max_wait_seconds,
        sample_size=args.stability_samples,
        load_only=args.load_only_stable,
    )
    deadline = time.monotonic() + args.max_wait_seconds
    stable_since: float | None = None
    last_counts: dict | None = None
    samples = []

    while time.monotonic() < deadline:
        try:
            load = read_vm_load1(args)
        except Exception as exc:
            sample = {
                "at": now_iso(),
                "error": "load_sample_failed",
                "message": str(exc),
                "route_count_check_skipped": args.load_only_stable,
            }
            samples.append(sample)
            summary.update_stage("wait_rib_stable", latest=sample, samples_tail=samples[-10:])
            log(f"wait_rib_stable load sample failed: {exc}; retrying after {args.poll_interval}s")
            stable_since = None
            time.sleep(args.poll_interval)
            continue
        counts = {}
        count_error = None
        if not args.load_only_stable:
            try:
                counts = remote_sample_counts(args, args.stability_samples)
            except Exception as exc:
                count_error = str(exc)
        same_counts = None if args.load_only_stable else (last_counts == counts if last_counts is not None else False)
        enough_samples = True if args.load_only_stable else len(counts) >= args.stability_samples
        low_load = load < args.load_threshold
        now = time.monotonic()
        sample = {
            "at": now_iso(),
            "load1": load,
            "sample_count": len(counts),
            "expected_sample_count": args.stability_samples,
            "enough_samples": enough_samples,
            "same_counts": same_counts,
            "low_load": low_load,
            "route_count_check_skipped": args.load_only_stable,
            "stable_for_seconds": round(now - stable_since, 1) if stable_since else 0,
        }
        if count_error is not None:
            sample["error"] = "route_count_sample_failed"
            sample["message"] = count_error
        samples.append(sample)
        summary.update_stage("wait_rib_stable", latest=sample, samples_tail=samples[-10:])
        if args.load_only_stable:
            log(f"wait_rib_stable load-only load1={load:.2f} threshold={args.load_threshold:.2f}")
        elif count_error is not None:
            log(f"wait_rib_stable route-count sample failed: {count_error}; retrying after {args.poll_interval}s")
        else:
            log(f"wait_rib_stable load1={load:.2f} sample_count={len(counts)} same_counts={same_counts}")

        if args.load_only_stable and low_load:
            summary.finish_stage("wait_rib_stable", True, samples_tail=samples[-10:], final_load1=load, route_count_check_skipped=True)
            return

        if low_load and same_counts and enough_samples:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= args.stable_seconds:
                summary.finish_stage("wait_rib_stable", True, samples_tail=samples[-10:], final_counts=counts)
                return
        else:
            stable_since = None

        last_counts = counts
        time.sleep(args.poll_interval)

    raise RuntimeError("RIB/load stability gate timed out")


def validate_bird_stage(args: argparse.Namespace, summary: Summary) -> None:
    summary.start_stage("validate_bird", started_monotonic=time.monotonic(), samples=args.validate_samples)
    script = r"""
import json
import subprocess
import sys

project = sys.argv[1]
limit = int(sys.argv[2])
mode = sys.argv[3]
expect_delta = sys.argv[4] == "1"

ps = subprocess.check_output(
    ["docker", "ps", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.ID}} {{.Names}}"],
    text=True,
)
containers = [line.split()[0] for line in ps.splitlines() if len(line.split()) >= 2 and line.split()[1].startswith("as") and "brd-r" in line.split()[1]]
if limit > 0:
    if mode == "head":
        containers = containers[:limit]
    elif len(containers) > limit:
        step = max(1, len(containers) // limit)
        containers = containers[::step][:limit]

items = []
ok = True
for cid in containers:
    shell = r'''
name="$(hostname)"
status=0
birdc -s /run/bird/bird.ctl show status >/dev/null 2>&1 || status=$?
rib="$(birdc -s /run/bird/bird.ctl show route primary count 2>/dev/null | awk '/master4/ {print $6; exit}')"
delta="$(birdc -s /run/bird/bird.ctl show protocols 2>/dev/null | awk '/delta_main/ {print $4; exit}')"
events=0
[ -f /run/bird/delta-events.tsv ] && events="$(wc -l < /run/bird/delta-events.tsv)"
printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$rib" "$delta" "$events"
'''
    out = subprocess.check_output(["docker", "exec", cid, "sh", "-lc", shell], text=True).strip()
    fields = out.split("\t")
    item = {"container": cid, "raw": out, "success": False}
    if len(fields) == 5:
        name, status, rib, delta, events = fields
        rib_i = int(rib or -1) if (rib or "").isdigit() else -1
        events_i = int(events or 0) if (events or "").isdigit() else 0
        item.update({"name": name, "status": int(status), "rib": rib_i, "delta_state": delta, "events": events_i})
        item["success"] = item["status"] == 0 and item["rib"] >= 0
        if expect_delta:
            item["success"] = item["success"] and item["delta_state"] == "up" and item["events"] > 0
    ok = ok and item["success"]
    items.append(item)

print(json.dumps({"success": ok, "expect_delta": expect_delta, "sample_count": len(items), "total_brdnodes": len(containers), "samples": items}, indent=2))
sys.exit(0 if ok else 1)
"""
    validate_limit = args.limit_containers or args.validate_samples
    validate_mode = "head" if args.limit_containers else "spread"
    expect_delta = "0" if args.defer_delta else "1"
    remote = f"python3 - {shlex.quote(args.project_name)} {validate_limit} {validate_mode} {expect_delta}"
    proc = subprocess.run(
        ssh_cmd(args.vm_user, args.vm, remote),
        input=script,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"BIRD validation output was not JSON: {proc.stdout}")
    validation_success = bool(result.pop("success", False))
    summary.finish_stage("validate_bird", proc.returncode == 0 and validation_success, validation_success=validation_success, **result)
    if proc.returncode != 0 or not validation_success:
        raise RuntimeError("BIRD validation failed")


def write_fib_stage(args: argparse.Namespace, summary: Summary, remote_run_dir: Path, remote_asset_dir: Path) -> None:
    if args.pre_flush_fib:
        summary.start_stage("pre_flush_fib", started_monotonic=time.monotonic(), jobs=args.write_jobs, metric=args.metric)
        flush_stats = run_remote_host_writer(
            args,
            remote_asset_dir,
            remote_run_dir / "delta_host_writer_pre_flush_summary.json",
            remote_run_dir / "delta_host_writer_pre_flush_details.json",
            ["--flush-only"],
        )
        summary.finish_stage("pre_flush_fib", True, writer=flush_stats)

    summary.start_stage(
        "write_initial_fib",
        started_monotonic=time.monotonic(),
        jobs=args.write_jobs,
        metric=args.metric,
        writer=args.fib_writer,
        skip_deletes=args.initial_skip_deletes,
    )
    if args.fib_writer == "host-setns":
        extra = [
            "--source",
            args.initial_source,
            "--route-action",
            args.initial_route_action,
        ]
        if args.initial_skip_deletes:
            extra.append("--skip-deletes")
        if args.initial_omit_onlink:
            extra.append("--omit-onlink")
        stats = run_remote_host_writer(
            args,
            remote_asset_dir,
            remote_run_dir / "delta_host_writer_initial_summary.json",
            remote_run_dir / "delta_host_writer_initial_details.json",
            extra,
        )
        summary.finish_stage("write_initial_fib", True, writer_stats=stats)
    elif args.fib_writer == "k1-bulk":
        stats = run_remote_k1_birdc_writer(
            args,
            remote_asset_dir,
            remote_run_dir,
            remote_run_dir / "k1_bulk_initial_summary.json",
            remote_run_dir / "k1_bulk_initial_details.json",
        )
        summary.finish_stage("write_initial_fib", True, writer_stats=stats)
    elif args.fib_writer == "k3a-rtnl":
        stats = run_remote_k3a_rtnl_birdc_writer(
            args,
            remote_asset_dir,
            remote_run_dir,
            remote_run_dir / "k3a_rtnl_initial_summary.json",
            remote_run_dir / "k3a_rtnl_initial_details.json",
        )
        summary.finish_stage("write_initial_fib", True, writer_stats=stats)
    else:
        run_delta_installer(args, "write-initial", args.write_jobs)
        stats = collect_container_agent_stats(args, "/run/bird/delta-fib-initial.json")
        summary.finish_stage("write_initial_fib", True, agent_stats=stats)

    if args.defer_delta:
        if args.skip_enable_delta_after_initial:
            summary.skip_stage("enable_delta", "--skip-enable-delta-after-initial")
        else:
            summary.start_stage("enable_delta", started_monotonic=time.monotonic(), jobs=args.delta_jobs)
            run_delta_installer(args, "enable-delta", args.delta_jobs, ignore_initial_feed=True)
            summary.finish_stage("enable_delta", True)

    if args.skip_start_agents:
        summary.skip_stage("start_agents", "--skip-start-agents")
    else:
        summary.start_stage("start_agents", started_monotonic=time.monotonic(), jobs=args.agent_jobs)
        run_delta_installer(args, "start-agent", args.agent_jobs)
        summary.finish_stage("start_agents", True)


def validate_stage(args: argparse.Namespace, summary: Summary) -> None:
    summary.start_stage("validate", started_monotonic=time.monotonic(), samples=args.validate_samples, metric=args.metric)
    script = r"""
import json
import subprocess
import sys

project = sys.argv[1]
metric = sys.argv[2]
limit = int(sys.argv[3])
mode = sys.argv[4]

ps = subprocess.check_output(
    ["docker", "ps", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.ID}} {{.Names}}"],
    text=True,
)
containers = [line.split()[0] for line in ps.splitlines() if len(line.split()) >= 2 and line.split()[1].startswith("as") and "brd-r" in line.split()[1]]
if limit > 0:
    if mode == "head":
        containers = containers[:limit]
    elif len(containers) > limit:
        step = max(1, len(containers) // limit)
        containers = containers[::step][:limit]

items = []
ok = True
for cid in containers:
    shell = r'''
name="$(hostname)"
rib="$(birdc -s /run/bird/bird.ctl show route primary count 2>/dev/null | awk '/master4/ {print $6; exit}')"
fib="$(ip route show proto bird metric METRIC | awk '/^[[:space:]]/ {next} NF {n++} END {print n+0}')"
delta="$(birdc -s /run/bird/bird.ctl show protocols 2>/dev/null | awk '/delta_main/ {print $4; exit}')"
agent=0
if [ -s /run/bird/delta-fib-agent.pid ] && kill -0 "$(cat /run/bird/delta-fib-agent.pid)" >/dev/null 2>&1; then agent=1; fi
errors=0
if [ -f /run/bird/delta-fib-initial.json ]; then errors="$(python3 - <<'PY'
import json
p="/run/bird/delta-fib-initial.json"
try:
    data=json.load(open(p))
    print(len(data.get("batch_errors", [])))
except Exception:
    print(999)
PY
)"
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rib" "$fib" "$delta" "$agent" "$errors"
'''.replace("METRIC", metric)
    out = subprocess.check_output(["docker", "exec", cid, "sh", "-lc", shell], text=True).strip()
    fields = out.split("\t")
    item = {"container": cid, "raw": out}
    if len(fields) == 6:
        name, rib, fib, delta, agent, errors = fields
        item.update({"name": name, "rib": int(rib or -1), "fib": int(fib or -1), "delta_state": delta, "agent": agent == "1", "initial_errors": int(errors)})
        item["success"] = item["rib"] == item["fib"] and item["delta_state"] == "up" and item["agent"] and item["initial_errors"] == 0
    else:
        item["success"] = False
    ok = ok and item["success"]
    items.append(item)

print(json.dumps({"success": ok, "sample_count": len(items), "total_brdnodes": len(containers), "samples": items}, indent=2))
sys.exit(0 if ok else 1)
"""
    validate_limit = args.limit_containers or args.validate_samples
    validate_mode = "head" if args.limit_containers else "spread"
    remote = f"python3 - {shlex.quote(args.project_name)} {args.metric} {validate_limit} {validate_mode}"
    proc = subprocess.run(
        ssh_cmd(args.vm_user, args.vm, remote),
        input=script,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"validation output was not JSON: {proc.stdout}")
    validation_success = bool(result.pop("success", False))
    summary.finish_stage("validate", proc.returncode == 0 and validation_success, validation_success=validation_success, **result)
    if proc.returncode != 0:
        raise RuntimeError("validation failed")


def fetch_stage(args: argparse.Namespace, summary: Summary, remote_run_dir: Path, local_run_dir: Path) -> None:
    summary.start_stage("fetch_artifacts", started_monotonic=time.monotonic())
    artifact_dir = local_run_dir / "remote"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name in ["build_summary.json", "up_summary.json"]:
        run(rsync_cmd(args.vm_user, args.vm) + [f"{args.vm_user}@{args.vm}:{remote_run_dir / name}", str(artifact_dir / name)], check=False)
    run(
        rsync_cmd(args.vm_user, args.vm)
        + [f"{args.vm_user}@{args.vm}:{remote_run_dir}/run_manifest.json", str(artifact_dir / "run_manifest.json")],
        check=False,
    )
    run(
        rsync_cmd(args.vm_user, args.vm)
        + [f"{args.vm_user}@{args.vm}:{remote_run_dir}/output/.deltaFastDockerBuild/summary.json", str(artifact_dir / "delta_fast_summary.json")],
        check=False,
    )
    summary.finish_stage("fetch_artifacts", True, artifact_dir=str(artifact_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--vm", default=DEFAULT_VM)
    parser.add_argument("--vm-user", default=DEFAULT_USER)
    parser.add_argument("--project-name", default=DEFAULT_PROJECT)
    parser.add_argument("--build-mode", choices=["delta-fast", "fast", "reuse"], default="delta-fast")
    parser.add_argument("--skip-build-up", action="store_true")
    parser.add_argument(
        "--skip-delta-start",
        action="store_true",
        help="Resume from an existing container set with BIRD already restarted; continue at RIB stability wait.",
    )
    parser.add_argument("--down-first", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-install", action="store_true", help="Run install phase even when using delta-fast images.")
    parser.add_argument(
        "--defer-delta",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Start BIRD without delta protocol, then enable delta after the initial FIB sync.",
    )
    parser.add_argument(
        "--skip-enable-delta-after-initial",
        action="store_true",
        help="With --defer-delta, leave delta disabled after initial FIB sync. Useful for full birdc-primary controls.",
    )
    parser.add_argument(
        "--skip-start-agents",
        action="store_true",
        help="Do not start resident delta_fib_agent processes after initial FIB sync.",
    )
    parser.add_argument(
        "--skip-final-validate",
        action="store_true",
        help="Skip the final validation that requires delta protocol and resident agents.",
    )
    parser.add_argument("--limit-containers", type=int, default=0, help="Limit delta install/restart/write/agent phases for smoke tests.")
    parser.add_argument("--metric", type=int, default=232)
    parser.add_argument("--bird-binary", type=Path, default=SCRIPT_DIR / "bird-2.17.2-delta" / "bird-focal")
    parser.add_argument("--delta-jobs", type=int, default=16)
    parser.add_argument("--restart-jobs", type=int, default=8)
    parser.add_argument("--restart-delay", type=float, default=0.0, help="Sleep between serial restart-bird containers when --restart-jobs=1.")
    parser.add_argument("--restart-load-gate-every", type=int, default=0, help="For serial restart-bird, wait for load1 below the threshold every N containers. 0 disables.")
    parser.add_argument("--restart-load-threshold", type=float, default=50.0)
    parser.add_argument("--restart-load-poll-interval", type=float, default=30.0)
    parser.add_argument("--restart-load-max-wait", type=int, default=1800)
    parser.add_argument("--write-jobs", type=int, default=8)
    parser.add_argument("--agent-jobs", type=int, default=16)
    parser.add_argument("--fib-writer", choices=["host-setns", "container-agent", "k1-bulk", "k3a-rtnl"], default="host-setns")
    parser.add_argument(
        "--pre-flush-fib",
        action="store_true",
        help="Flush existing proto bird metric routes before initial write. Use for repeated tests; fresh containers do not need it.",
    )
    parser.add_argument(
        "--initial-skip-deletes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip delete events during initial clean-FIB writes. Do not use for reconvergence delta writes.",
    )
    parser.add_argument("--initial-source", choices=["event", "birdc-primary"], default="event")
    parser.add_argument("--initial-route-action", choices=["replace", "add"], default="replace")
    parser.add_argument("--initial-omit-onlink", action="store_true")
    parser.add_argument("--k1-batch", type=int, default=2000)
    parser.add_argument("--k1-ops-per-attr", type=int, default=2000)
    parser.add_argument("--k1-dump-jobs", type=int, default=16)
    parser.add_argument("--k1-command-timeout", type=float, default=240)
    parser.add_argument("--k1-protocol", type=int, default=12)
    parser.add_argument("--k1-route-action", choices=["replace", "add", "del"], default="replace")
    parser.add_argument("--k1-multipath-policy", choices=["skip", "first"], default="first")
    parser.add_argument("--k1-keep-dumps", action="store_true")
    parser.add_argument("--k3a-protocol", type=int, default=12)
    parser.add_argument("--k3a-batch", type=int, default=512)
    parser.add_argument("--k3a-ack-mode", choices=["each", "none"], default="each")
    parser.add_argument("--k3a-multipath-policy", choices=["skip", "first"], default="first")
    parser.add_argument("--load-threshold", type=float, default=50.0)
    parser.add_argument("--load-only-stable", action="store_true", help="Treat RIB as stable as soon as load1 drops below --load-threshold; skip birdc route-count sampling.")
    parser.add_argument("--stable-seconds", type=int, default=120)
    parser.add_argument("--max-wait-seconds", type=int, default=1800)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--stability-samples", type=int, default=20)
    parser.add_argument("--validate-samples", type=int, default=10)
    parser.add_argument("--summary-path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.bird_binary = args.bird_binary.expanduser().resolve()
    if not (args.output_dir / "docker-compose.yml").is_file():
        raise RuntimeError(f"missing docker-compose.yml in {args.output_dir}")
    if not args.bird_binary.is_file():
        raise RuntimeError(f"missing patched BIRD binary: {args.bird_binary}")

    service_count = compose_service_count(args.output_dir)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"bird-delta-{service_count}_{timestamp}_{slug(args.project_name)}"
    local_run_dir = SCRIPT_DIR / "runs" / run_name
    remote_run_dir = Path(f"/home/{args.vm_user}/bird/runs/{run_name}")
    remote_output = remote_run_dir / "output"
    remote_asset_dir = remote_run_dir / "delta-assets"
    summary_path = args.summary_path.expanduser().resolve() if args.summary_path else local_run_dir / "summary.json"

    summary = Summary(
        summary_path,
        {
            "mode": "bird-source-delta-experiment",
            "command": sys.argv,
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "local_output_dir": str(args.output_dir),
            "local_run_dir": str(local_run_dir),
            "remote_run_dir": str(remote_run_dir),
            "remote_output_dir": str(remote_output),
            "project_name": args.project_name,
            "topology_services": service_count,
            "settings": {
                "build_mode": args.build_mode,
                "skip_build_up": args.skip_build_up,
                "skip_delta_start": args.skip_delta_start,
                "down_first": args.down_first,
                "defer_delta": args.defer_delta,
                "skip_enable_delta_after_initial": args.skip_enable_delta_after_initial,
                "skip_start_agents": args.skip_start_agents,
                "skip_final_validate": args.skip_final_validate,
                "metric": args.metric,
                "delta_jobs": args.delta_jobs,
                "restart_jobs": args.restart_jobs,
                "restart_delay": args.restart_delay,
                "restart_load_gate_every": args.restart_load_gate_every,
                "restart_load_threshold": args.restart_load_threshold,
                "restart_load_poll_interval": args.restart_load_poll_interval,
                "restart_load_max_wait": args.restart_load_max_wait,
                "write_jobs": args.write_jobs,
                "agent_jobs": args.agent_jobs,
                "fib_writer": args.fib_writer,
                "k1_batch": args.k1_batch,
                "k1_dump_jobs": args.k1_dump_jobs,
                "k1_command_timeout": args.k1_command_timeout,
                "k1_protocol": args.k1_protocol,
                "k1_route_action": args.k1_route_action,
                "k1_multipath_policy": args.k1_multipath_policy,
                "pre_flush_fib": args.pre_flush_fib,
                "initial_skip_deletes": args.initial_skip_deletes,
                "initial_source": args.initial_source,
                "initial_route_action": args.initial_route_action,
                "initial_omit_onlink": args.initial_omit_onlink,
                "limit_containers": args.limit_containers,
                "load_threshold": args.load_threshold,
                "load_only_stable": args.load_only_stable,
                "stable_seconds": args.stable_seconds,
                "max_wait_seconds": args.max_wait_seconds,
            },
        },
    )

    error = None
    try:
        sync_stage(args, summary, remote_run_dir, remote_output, remote_asset_dir)
        manifest = {
            "created_at": now_iso(),
            "local_output_dir": str(args.output_dir),
            "remote_run_dir": str(remote_run_dir),
            "remote_output_dir": str(remote_output),
            "project_name": args.project_name,
            "topology_services": service_count,
            "settings": summary.data["settings"],
        }
        run(
            ssh_cmd(
                args.vm_user,
                args.vm,
                f"cat > {remote_quote(remote_run_dir / 'run_manifest.json')} <<'EOF'\n{json.dumps(manifest, indent=2, ensure_ascii=False)}\nEOF",
            )
        )
        build_up_stage(args, summary, remote_run_dir, remote_output, remote_asset_dir)
        bird_state = probe_bird_state_stage(args, summary)
        if args.skip_delta_start:
            summary.skip_stage(
                "restart_bird",
                "skipped by --skip-delta-start; reusing existing BIRD processes",
                probed_state=bird_state.get("state"),
            )
        else:
            delta_start_stage(args, summary, bird_state)
        wait_stable_stage(args, summary)
        validate_bird_stage(args, summary)
        write_fib_stage(args, summary, remote_run_dir, remote_asset_dir)
        if args.skip_final_validate:
            summary.skip_stage("validate", "--skip-final-validate")
        else:
            validate_stage(args, summary)
        fetch_stage(args, summary, remote_run_dir, local_run_dir)
    except Exception as exc:
        error = str(exc)
        log(f"ERROR: {error}")
    finally:
        summary.finalize(error is None, error)

    return 0 if error is None else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
