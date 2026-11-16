#!/usr/bin/env python3
"""Run BIRD start-mode A/B experiments on the RTNL VM.

The script intentionally stops after BIRD/RIB validation. It does not write
routes to the main FIB, so it can isolate control-plane start/convergence load
from start-kernel/FIB writer load.
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
from typing import Any

import yaml


REPO_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{now_iso()} {message}", flush=True)


def slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value)
    return safe.strip("-") or "run"


def remote_quote(path: str | Path) -> str:
    return shlex.quote(str(path))


def ssh_base(user: str, host: str) -> list[str]:
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
    ]


def ssh_cmd(user: str, host: str, remote: str) -> list[str]:
    return ssh_base(user, host) + [remote]


def rsync_cmd(user: str, host: str) -> list[str]:
    return [
        "rsync",
        "-a",
        "--delete",
        "-e",
        " ".join(shlex.quote(part) for part in ssh_base(user, host)[:-1]),
    ]


def run(
    cmd: list[str],
    *,
    cwd: Path = REPO_DIR,
    input_text: str | None = None,
    capture: bool = False,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    log("$ " + " ".join(shlex.quote(part) for part in cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        input=input_text,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        check=False,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        output = (proc.stdout or "").strip()
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}\n{output}")
    return proc


def ssh_capture_retry(
    args: argparse.Namespace,
    remote: str,
    *,
    input_text: str | None = None,
    attempts: int = 5,
    delay_seconds: int = 10,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    last = None
    for attempt in range(1, attempts + 1):
        proc = run(
            ssh_cmd(args.vm_user, args.vm, remote),
            input_text=input_text,
            capture=True,
            check=False,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return proc
        last = proc
        log(f"ssh attempt {attempt}/{attempts} failed: {(proc.stdout or '').strip()[-300:]}")
        if attempt < attempts:
            time.sleep(delay_seconds)
    raise RuntimeError(f"ssh command failed after {attempts} attempts: {remote}\n{(last.stdout if last else '').strip()}")


class Summary:
    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path = path
        self.data = data
        self.data["started_at"] = now_iso()
        self.data["stages"] = {}
        self._starts: dict[str, float] = {}
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def start_stage(self, name: str, **extra: Any) -> None:
        self._starts[name] = time.monotonic()
        self.data["stages"][name] = {"started_at": now_iso(), "success": None, **extra}
        self.write()

    def update_stage(self, name: str, **extra: Any) -> None:
        self.data["stages"].setdefault(name, {}).update(extra)
        self.write()

    def finish_stage(self, name: str, success: bool, **extra: Any) -> None:
        started = self._starts.get(name)
        duration = round(time.monotonic() - started, 3) if started is not None else None
        self.data["stages"].setdefault(name, {}).update(
            {"success": success, "ended_at": now_iso(), "duration_seconds": duration, **extra}
        )
        self.write()

    def skip_stage(self, name: str, reason: str, **extra: Any) -> None:
        self.data["stages"][name] = {
            "started_at": now_iso(),
            "ended_at": now_iso(),
            "success": True,
            "skipped": True,
            "reason": reason,
            "duration_seconds": 0.0,
            **extra,
        }
        self.write()

    def finalize(self, success: bool, error: str | None = None) -> None:
        self.data["success"] = success
        self.data["error"] = error
        self.data["ended_at"] = now_iso()
        self.write()


def compose_service_count(output_dir: Path) -> int:
    data = yaml.safe_load((output_dir / "docker-compose.yml").read_text(encoding="utf-8"))
    return len((data or {}).get("services") or {})


def sync_stage(args: argparse.Namespace, summary: Summary, remote_run_dir: Path, remote_output: Path, remote_asset_dir: Path) -> None:
    summary.start_stage("sync", remote_run_dir=str(remote_run_dir))
    run(ssh_cmd(args.vm_user, args.vm, f"mkdir -p {remote_quote(remote_run_dir)} {remote_quote(remote_output)} {remote_quote(remote_asset_dir)}"))
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
    files = [
        SCRIPT_DIR / "run.py",
        SCRIPT_DIR / "run_all.py",
        SCRIPT_DIR / "build_delta_fast_images.py",
        SCRIPT_DIR / "delta_fib_agent.py",
        SCRIPT_DIR / "install_bird_delta_vm.sh",
        REPO_DIR / "1buildFastDockerImages.py",
    ]
    for path in files:
        run(rsync_cmd(args.vm_user, args.vm) + [str(path), f"{args.vm_user}@{args.vm}:{remote_run_dir / path.name}"])
    run(rsync_cmd(args.vm_user, args.vm) + [str(args.bird_binary), f"{args.vm_user}@{args.vm}:{remote_asset_dir / 'bird'}"])
    run(ssh_cmd(args.vm_user, args.vm, f"chmod +x {remote_quote(remote_run_dir)}/*.py {remote_quote(remote_run_dir / 'install_bird_delta_vm.sh')} {remote_quote(remote_asset_dir / 'bird')}"))
    summary.finish_stage("sync", True)


def build_stage(args: argparse.Namespace, summary: Summary, remote_run_dir: Path, remote_output: Path, remote_asset_dir: Path) -> None:
    if args.skip_build:
        summary.skip_stage("build", "--skip-build")
        return
    summary.start_stage("build", image_mode=args.image_mode)
    if args.image_mode == "patched":
        remote_cmd = [
            "python3",
            str(remote_run_dir / "build_delta_fast_images.py"),
            str(remote_output / "docker-compose.yml"),
            "--fast-builder",
            str(remote_run_dir / "1buildFastDockerImages.py"),
            "--bird-binary",
            str(remote_asset_dir / "bird"),
            "--agent",
            str(remote_run_dir / "delta_fib_agent.py"),
            "--work-dir",
            str(remote_output / ".deltaFastDockerBuild"),
            "--project-name",
            args.project_name,
            "--metric",
            str(args.metric),
            "--skip-service",
            "seedemu_internet_map",
        ]
        run(ssh_cmd(args.vm_user, args.vm, " ".join(shlex.quote(part) for part in remote_cmd)))
        summary.finish_stage("build", True, mode="delta-fast", fast_summary=str(remote_output / ".deltaFastDockerBuild" / "summary.json"))
        return

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
    summary.finish_stage("build", True, mode="fast", remote_summary=str(remote_run_dir / "build_summary.json"))


def up_stage(args: argparse.Namespace, summary: Summary, remote_run_dir: Path, remote_output: Path) -> None:
    if args.skip_up:
        summary.skip_stage("up", "--skip-up")
        return
    summary.start_stage("up", down_first=args.down_first)
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
    if args.down_first:
        remote_cmd.append("--down-first")
    run(ssh_cmd(args.vm_user, args.vm, " ".join(shlex.quote(part) for part in remote_cmd)))
    summary.finish_stage("up", True, remote_summary=str(remote_run_dir / "up_summary.json"))


def start_only_stage(args: argparse.Namespace, summary: Summary, remote_run_dir: Path, remote_output: Path) -> None:
    summary.start_stage(
        "start_bird",
        start_mode="start-only",
        entry_method=args.entry_method,
        start_bird_delay=args.start_bird_delay,
    )
    remote_cmd = [
        "python3",
        str(remote_run_dir / "run_all.py"),
        str(remote_output),
        "--project-name",
        args.project_name,
        "--no-build",
        "--no-up",
        "--no-start-kernel",
        "--entry-method",
        args.entry_method,
        "--start-bird-delay",
        str(args.start_bird_delay),
        "--start-stage-jobs",
        str(args.start_stage_jobs),
        "--nsenter-sudo",
        args.nsenter_sudo,
        "--summary-path",
        str(remote_run_dir / "start_only_summary.json"),
    ]
    run(ssh_cmd(args.vm_user, args.vm, " ".join(shlex.quote(part) for part in remote_cmd)))
    proc = run(ssh_cmd(args.vm_user, args.vm, f"cat {remote_quote(remote_run_dir / 'start_only_summary.json')}"), capture=True)
    remote_summary = json.loads(proc.stdout or "{}")
    stage = (remote_summary.get("stages") or {}).get("start_bird", {})
    summary.finish_stage("start_bird", bool(stage.get("success")), remote_stage=stage, remote_summary=str(remote_run_dir / "start_only_summary.json"))
    if not stage.get("success"):
        raise RuntimeError("start-only BIRD stage failed")


def full_restart_stage(args: argparse.Namespace, summary: Summary) -> None:
    summary.start_stage("start_bird", start_mode="full-restart", jobs=args.restart_jobs, delta_enabled=False)
    env = {
        "VM_HOST": args.vm,
        "VM_USER": args.vm_user,
        "PROJECT_NAME": args.project_name,
        "LIMIT_CONTAINERS": "0",
        "DELTA_FIB_METRIC": str(args.metric),
        "DELTA_JOBS": str(args.restart_jobs),
        "DELTA_ENABLE_ON_RESTART": "0",
    }
    cmd = ["env"] + [f"{key}={value}" for key, value in env.items()] + [str(SCRIPT_DIR / "install_bird_delta_vm.sh"), "restart-bird"]
    run(cmd)
    summary.finish_stage("start_bird", True)


def wait_rib_stable_stage(args: argparse.Namespace, summary: Summary) -> None:
    summary.start_stage(
        "wait_rib_stable",
        load_threshold=args.load_threshold,
        stable_seconds=args.stable_seconds,
        max_wait_seconds=args.max_wait_seconds,
        poll_interval=args.poll_interval,
        sample_size=args.stability_samples,
    )
    deadline = time.monotonic() + args.max_wait_seconds
    stable_since: float | None = None
    last_counts: dict[str, int] | None = None
    samples: list[dict[str, Any]] = []

    while time.monotonic() < deadline:
        load_proc = ssh_capture_retry(args, "awk '{print $1}' /proc/loadavg", timeout=60)
        load = float((load_proc.stdout or "999").strip().split()[0])
        counts = remote_sample_counts(args, args.stability_samples)
        same_counts = last_counts == counts if last_counts is not None else False
        enough_samples = len(counts) >= args.stability_samples
        low_load = load < args.load_threshold
        now = time.monotonic()
        sample = {
            "at": now_iso(),
            "load1": load,
            "sample_count": len(counts),
            "expected_sample_count": args.stability_samples,
            "same_counts": same_counts,
            "enough_samples": enough_samples,
            "low_load": low_load,
            "stable_for_seconds": round(now - stable_since, 1) if stable_since else 0,
        }
        samples.append(sample)
        summary.update_stage("wait_rib_stable", latest=sample, samples_tail=samples[-12:])
        log(f"wait_rib_stable load1={load:.2f} samples={len(counts)} same_counts={same_counts}")
        if low_load and same_counts and enough_samples:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= args.stable_seconds:
                summary.finish_stage("wait_rib_stable", True, samples_tail=samples[-12:], final_counts=counts)
                return
        else:
            stable_since = None
        last_counts = counts
        time.sleep(args.poll_interval)
    raise RuntimeError("RIB/load stability gate timed out")


def remote_sample_counts(args: argparse.Namespace, sample_size: int) -> dict[str, int]:
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
    proc = ssh_capture_retry(
        args,
        f"bash -s -- {shlex.quote(args.project_name)} {sample_size}",
        input_text=script,
        timeout=180,
    )
    counts: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1].isdigit():
            counts[fields[0]] = int(fields[1])
    return counts


def validate_bird_stage(args: argparse.Namespace, summary: Summary) -> None:
    summary.start_stage("validate_bird", samples=args.validate_samples)
    script = r"""
import json
import subprocess
import sys

project = sys.argv[1]
limit = int(sys.argv[2])
ps = subprocess.check_output(
    ["docker", "ps", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.ID}} {{.Names}}"],
    text=True,
)
containers = [line.split()[0] for line in ps.splitlines() if len(line.split()) >= 2 and line.split()[1].startswith("as") and "brd-r" in line.split()[1]]
if len(containers) > limit:
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
printf '%s\t%s\t%s\n' "$name" "$status" "$rib"
'''
    out = subprocess.check_output(["docker", "exec", cid, "sh", "-lc", shell], text=True).strip()
    fields = out.split("\t")
    item = {"container": cid, "raw": out, "success": False}
    if len(fields) == 3:
        name, status, rib = fields
        rib_i = int(rib or -1) if (rib or "").isdigit() else -1
        item.update({"name": name, "status": int(status), "rib": rib_i})
        item["success"] = item["status"] == 0 and item["rib"] >= 0
    ok = ok and item["success"]
    items.append(item)
print(json.dumps({"success": ok, "sample_count": len(items), "samples": items}, indent=2))
sys.exit(0 if ok else 1)
"""
    proc = run(
        ssh_cmd(args.vm_user, args.vm, f"python3 - {shlex.quote(args.project_name)} {args.validate_samples}"),
        input_text=script,
        capture=True,
        check=False,
        timeout=240,
    )
    try:
        result = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        raise RuntimeError(f"BIRD validation output was not JSON: {proc.stdout}")
    success = proc.returncode == 0 and bool(result.get("success"))
    summary.finish_stage("validate_bird", success, **{k: v for k, v in result.items() if k != "success"})
    if not success:
        raise RuntimeError("BIRD validation failed")


def fetch_remote_artifacts(args: argparse.Namespace, summary: Summary, remote_run_dir: Path, local_run_dir: Path) -> None:
    summary.start_stage("fetch")
    for name in ["build_summary.json", "up_summary.json", "start_only_summary.json"]:
        run(
            rsync_cmd(args.vm_user, args.vm)
            + [f"{args.vm_user}@{args.vm}:{remote_run_dir / name}", str(local_run_dir / name)],
            check=False,
        )
    summary.finish_stage("fetch", True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--vm", default="192.168.122.60")
    parser.add_argument("--vm-user", default="lxl")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--image-mode", choices=["stock", "patched"], required=True)
    parser.add_argument("--start-mode", choices=["start-only", "full-restart"], required=True)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-up", action="store_true")
    parser.add_argument("--down-first", action="store_true")
    parser.add_argument("--metric", type=int, default=232)
    parser.add_argument("--bird-binary", type=Path, default=SCRIPT_DIR / "bird-2.17.2-delta" / "bird-focal")
    parser.add_argument("--restart-jobs", type=int, default=2)
    parser.add_argument("--entry-method", choices=["nsenter", "docker-exec"], default="nsenter")
    parser.add_argument("--nsenter-sudo", choices=["auto", "always", "never"], default="always")
    parser.add_argument("--start-bird-delay", type=float, default=0.02)
    parser.add_argument("--start-stage-jobs", type=int, default=1)
    parser.add_argument("--load-threshold", type=float, default=50.0)
    parser.add_argument("--stable-seconds", type=int, default=120)
    parser.add_argument("--max-wait-seconds", type=int, default=1800)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--stability-samples", type=int, default=20)
    parser.add_argument("--validate-samples", type=int, default=10)
    parser.add_argument("--run-root", type=Path, default=SCRIPT_DIR / "runs")
    parser.add_argument("--summary-path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.bird_binary = args.bird_binary.expanduser().resolve()
    if not (args.output_dir / "docker-compose.yml").is_file():
        raise RuntimeError(f"missing docker-compose.yml in {args.output_dir}")
    if args.image_mode == "patched" and not args.bird_binary.is_file():
        raise RuntimeError(f"missing patched BIRD binary: {args.bird_binary}")

    service_count = compose_service_count(args.output_dir)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"start-bird-ab-{service_count}_{timestamp}_{slug(args.project_name)}"
    local_run_dir = args.run_root.expanduser().resolve() / run_name
    remote_run_dir = Path(f"/home/{args.vm_user}/bird/runs/{run_name}")
    remote_output = remote_run_dir / "output"
    remote_asset_dir = remote_run_dir / "assets"
    summary_path = args.summary_path.expanduser().resolve() if args.summary_path else local_run_dir / "summary.json"

    summary = Summary(
        summary_path,
        {
            "mode": "start-bird-ab",
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
                "image_mode": args.image_mode,
                "start_mode": args.start_mode,
                "skip_build": args.skip_build,
                "skip_up": args.skip_up,
                "down_first": args.down_first,
                "restart_jobs": args.restart_jobs,
                "entry_method": args.entry_method,
                "start_bird_delay": args.start_bird_delay,
                "load_threshold": args.load_threshold,
                "stable_seconds": args.stable_seconds,
                "max_wait_seconds": args.max_wait_seconds,
                "stability_samples": args.stability_samples,
            },
        },
    )

    error = None
    try:
        sync_stage(args, summary, remote_run_dir, remote_output, remote_asset_dir)
        build_stage(args, summary, remote_run_dir, remote_output, remote_asset_dir)
        up_stage(args, summary, remote_run_dir, remote_output)
        if args.start_mode == "start-only":
            start_only_stage(args, summary, remote_run_dir, remote_output)
        else:
            full_restart_stage(args, summary)
        wait_rib_stable_stage(args, summary)
        validate_bird_stage(args, summary)
        fetch_remote_artifacts(args, summary, remote_run_dir, local_run_dir)
    except Exception as exc:
        error = str(exc)
        log(f"ERROR: {error}")
    finally:
        summary.finalize(error is None, error)
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
