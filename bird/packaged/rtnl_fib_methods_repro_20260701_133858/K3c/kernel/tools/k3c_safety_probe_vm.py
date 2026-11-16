#!/usr/bin/env python3
"""Run focused safety/impact probes against the K3c RT bulk kernel VM."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path


SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "LogLevel=ERROR",
    "-o",
    "ConnectTimeout=10",
]


def run(cmd: list[str], *, stdin: str | None = None, timeout: int = 300) -> dict:
    started = time.monotonic()
    proc = subprocess.run(cmd, input=stdin, text=True, capture_output=True, timeout=timeout)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def ssh(args: argparse.Namespace, remote_cmd: str, *, timeout: int = 300) -> dict:
    return run(["ssh", *SSH_OPTS, f"{args.vm_user}@{args.vm}", remote_cmd], timeout=timeout)


def remote_bash(args: argparse.Namespace, script: str, *, sudo: bool = True, timeout: int = 300) -> dict:
    prefix = "sudo -n bash -s" if sudo else "bash -s"
    return run(["ssh", *SSH_OPTS, f"{args.vm_user}@{args.vm}", prefix], stdin=script, timeout=timeout)


def rsync_to_vm(args: argparse.Namespace, src: Path, dst: str) -> dict:
    return run(["rsync", "-a", "-e", "ssh " + " ".join(SSH_OPTS), str(src), f"{args.vm_user}@{args.vm}:{dst}"], timeout=300)


def parse_int(stdout: str, key: str) -> int | None:
    match = re.search(rf"^{re.escape(key)}=(\d+)$", stdout, re.MULTILINE)
    return int(match.group(1)) if match else None


def add_result(summary: dict, name: str, result: dict, success: bool | None = None, note: str = "") -> None:
    if success is None:
        success = result["returncode"] == 0
    summary["tests"][name] = {
        "success": bool(success),
        "returncode": result["returncode"],
        "duration_seconds": result["duration_seconds"],
        "stdout_tail": result["stdout"][-4000:],
        "stderr_tail": result["stderr"][-4000:],
        "note": note,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vm", default="192.168.122.60")
    parser.add_argument("--vm-user", default="lxl")
    parser.add_argument("--project-name", default="topology-3083")
    parser.add_argument("--run-dir", required=True, help="Local experiment run directory for summary output.")
    parser.add_argument("--routes", type=int, default=10000)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "k3c_safety_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = f"/home/lxl/bird/runs/{run_dir.name}/k3c_safety_probe"
    remote_public_dir = f"/tmp/k3c_safety_probe_{run_dir.name}"
    bulk_src = Path(__file__).with_name("kernel_bulk_writer.py").resolve()

    summary: dict = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "vm": args.vm,
        "vm_user": args.vm_user,
        "project_name": args.project_name,
        "local_dir": str(out_dir),
        "remote_dir": remote_dir,
        "remote_public_dir": remote_public_dir,
        "tests": {},
    }

    add_result(summary, "prepare_remote_dir", ssh(args, f"mkdir -p {shlex.quote(remote_dir)}"))
    add_result(summary, "prepare_remote_public_dir", ssh(args, f"rm -rf {shlex.quote(remote_public_dir)} && mkdir -p {shlex.quote(remote_public_dir)} && chmod 0755 {shlex.quote(remote_public_dir)}"))
    add_result(summary, "sync_bulk_writer", rsync_to_vm(args, bulk_src, f"{remote_dir}/kernel_bulk_writer.py"))
    add_result(summary, "sync_public_bulk_writer", rsync_to_vm(args, bulk_src, f"{remote_public_dir}/kernel_bulk_writer.py"))
    add_result(summary, "chmod_public_bulk_writer", ssh(args, f"chmod 0755 {shlex.quote(remote_public_dir)}/kernel_bulk_writer.py"))
    add_result(summary, "vm_identity", ssh(args, "uname -r; hostname; cat /proc/loadavg"))

    start_ts = ssh(args, "date '+%Y-%m-%d %H:%M:%S'")["stdout"].strip()
    summary["kernel_log_since"] = start_ts

    standard_script = r"""
set -euo pipefail
NS="k3c_probe_std_$$"
cleanup() { ip netns del "$NS" 2>/dev/null || true; }
trap cleanup EXIT
ip netns add "$NS"
ip -n "$NS" link set lo up
ip -n "$NS" route replace 198.51.100.0/24 dev lo proto static metric 100
ip -n "$NS" route get 198.51.100.1 >/tmp/${NS}.get
count_before=$(ip -n "$NS" route show 198.51.100.0/24 | wc -l)
ip -n "$NS" route del 198.51.100.0/24 dev lo proto static metric 100
count_after=$(ip -n "$NS" route show 198.51.100.0/24 | wc -l)
root_leak=$(ip route show 198.51.100.0/24 | wc -l)
cat /tmp/${NS}.get
echo "count_before=${count_before}"
echo "count_after=${count_after}"
echo "root_leak=${root_leak}"
test "$count_before" -eq 1
test "$count_after" -eq 0
test "$root_leak" -eq 0
"""
    add_result(
        summary,
        "standard_rtnetlink_temp_netns",
        remote_bash(args, standard_script, timeout=120),
        note="Standard iproute2 rtnetlink add/get/delete in an isolated netns should work and not leak to root netns.",
    )

    bulk_script = f"""
set -euo pipefail
NS="k3c_probe_bulk_$$"
cleanup() {{ ip netns del "$NS" 2>/dev/null || true; }}
trap cleanup EXIT
ip netns add "$NS"
ip -n "$NS" link set lo up
python3 {remote_dir}/kernel_bulk_writer.py --netns-path /var/run/netns/$NS --routes {args.routes} --batch 2000 --ops-per-attr 2000 --base 198.18.0.0 --route-type blackhole --op replace --protocol 12 --metric 232 --summary-path {remote_dir}/bulk_replace.json >/tmp/${{NS}}.replace.out
count_after_replace=$(ip -n "$NS" route show | grep -c 'metric 232' || true)
python3 {remote_dir}/kernel_bulk_writer.py --netns-path /var/run/netns/$NS --routes {args.routes} --batch 2000 --ops-per-attr 2000 --base 198.18.0.0 --route-type blackhole --op del --protocol 12 --metric 232 --summary-path {remote_dir}/bulk_delete.json >/tmp/${{NS}}.delete.out
count_after_delete=$(ip -n "$NS" route show | grep -c 'metric 232' || true)
root_leak=$(ip route show | grep -c '198\\.18\\.' || true)
cat /tmp/${{NS}}.replace.out
cat /tmp/${{NS}}.delete.out
echo "count_after_replace=${{count_after_replace}}"
echo "count_after_delete=${{count_after_delete}}"
echo "root_leak=${{root_leak}}"
test "$count_after_replace" -eq {args.routes}
test "$count_after_delete" -eq 0
test "$root_leak" -eq 0
"""
    add_result(
        summary,
        "k3c_bulk_temp_netns",
        remote_bash(args, bulk_script, timeout=300),
        note="K3c BIRD_RT_BULK should write/delete synthetic blackhole routes in a temporary netns only.",
    )

    permission_script = f"""
set +e
sudo -u nobody python3 {remote_public_dir}/kernel_bulk_writer.py --routes 1 --batch 1 --base 198.19.0.1 --route-type blackhole --op replace --protocol 12 --metric 232 >{remote_dir}/unprivileged.out 2>{remote_dir}/unprivileged.err
rc=$?
sudo python3 {remote_dir}/kernel_bulk_writer.py --routes 1 --batch 1 --base 198.19.0.1 --route-type blackhole --op del --protocol 12 --metric 232 >/dev/null 2>&1 || true
cat {remote_dir}/unprivileged.out
cat {remote_dir}/unprivileged.err
echo "unprivileged_rc=$rc"
test "$rc" -ne 0
"""
    add_result(
        summary,
        "bulk_unprivileged_rejected",
        remote_bash(args, permission_script, sudo=False, timeout=120),
        note="Unprivileged caller should not be able to use the route bulk API.",
    )

    lifecycle_script = r"""
set -euo pipefail
name="k3c-probe-lifecycle-$$"
docker run -d --name "$name" --label k3c_probe=1 --network none ubuntu:24.04 sleep 120 >/tmp/${name}.cid
docker inspect -f '{{.State.Pid}} {{.State.Running}}' "$name"
docker rm -f "$name" >/dev/null
remaining=$(docker ps -aq --filter name="$name" | wc -l)
echo "remaining=${remaining}"
test "$remaining" -eq 0
"""
    add_result(
        summary,
        "docker_netns_lifecycle",
        remote_bash(args, lifecycle_script, sudo=False, timeout=180),
        note="Docker should still create and remove a container/network namespace while the K3c topology is running.",
    )

    monitor_script = f"""
set -euo pipefail
NS="k3c_probe_mon_$$"
cleanup() {{ ip netns del "$NS" 2>/dev/null || true; }}
trap cleanup EXIT
ip netns add "$NS"
ip -n "$NS" link set lo up
ip netns exec "$NS" timeout 5 ip monitor route >{remote_dir}/monitor_standard.out 2>/dev/null &
mon=$!
sleep 0.5
ip -n "$NS" route replace 203.0.113.0/24 dev lo proto static metric 101
ip -n "$NS" route del 203.0.113.0/24 dev lo proto static metric 101
wait "$mon" || true
standard_events=$(wc -l < {remote_dir}/monitor_standard.out)
ip netns exec "$NS" timeout 5 ip monitor route >{remote_dir}/monitor_bulk.out 2>/dev/null &
mon=$!
sleep 0.5
python3 {remote_dir}/kernel_bulk_writer.py --netns-path /var/run/netns/$NS --routes 32 --batch 32 --ops-per-attr 32 --base 198.18.100.0 --route-type blackhole --op replace --protocol 12 --metric 232 >/dev/null
python3 {remote_dir}/kernel_bulk_writer.py --netns-path /var/run/netns/$NS --routes 32 --batch 32 --ops-per-attr 32 --base 198.18.100.0 --route-type blackhole --op del --protocol 12 --metric 232 >/dev/null
wait "$mon" || true
bulk_events=$(wc -l < {remote_dir}/monitor_bulk.out)
echo "standard_events=${{standard_events}}"
echo "bulk_events=${{bulk_events}}"
test "$standard_events" -gt 0
"""
    monitor_result = remote_bash(args, monitor_script, timeout=180)
    standard_events = parse_int(monitor_result["stdout"], "standard_events")
    bulk_events = parse_int(monitor_result["stdout"], "bulk_events")
    add_result(
        summary,
        "route_monitor_notifications",
        monitor_result,
        success=monitor_result["returncode"] == 0,
        note=(
            "Standard rtnetlink route changes should emit monitor events. "
            f"K3c bulk emitted {bulk_events} monitor events in this run; 0 means notification consumers are bypassed by the fast path."
        ),
    )
    summary["tests"]["route_monitor_notifications"]["standard_events"] = standard_events
    summary["tests"]["route_monitor_notifications"]["bulk_events"] = bulk_events

    severe_kernel_log_cmd = (
        "sudo -n journalctl -k --since "
        + shlex.quote(start_ts)
        + " --no-pager | egrep -i 'BUG:|Oops|panic|WARNING:|lockdep|KASAN|refcount|use-after-free|general protection|rtnl_mutex|bird_rtbulk|fib_create_info' || true"
    )
    kernel_log = ssh(args, severe_kernel_log_cmd, timeout=120)
    add_result(
        summary,
        "kernel_log_severe_scan",
        kernel_log,
        success=kernel_log["returncode"] == 0 and not kernel_log["stdout"].strip(),
        note="No matching kernel warning/Oops/refcount/lockdep lines should appear during the probe window.",
    )

    lifecycle_log_cmd = (
        "sudo -n journalctl -k --since "
        + shlex.quote(start_ts)
        + " --no-pager | grep 'unregister_netdevice: waiting' || true"
    )
    lifecycle_log = ssh(args, lifecycle_log_cmd, timeout=120)
    lifecycle_lines = [line for line in lifecycle_log["stdout"].splitlines() if line.strip()]
    add_result(
        summary,
        "kernel_log_lifecycle_warnings",
        lifecycle_log,
        success=lifecycle_log["returncode"] == 0,
        note="Records unregister_netdevice wait messages separately; these indicate netns/device teardown delay, not an Oops by themselves.",
    )
    summary["tests"]["kernel_log_lifecycle_warnings"]["line_count"] = len(lifecycle_lines)

    workqueue_log_cmd = (
        "sudo -n journalctl -k --since "
        + shlex.quote(start_ts)
        + " --no-pager | grep 'workqueue:' || true"
    )
    workqueue_log = ssh(args, workqueue_log_cmd, timeout=120)
    workqueue_lines = [line for line in workqueue_log["stdout"].splitlines() if line.strip()]
    add_result(
        summary,
        "kernel_log_workqueue_warnings",
        workqueue_log,
        success=workqueue_log["returncode"] == 0,
        note="Records workqueue CPU-hog notices separately; these are load/lifecycle signals, not direct evidence of memory corruption.",
    )
    summary["tests"]["kernel_log_workqueue_warnings"]["line_count"] = len(workqueue_lines)

    fetch = run(["rsync", "-a", "-e", "ssh " + " ".join(SSH_OPTS), f"{args.vm_user}@{args.vm}:{remote_dir}/", str(out_dir / "remote/")], timeout=300)
    add_result(summary, "fetch_remote_probe_artifacts", fetch)

    summary["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    summary["success"] = all(item["success"] for item in summary["tests"].values())
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
