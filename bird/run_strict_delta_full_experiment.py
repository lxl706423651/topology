#!/usr/bin/env python3
"""Run a realistic post-failure delta-vs-full FIB comparison on the VM.

The delta arm does not mark offsets immediately before the fault. It writes the
initial FIB, enables BIRD delta logging, records a committed checkpoint, injects
the fault, waits for load to drop, then consumes only events after that
checkpoint. The full arm injects the same fault after a clean initial FIB sync
and then dumps/writes the complete selected RIB through birdc show route primary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
from pathlib import Path
import re
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
DEFAULT_DOMAIN = "bird-rtnl-dev"
DEFAULT_3083_OUTPUT = SCRIPT_DIR / "runs" / "F_reconf_multiscale_20260614_094101" / "scale_3083" / "output"
AS_RE = re.compile(r"^as(\d+)brd-r")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def shjoin(cmd: list[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    capture: bool = False,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    log("$ " + shjoin(cmd))
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=cwd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        output = f"\n{proc.stdout[-4000:]}" if capture and proc.stdout else ""
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {shjoin(cmd)}{output}")
    return proc


def ssh_cmd(user: str, host: str, remote: str, connect_timeout: int = 30) -> list[str]:
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        f"{user}@{host}",
        remote,
    ]


def ssh(
    args: argparse.Namespace,
    remote: str,
    *,
    input_text: str | None = None,
    capture: bool = False,
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    return run(ssh_cmd(args.vm_user, args.vm, remote), input_text=input_text, capture=capture, timeout=timeout, check=check)


def rsync_cmd(user: str, host: str) -> list[str]:
    return [
        "rsync",
        "-a",
        "-e",
        "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR",
    ]


def q(value: str | Path) -> str:
    return shlex.quote(str(value))


class Summary:
    def __init__(self, path: Path, initial: dict) -> None:
        self.path = path
        self.started = time.monotonic()
        self.data = {
            **initial,
            "started_at": now_iso(),
            "ended_at": None,
            "duration_seconds": None,
            "success": None,
            "error": None,
            "stages": {},
        }
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def stage(self, name: str, success: bool, **info) -> None:
        self.data["stages"][name] = {
            "at": now_iso(),
            "success": success,
            **info,
        }
        self.write()

    def update(self, **info) -> None:
        self.data.update(info)
        self.write()

    def finalize(self, success: bool, error: str | None = None) -> None:
        self.data["success"] = success
        self.data["error"] = error
        self.data["ended_at"] = now_iso()
        self.data["duration_seconds"] = round(time.monotonic() - self.started, 3)
        self.write()


def compose_service_count(output_dir: Path) -> int:
    data = yaml.safe_load((output_dir / "docker-compose.yml").read_text(encoding="utf-8"))
    return len(data.get("services", {}))


def jsonable_settings(args: argparse.Namespace) -> dict:
    settings = {}
    for key, value in vars(args).items():
        settings[key] = str(value) if isinstance(value, Path) else value
    return settings


def ensure_output_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.output_dir:
        output_dir = args.output_dir.expanduser().resolve()
        if not (output_dir / "docker-compose.yml").is_file():
            raise RuntimeError(f"missing docker-compose.yml in {output_dir}")
        return output_dir

    if args.scale == 3083 and (DEFAULT_3083_OUTPUT / "docker-compose.yml").is_file():
        return DEFAULT_3083_OUTPUT.resolve()

    if not (REPO_DIR / "autocoder1.py").is_file():
        raise RuntimeError("no --output-dir was given and autocoder1.py is missing")
    generated = run_dir / f"generated_output_{args.scale}"
    run([sys.executable, "autocoder1.py", str(args.scale)], cwd=REPO_DIR, timeout=args.generate_timeout)
    if generated.exists():
        shutil.rmtree(generated)
    shutil.copytree(REPO_DIR / "output", generated)
    return generated.resolve()


def wait_for_ssh(args: argparse.Namespace, attempts: int = 72, delay: int = 5) -> None:
    for index in range(1, attempts + 1):
        proc = ssh(args, "echo ready", capture=True, timeout=10, check=False)
        if proc.returncode == 0 and "ready" in (proc.stdout or ""):
            return
        log(f"waiting for VM SSH {index}/{attempts}")
        time.sleep(delay)
    raise RuntimeError("VM SSH did not become ready")


def host_has_seedemu_images() -> bool:
    proc = run(
        ["docker", "image", "inspect", "handsonsecurity/seedemu-base:2.0", "handsonsecurity/seedemu-router:2.0"],
        capture=True,
        check=False,
    )
    return proc.returncode == 0


def reset_vm(args: argparse.Namespace, summary: Summary, stage_name: str) -> None:
    started = time.monotonic()
    run(["virsh", "destroy", args.domain], check=False)
    run(["virsh", "start", args.domain])
    wait_for_ssh(args)
    reset_cmd = "/home/lxl/bird/reset_docker_vm.sh"
    if args.skip_seedemu_rebuild:
        reset_cmd += " --skip-rebuild-seedemu-images"
    ssh(args, reset_cmd, timeout=args.reset_timeout)

    imported = False
    if args.import_seedemu_images and host_has_seedemu_images():
        run(
            [
                "bash",
                "-lc",
                (
                    "docker save handsonsecurity/seedemu-base:2.0 handsonsecurity/seedemu-router:2.0 "
                    f"| ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR {q(args.vm_user + '@' + args.vm)} "
                    "'docker load >/dev/null'"
                ),
            ],
            timeout=args.import_timeout,
        )
        ssh(
            args,
            "docker tag handsonsecurity/seedemu-base:2.0 handsonsecurity/seedemu-base:latest || true; "
            "docker tag handsonsecurity/seedemu-router:2.0 handsonsecurity/seedemu-router:latest || true",
        )
        imported = True

    verify = ssh(
        args,
        "printf 'containers='; docker ps -aq | wc -l; "
        "printf 'volumes='; docker volume ls -q | wc -l; "
        "printf 'custom_networks='; docker network ls --filter type=custom -q | wc -l; "
        "docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep 'handsonsecurity/seedemu-\\(base\\|router\\)' || true",
        capture=True,
    )
    summary.stage(stage_name, True, duration_seconds=round(time.monotonic() - started, 3), imported_seedemu_images=imported, verify=verify.stdout)


def run_setup(args: argparse.Namespace, output_dir: Path, run_dir: Path, arm: str, project: str, enable_delta_after_initial: bool) -> dict:
    arm_dir = run_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    summary_path = arm_dir / "setup_summary.json"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_bird_delta_experiment.py"),
        str(output_dir),
        "--vm",
        args.vm,
        "--vm-user",
        args.vm_user,
        "--project-name",
        project,
        "--build-mode",
        "delta-fast",
        "--defer-delta",
        "--restart-jobs",
        "1",
        "--restart-delay",
        str(args.restart_delay),
        "--load-only-stable",
        "--load-threshold",
        str(args.load_threshold),
        "--max-wait-seconds",
        str(args.max_wait_seconds),
        "--poll-interval",
        str(args.poll_interval),
        "--write-jobs",
        str(args.initial_write_jobs),
        "--initial-source",
        "birdc-primary",
        "--initial-route-action",
        "replace",
        "--skip-start-agents",
        "--skip-final-validate",
        "--summary-path",
        str(summary_path),
    ]
    if not enable_delta_after_initial:
        cmd.append("--skip-enable-delta-after-initial")
    run(cmd, cwd=REPO_DIR, timeout=args.setup_timeout)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not data.get("success"):
        raise RuntimeError(f"{arm} setup failed: {data.get('error')}")
    return data


def remote_python(args: argparse.Namespace, script: str, argv: list[str], timeout: float = 300) -> dict:
    remote = "python3 - " + " ".join(q(item) for item in argv)
    proc = ssh(args, remote, input_text=script, capture=True, timeout=timeout)
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"remote python did not return JSON: {proc.stdout}") from exc


def choose_victims(args: argparse.Namespace, project: str, count: int) -> list[str]:
    script = r"""
import json
import re
import subprocess
import sys

project = sys.argv[1]
count = int(sys.argv[2])
as_re = re.compile(r"^as(\d+)brd-r")
ps = subprocess.check_output(
    ["docker", "ps", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.Names}}"],
    text=True,
)
names = sorted(line.strip() for line in ps.splitlines() if line.strip())
seen = set()
victims = []
brdnodes = [name for name in names if as_re.match(name)]
for name in names:
    match = as_re.match(name)
    if not match:
        continue
    asn = match.group(1)
    if asn in seen:
        continue
    victims.append(name)
    seen.add(asn)
    if len(victims) >= count:
        break
selected = set(victims)
for name in brdnodes:
    if len(victims) >= count:
        break
    if name in selected:
        continue
    victims.append(name)
    selected.add(name)
print(json.dumps({
    "victims": victims,
    "running_brdnodes": len(brdnodes),
    "unique_as_selected": len(seen),
    "requested": count,
}, indent=2))
"""
    result = remote_python(args, script, [project, str(count)], timeout=120)
    victims = result.get("victims") or []
    if len(victims) < count:
        raise RuntimeError(f"could only choose {len(victims)} victims from {result.get('running_brdnodes')} brdnodes")
    return victims


def stop_victims(args: argparse.Namespace, victims: list[str]) -> dict:
    script = r"""
import json
import subprocess
import sys

victims = sys.argv[1:]
rows = []
for name in victims:
    proc = subprocess.run(["docker", "stop", "-t", "1", name], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    rows.append({"name": name, "returncode": proc.returncode, "output": (proc.stdout or "").strip()[-500:]})
print(json.dumps({"victims": victims, "rows": rows, "success": all(row["returncode"] == 0 for row in rows)}, indent=2))
"""
    result = remote_python(args, script, victims, timeout=max(120, 15 * len(victims)))
    if not result.get("success"):
        raise RuntimeError(f"stopping victims failed: {result}")
    return result


def checkpoint_delta(args: argparse.Namespace, project: str, checkpoint_path: str) -> dict:
    script = r"""
import json
import subprocess
import sys

project = sys.argv[1]
checkpoint = sys.argv[2]
ps = subprocess.check_output(
    ["docker", "ps", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.ID}} {{.Names}}"],
    text=True,
)
containers = [line.split()[0] for line in ps.splitlines() if len(line.split()) >= 2 and line.split()[1].startswith("as") and "brd-r" in line.split()[1]]
sizes = []
failed = []
for cid in containers:
    shell = f'mkdir -p "$(dirname {checkpoint})"; size="$(stat -c %s /run/bird/delta-events.tsv 2>/dev/null || echo 0)"; printf "%s\n" "$size" > {checkpoint}; printf "%s\n" "$size"'
    proc = subprocess.run(["docker", "exec", cid, "sh", "-lc", shell], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    if proc.returncode == 0 and proc.stdout.strip().isdigit():
        sizes.append(int(proc.stdout.strip()))
    else:
        failed.append({"cid": cid, "returncode": proc.returncode, "output": (proc.stdout or "").strip()[-300:]})
print(json.dumps({
    "checkpoint": checkpoint,
    "containers": len(containers),
    "successes": len(sizes),
    "failures": len(failed),
    "failed_samples": failed[:10],
    "min_offset": min(sizes) if sizes else 0,
    "max_offset": max(sizes) if sizes else 0,
    "sum_offsets": sum(sizes),
}, indent=2))
"""
    result = remote_python(args, script, [project, checkpoint_path], timeout=600)
    if result.get("failures"):
        raise RuntimeError(f"checkpoint failed: {result}")
    return result


def read_event_sizes(args: argparse.Namespace, project: str) -> dict:
    script = r"""
import json
import subprocess
import sys

project = sys.argv[1]
ps = subprocess.check_output(
    ["docker", "ps", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.ID}} {{.Names}}"],
    text=True,
)
containers = [line.split()[0] for line in ps.splitlines() if len(line.split()) >= 2 and line.split()[1].startswith("as") and "brd-r" in line.split()[1]]
sizes = []
failed = 0
for cid in containers:
    proc = subprocess.run(
        ["docker", "exec", cid, "sh", "-lc", "stat -c %s /run/bird/delta-events.tsv 2>/dev/null || echo 0"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
    )
    if proc.returncode == 0 and proc.stdout.strip().isdigit():
        sizes.append(int(proc.stdout.strip()))
    else:
        failed += 1
print(json.dumps({
    "containers": len(containers),
    "successes": len(sizes),
    "failures": failed,
    "sum_offsets": sum(sizes),
    "min_offset": min(sizes) if sizes else 0,
    "max_offset": max(sizes) if sizes else 0,
}, indent=2))
"""
    return remote_python(args, script, [project], timeout=600)


def wait_event_sizes_stable(args: argparse.Namespace, summary: Summary, stage_name: str, project: str) -> dict:
    started = time.monotonic()
    last_sum: int | None = None
    stable_hits = 0
    samples = []
    while time.monotonic() - started < args.max_wait_seconds:
        sample = read_event_sizes(args, project)
        sample["at"] = now_iso()
        samples.append(sample)
        if last_sum is not None and sample.get("sum_offsets") == last_sum and sample.get("failures") == 0:
            stable_hits += 1
        else:
            stable_hits = 0
        last_sum = int(sample.get("sum_offsets") or 0)
        summary.stage(
            stage_name,
            False,
            stable_hits=stable_hits,
            required_stable_hits=args.delta_event_stable_samples,
            samples_tail=samples[-10:],
            duration_seconds=round(time.monotonic() - started, 3),
        )
        log(
            f"{stage_name}: event_bytes_sum={sample.get('sum_offsets')} "
            f"stable_hits={stable_hits}/{args.delta_event_stable_samples}"
        )
        if stable_hits >= args.delta_event_stable_samples:
            result = {
                "duration_seconds": round(time.monotonic() - started, 3),
                "stable_hits": stable_hits,
                "samples_tail": samples[-10:],
            }
            summary.stage(stage_name, True, **result)
            return result
        time.sleep(args.delta_event_stable_interval)
    raise RuntimeError(f"{stage_name} timed out waiting for stable delta event files")


def wait_load_below(args: argparse.Namespace, summary: Summary, stage_name: str) -> dict:
    started = time.monotonic()
    deadline = started + args.max_wait_seconds
    samples = []
    while time.monotonic() < deadline:
        proc = ssh(args, "awk '{print $1}' /proc/loadavg", capture=True, timeout=60)
        load = float((proc.stdout or "999").strip().split()[0])
        sample = {"at": now_iso(), "load1": load}
        samples.append(sample)
        summary.stage(stage_name, False, latest=sample, samples_tail=samples[-20:], duration_seconds=round(time.monotonic() - started, 3))
        log(f"{stage_name}: load1={load:.2f}, threshold={args.load_threshold:.2f}")
        if load < args.load_threshold:
            result = {"final_load1": load, "samples_tail": samples[-20:], "duration_seconds": round(time.monotonic() - started, 3)}
            summary.stage(stage_name, True, **result)
            return result
        time.sleep(args.poll_interval)
    raise RuntimeError(f"{stage_name} timed out waiting for load < {args.load_threshold}")


def fetch_remote_file(args: argparse.Namespace, remote_path: Path, local_path: Path, required: bool = True) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = rsync_cmd(args.vm_user, args.vm) + [f"{args.vm_user}@{args.vm}:{remote_path}", str(local_path)]
    run(cmd, check=required)


def run_host_writer(
    args: argparse.Namespace,
    project: str,
    remote_asset_dir: Path,
    local_stage_dir: Path,
    remote_stage_dir: Path,
    *,
    source: str,
    jobs: int,
    dump_only: bool = False,
    commands_dir: Path | None = None,
    offset_mark: str | None = None,
    batch_lines: int | None = None,
    batch_sleep_ms: float | None = None,
) -> dict:
    local_stage_dir.mkdir(parents=True, exist_ok=True)
    remote_summary = remote_stage_dir / f"{source}_{'dump' if dump_only else 'write'}_summary.json"
    remote_details = remote_stage_dir / f"{source}_{'dump' if dump_only else 'write'}_details.json"
    remote_cmd = [
        "sudo",
        "-n",
        "python3",
        str(remote_asset_dir / "delta_fib_host_writer.py"),
        "--project-name",
        project,
        "--metric",
        str(args.metric),
        "--source",
        source,
        "--jobs",
        str(jobs),
        "--executor",
        "process",
        "--write-mode",
        "setns-ip",
        "--command-timeout",
        str(args.command_timeout),
        "--summary-path",
        str(remote_summary),
        "--details-path",
        str(remote_details),
    ]
    if dump_only:
        remote_cmd.append("--dump-only")
    if commands_dir:
        remote_cmd.extend(["--commands-dir", str(commands_dir)])
    if offset_mark:
        remote_cmd.extend(["--offset-mark", offset_mark])
    if batch_lines is not None:
        remote_cmd.extend(["--batch-lines", str(batch_lines)])
    if batch_sleep_ms is not None:
        remote_cmd.extend(["--batch-sleep-ms", str(batch_sleep_ms)])

    ssh(args, "mkdir -p " + q(remote_stage_dir), timeout=60)
    ssh(args, " ".join(q(part) for part in remote_cmd), timeout=args.writer_timeout)
    local_summary = local_stage_dir / remote_summary.name
    local_details = local_stage_dir / remote_details.name
    fetch_remote_file(args, remote_summary, local_summary)
    fetch_remote_file(args, remote_details, local_details, required=False)
    result = json.loads(local_summary.read_text(encoding="utf-8"))
    if result.get("failures") or result.get("batch_errors_total"):
        raise RuntimeError(f"writer failed for {source}: {result.get('failures')} failures, {result.get('batch_errors_total')} batch errors")
    return result


def run_arm(
    args: argparse.Namespace,
    output_dir: Path,
    run_dir: Path,
    summary: Summary,
    *,
    arm: str,
    project: str,
    enable_delta_after_initial: bool,
    victims: list[str] | None = None,
) -> tuple[list[str], dict]:
    reset_vm(args, summary, f"{arm}_reset_vm")
    setup = run_setup(args, output_dir, run_dir, arm, project, enable_delta_after_initial)
    remote_run_dir = Path(setup["remote_run_dir"])
    remote_asset_dir = Path(setup["remote_run_dir"]) / "delta-assets"
    remote_stage_dir = remote_run_dir / f"strict_{arm}"
    arm_dir = run_dir / arm
    summary.stage(f"{arm}_setup", True, summary_path=str(arm_dir / "setup_summary.json"), remote_run_dir=str(remote_run_dir))

    checkpoint = None
    if enable_delta_after_initial:
        if args.post_enable_delta_sleep > 0:
            time.sleep(args.post_enable_delta_sleep)
        event_stable = wait_event_sizes_stable(args, summary, f"{arm}_delta_events_initial_stable", project)
        summary.stage(f"{arm}_delta_events_initial_stable_done", True, **event_stable)
        checkpoint = checkpoint_delta(args, project, args.checkpoint_path)
        summary.stage(f"{arm}_checkpoint", True, **checkpoint)

    selected_victims = victims or choose_victims(args, project, args.down_count)
    (arm_dir / "victims.json").write_text(json.dumps(selected_victims, indent=2) + "\n", encoding="utf-8")
    stopped = stop_victims(args, selected_victims)
    stopped_summary = {key: value for key, value in stopped.items() if key != "success"}
    summary.stage(f"{arm}_stop_victims", True, **stopped_summary)
    wait_load_below(args, summary, f"{arm}_wait_load_after_fault")

    commands_dir = remote_stage_dir / "commands"
    if arm == "delta":
        dump_stats = run_host_writer(
            args,
            project,
            remote_asset_dir,
            arm_dir / "delta_dump",
            remote_stage_dir,
            source="event",
            jobs=args.delta_dump_jobs,
            dump_only=True,
            commands_dir=commands_dir,
            offset_mark=args.checkpoint_path,
        )
        summary.stage("delta_dump_events", True, writer=dump_stats)
        write_stats = run_host_writer(
            args,
            project,
            remote_asset_dir,
            arm_dir / "delta_write",
            remote_stage_dir,
            source="command-files",
            jobs=args.delta_write_jobs,
            commands_dir=commands_dir,
            batch_lines=args.delta_batch_lines,
            batch_sleep_ms=args.delta_batch_sleep_ms,
        )
        summary.stage("delta_write_fib", True, writer=write_stats)
        updated = checkpoint_delta(args, project, args.checkpoint_path)
        summary.stage("delta_checkpoint_after_write", True, **updated)
    else:
        dump_stats = run_host_writer(
            args,
            project,
            remote_asset_dir,
            arm_dir / "full_dump",
            remote_stage_dir,
            source="birdc-primary",
            jobs=args.full_dump_jobs,
            dump_only=True,
            commands_dir=commands_dir,
        )
        summary.stage("full_dump_birdc_primary", True, writer=dump_stats)
        write_stats = run_host_writer(
            args,
            project,
            remote_asset_dir,
            arm_dir / "full_write",
            remote_stage_dir,
            source="command-files",
            jobs=args.full_write_jobs,
            commands_dir=commands_dir,
            batch_lines=args.full_batch_lines,
            batch_sleep_ms=args.full_batch_sleep_ms,
        )
        summary.stage("full_write_fib", True, writer=write_stats)

    return selected_victims, {
        "setup": setup,
        "checkpoint": checkpoint,
        "remote_run_dir": str(remote_run_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--vm", default=DEFAULT_VM)
    parser.add_argument("--vm-user", default=DEFAULT_USER)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--metric", type=int, default=232)
    parser.add_argument("--down-count", type=int, default=10)
    parser.add_argument("--load-threshold", type=float, default=40.0)
    parser.add_argument("--max-wait-seconds", type=int, default=7200)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--restart-delay", type=float, default=0.1)
    parser.add_argument("--initial-write-jobs", type=int, default=8)
    parser.add_argument("--delta-dump-jobs", type=int, default=8)
    parser.add_argument("--delta-write-jobs", type=int, default=4)
    parser.add_argument("--full-dump-jobs", type=int, default=4)
    parser.add_argument("--full-write-jobs", type=int, default=8)
    parser.add_argument("--delta-batch-lines", type=int, default=1000)
    parser.add_argument("--delta-batch-sleep-ms", type=float, default=2.0)
    parser.add_argument("--full-batch-lines", type=int, default=1000)
    parser.add_argument("--full-batch-sleep-ms", type=float, default=5.0)
    parser.add_argument("--command-timeout", type=float, default=600)
    parser.add_argument("--setup-timeout", type=float, default=14400)
    parser.add_argument("--writer-timeout", type=float, default=7200)
    parser.add_argument("--reset-timeout", type=float, default=1800)
    parser.add_argument("--import-timeout", type=float, default=1800)
    parser.add_argument("--generate-timeout", type=float, default=1800)
    parser.add_argument("--post-enable-delta-sleep", type=float, default=5.0)
    parser.add_argument("--delta-event-stable-samples", type=int, default=2)
    parser.add_argument("--delta-event-stable-interval", type=float, default=10.0)
    parser.add_argument("--checkpoint-path", default="/run/bird/delta-fib-checkpoint")
    parser.add_argument("--skip-seedemu-rebuild", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--import-seedemu-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-full-arm", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (args.run_dir or (SCRIPT_DIR / "runs" / f"strict_delta_full_{args.scale}_{timestamp}")).expanduser().resolve()
    summary = Summary(
        run_dir / "summary.json",
        {
            "mode": "strict-post-fault-delta-vs-full",
            "command": sys.argv,
            "host": socket.gethostname(),
            "user": getpass.getuser(),
            "scale": args.scale,
            "vm": args.vm,
            "domain": args.domain,
            "settings": jsonable_settings(args),
        },
    )
    error = None
    try:
        output_dir = ensure_output_dir(args, run_dir)
        service_count = compose_service_count(output_dir)
        summary.update(output_dir=str(output_dir), topology_services=service_count, run_dir=str(run_dir))
        delta_project = f"topology-{args.scale}-delta"
        full_project = f"topology-{args.scale}-full"
        victims, delta_info = run_arm(
            args,
            output_dir,
            run_dir,
            summary,
            arm="delta",
            project=delta_project,
            enable_delta_after_initial=True,
        )
        summary.update(victims=victims, delta=delta_info)
        if not args.skip_full_arm:
            _, full_info = run_arm(
                args,
                output_dir,
                run_dir,
                summary,
                arm="full",
                project=full_project,
                enable_delta_after_initial=False,
                victims=victims,
            )
            summary.update(full=full_info)
    except Exception as exc:
        error = str(exc)
        log(f"ERROR: {error}")
    finally:
        summary.finalize(error is None, error)
    return 0 if error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
