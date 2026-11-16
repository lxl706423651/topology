#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VM_HOST="${VM_HOST:-192.168.122.60}"
VM_USER="${VM_USER:-lxl}"
PROJECT_NAME="${PROJECT_NAME:-topology-291e63c1}"
VM_RUN_ROOT="${VM_RUN_ROOT:-/home/lxl/bird/runs}"
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT:-${SCRIPT_DIR}/runs}"
RUN_NAME="${RUN_NAME:-}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-}"

DOWN_COUNT="${DOWN_COUNT:-10}"
STOP_TIMEOUT="${STOP_TIMEOUT:-1}"
SELECT_STRATEGY="${SELECT_STRATEGY:-spread}"
METRIC="${METRIC:-232}"
LOAD_THRESHOLD="${LOAD_THRESHOLD:-50}"
LOAD_ONLY_STABLE="${LOAD_ONLY_STABLE:-0}"
STABLE_SECONDS="${STABLE_SECONDS:-120}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-1800}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"
STABILITY_SAMPLES="${STABILITY_SAMPLES:-20}"
VALIDATE_SAMPLES="${VALIDATE_SAMPLES:-10}"
INSTALL_JOBS="${INSTALL_JOBS:-16}"
WRITE_JOBS="${WRITE_JOBS:-8}"
OFFSET_JOBS="${OFFSET_JOBS:-${WRITE_JOBS}}"
FULL_WRITE_JOBS="${FULL_WRITE_JOBS:-${WRITE_JOBS}}"
RUN_FULL_COMPARISON="${RUN_FULL_COMPARISON:-0}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-180}"
FULL_COMMAND_TIMEOUT="${FULL_COMMAND_TIMEOUT:-1800}"
WRITE_BACKEND="${WRITE_BACKEND:-host-setns}"
SKIP_AGENT_INSTALL="${SKIP_AGENT_INSTALL:-0}"
USE_OFFSET_MARK="${USE_OFFSET_MARK:-1}"
K1_DELTA_JOBS="${K1_DELTA_JOBS:-${WRITE_JOBS}}"
K1_DELTA_BATCH="${K1_DELTA_BATCH:-2000}"
K1_OPS_PER_ATTR="${K1_OPS_PER_ATTR:-2000}"
K1_MULTIPATH_POLICY="${K1_MULTIPATH_POLICY:-first}"
K1_DELTA_MIXED_BATCHES="${K1_DELTA_MIXED_BATCHES:-0}"
K3A_DELTA_JOBS="${K3A_DELTA_JOBS:-${WRITE_JOBS}}"
K3A_BATCH="${K3A_BATCH:-512}"
K3A_ACK_MODE="${K3A_ACK_MODE:-each}"
K3A_MULTIPATH_POLICY="${K3A_MULTIPATH_POLICY:-first}"
MARK_OFFSETS_BEFORE_FAILURE="${MARK_OFFSETS_BEFORE_FAILURE:-0}"

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ConnectTimeout=10
)

usage() {
  cat <<EOF
Usage:
  $0 [run]
  $0 status

This is the source-BIRD-delta reconvergence experiment. It does not dump the
whole BIRD RIB. By default it expects BIRD delta to ignore the initial feed, so
after brdnodes stop and RIB/load stabilizes it reads event-file route changes
from offset 0. Set MARK_OFFSETS_BEFORE_FAILURE=1 only for legacy runs that need
the old pre-failure offset mark behavior.

Environment overrides:
  VM_HOST=${VM_HOST}
  VM_USER=${VM_USER}
  PROJECT_NAME=${PROJECT_NAME}
  DOWN_COUNT=${DOWN_COUNT}
  STABLE_SECONDS=${STABLE_SECONDS}
  LOAD_THRESHOLD=${LOAD_THRESHOLD}
  LOAD_ONLY_STABLE=${LOAD_ONLY_STABLE}
  WRITE_JOBS=${WRITE_JOBS}
  OFFSET_JOBS=${OFFSET_JOBS}
  WRITE_BACKEND=${WRITE_BACKEND}
  SKIP_AGENT_INSTALL=${SKIP_AGENT_INSTALL}
  USE_OFFSET_MARK=${USE_OFFSET_MARK}
  K1_DELTA_JOBS=${K1_DELTA_JOBS}
  K1_DELTA_BATCH=${K1_DELTA_BATCH}
  K1_OPS_PER_ATTR=${K1_OPS_PER_ATTR}
  K1_MULTIPATH_POLICY=${K1_MULTIPATH_POLICY}
  K1_DELTA_MIXED_BATCHES=${K1_DELTA_MIXED_BATCHES}
  K3A_DELTA_JOBS=${K3A_DELTA_JOBS}
  K3A_BATCH=${K3A_BATCH}
  K3A_ACK_MODE=${K3A_ACK_MODE}
  K3A_MULTIPATH_POLICY=${K3A_MULTIPATH_POLICY}
  MARK_OFFSETS_BEFORE_FAILURE=${MARK_OFFSETS_BEFORE_FAILURE}
  FULL_WRITE_JOBS=${FULL_WRITE_JOBS}
  RUN_FULL_COMPARISON=${RUN_FULL_COMPARISON}
  FULL_COMMAND_TIMEOUT=${FULL_COMMAND_TIMEOUT}
EOF
}

ssh_vm() {
  ssh "${SSH_OPTS[@]}" "${VM_USER}@${VM_HOST}" "$@"
}

rsync_from_vm() {
  rsync -a -e "ssh ${SSH_OPTS[*]}" "${VM_USER}@${VM_HOST}:$1" "$2"
}

rsync_to_vm() {
  rsync -a -e "ssh ${SSH_OPTS[*]}" "$1" "${VM_USER}@${VM_HOST}:$2"
}

stamp() {
  date +%Y%m%d_%H%M%S
}

phase_status() {
  ssh_vm 'date -Is; cat /proc/loadavg; printf "project_running="; docker ps --filter label=com.docker.compose.project='"${PROJECT_NAME}"' -q | wc -l; printf "brd_running="; docker ps --filter label=com.docker.compose.project='"${PROJECT_NAME}"' --format "{{.Names}}" | grep -Ec "^as[0-9]+brd-r" || true; printf "agents="; pgrep -af delta_fib_agent.py | wc -l'
}

phase_run() {
  local run_stamp remote_dir local_dir
  run_stamp="$(stamp)"
  if [ -n "${RUN_NAME}" ]; then
    default_name="${RUN_NAME}"
  else
    default_name="reconvergence-bird-delta_${run_stamp}_${PROJECT_NAME}"
  fi
  remote_dir="${REMOTE_RUN_DIR:-${VM_RUN_ROOT}/${default_name}}"
  local_dir="${LOCAL_RUN_DIR:-${LOCAL_RUN_ROOT}/${default_name}}"
  mkdir -p "${local_dir}"
  ssh_vm "mkdir -p '${remote_dir}'"
  ssh_vm "mkdir -p /home/${VM_USER}/bird/delta-assets"
  rsync_to_vm "${SCRIPT_DIR}/delta_fib_agent.py" "/home/${VM_USER}/bird/delta-assets/delta_fib_agent.py"
  rsync_to_vm "${SCRIPT_DIR}/delta_fib_host_writer.py" "/home/${VM_USER}/bird/delta-assets/delta_fib_host_writer.py"
  rsync_to_vm "${SCRIPT_DIR}/kernel/tools/kernel_bulk_writer.py" "/home/${VM_USER}/bird/delta-assets/kernel_bulk_writer.py"
  rsync_to_vm "${SCRIPT_DIR}/kernel/tools/kernel_bulk_command_writer.py" "/home/${VM_USER}/bird/delta-assets/kernel_bulk_command_writer.py"
  rsync_to_vm "${SCRIPT_DIR}/kernel/tools/rtnetlink_route_writer.py" "/home/${VM_USER}/bird/delta-assets/rtnetlink_route_writer.py"

  if [ "${SKIP_AGENT_INSTALL}" = "1" ]; then
    echo "install-agent: skipped by SKIP_AGENT_INSTALL=1"
    echo "stop-agent: skipped by SKIP_AGENT_INSTALL=1"
  else
    VM_HOST="${VM_HOST}" VM_USER="${VM_USER}" PROJECT_NAME="${PROJECT_NAME}" DELTA_JOBS="${INSTALL_JOBS}" \
      "${SCRIPT_DIR}/install_bird_delta_vm.sh" install-agent
    VM_HOST="${VM_HOST}" VM_USER="${VM_USER}" PROJECT_NAME="${PROJECT_NAME}" DELTA_JOBS="${INSTALL_JOBS}" \
      "${SCRIPT_DIR}/install_bird_delta_vm.sh" stop-agent
  fi

  local quoted_project quoted_dir
  quoted_project="$(printf '%q' "${PROJECT_NAME}")"
  quoted_dir="$(printf '%q' "${remote_dir}")"

  set +e
  ssh_vm "PROJECT_NAME=${quoted_project} RUN_DIR=${quoted_dir} DOWN_COUNT=${DOWN_COUNT} STOP_TIMEOUT=${STOP_TIMEOUT} SELECT_STRATEGY=${SELECT_STRATEGY} METRIC=${METRIC} LOAD_THRESHOLD=${LOAD_THRESHOLD} LOAD_ONLY_STABLE=${LOAD_ONLY_STABLE} STABLE_SECONDS=${STABLE_SECONDS} MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS} POLL_INTERVAL=${POLL_INTERVAL} STABILITY_SAMPLES=${STABILITY_SAMPLES} VALIDATE_SAMPLES=${VALIDATE_SAMPLES} WRITE_JOBS=${WRITE_JOBS} OFFSET_JOBS=${OFFSET_JOBS} WRITE_BACKEND=${WRITE_BACKEND} USE_OFFSET_MARK=${USE_OFFSET_MARK} K1_DELTA_JOBS=${K1_DELTA_JOBS} K1_DELTA_BATCH=${K1_DELTA_BATCH} K1_OPS_PER_ATTR=${K1_OPS_PER_ATTR} K1_MULTIPATH_POLICY=${K1_MULTIPATH_POLICY} K1_DELTA_MIXED_BATCHES=${K1_DELTA_MIXED_BATCHES} K3A_DELTA_JOBS=${K3A_DELTA_JOBS} K3A_BATCH=${K3A_BATCH} K3A_ACK_MODE=${K3A_ACK_MODE} K3A_MULTIPATH_POLICY=${K3A_MULTIPATH_POLICY} MARK_OFFSETS_BEFORE_FAILURE=${MARK_OFFSETS_BEFORE_FAILURE} FULL_WRITE_JOBS=${FULL_WRITE_JOBS} RUN_FULL_COMPARISON=${RUN_FULL_COMPARISON} COMMAND_TIMEOUT=${COMMAND_TIMEOUT} FULL_COMMAND_TIMEOUT=${FULL_COMMAND_TIMEOUT} python3 -" <<'PY'
import concurrent.futures
import datetime as dt
import json
import os
import re
import signal
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

PROJECT = os.environ["PROJECT_NAME"]
RUN_DIR = Path(os.environ["RUN_DIR"])
DOWN_COUNT = int(os.environ["DOWN_COUNT"])
STOP_TIMEOUT = int(os.environ["STOP_TIMEOUT"])
SELECT_STRATEGY = os.environ["SELECT_STRATEGY"]
METRIC = int(os.environ["METRIC"])
LOAD_THRESHOLD = float(os.environ["LOAD_THRESHOLD"])
LOAD_ONLY_STABLE = os.environ.get("LOAD_ONLY_STABLE", "0") == "1"
STABLE_SECONDS = float(os.environ["STABLE_SECONDS"])
MAX_WAIT_SECONDS = float(os.environ["MAX_WAIT_SECONDS"])
POLL_INTERVAL = float(os.environ["POLL_INTERVAL"])
STABILITY_SAMPLES = int(os.environ["STABILITY_SAMPLES"])
VALIDATE_SAMPLES = int(os.environ["VALIDATE_SAMPLES"])
WRITE_JOBS = int(os.environ["WRITE_JOBS"])
OFFSET_JOBS = int(os.environ.get("OFFSET_JOBS", str(WRITE_JOBS)))
WRITE_BACKEND = os.environ.get("WRITE_BACKEND", "host-setns")
USE_OFFSET_MARK = os.environ.get("USE_OFFSET_MARK", "1") == "1"
K1_DELTA_JOBS = int(os.environ.get("K1_DELTA_JOBS", str(WRITE_JOBS)))
K1_DELTA_BATCH = int(os.environ.get("K1_DELTA_BATCH", "2000"))
K1_OPS_PER_ATTR = int(os.environ.get("K1_OPS_PER_ATTR", "2000"))
K1_MULTIPATH_POLICY = os.environ.get("K1_MULTIPATH_POLICY", "first")
K1_DELTA_MIXED_BATCHES = os.environ.get("K1_DELTA_MIXED_BATCHES", "0") == "1"
K3A_DELTA_JOBS = int(os.environ.get("K3A_DELTA_JOBS", str(WRITE_JOBS)))
K3A_BATCH = int(os.environ.get("K3A_BATCH", "512"))
K3A_ACK_MODE = os.environ.get("K3A_ACK_MODE", "each")
K3A_MULTIPATH_POLICY = os.environ.get("K3A_MULTIPATH_POLICY", "first")
MARK_OFFSETS_BEFORE_FAILURE = os.environ.get("MARK_OFFSETS_BEFORE_FAILURE", "0") == "1"
FULL_WRITE_JOBS = int(os.environ["FULL_WRITE_JOBS"])
RUN_FULL_COMPARISON = os.environ.get("RUN_FULL_COMPARISON", "0") == "1"
COMMAND_TIMEOUT = float(os.environ["COMMAND_TIMEOUT"])
FULL_COMMAND_TIMEOUT = float(os.environ["FULL_COMMAND_TIMEOUT"])

ROUTER_RE = re.compile(r"^as(\d+)brd-r")
SUMMARY_PATH = RUN_DIR / "summary.json"


class LabError(RuntimeError):
    pass


def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


summary = {
    "schema_version": 1,
    "mode": "bird-source-delta-reconvergence",
    "project": PROJECT,
    "run_dir": str(RUN_DIR),
    "settings": {
        "down_count": DOWN_COUNT,
        "metric": METRIC,
        "load_threshold": LOAD_THRESHOLD,
        "load_only_stable": LOAD_ONLY_STABLE,
        "stable_seconds": STABLE_SECONDS,
        "write_jobs": WRITE_JOBS,
        "offset_jobs": OFFSET_JOBS,
        "write_backend": WRITE_BACKEND,
        "use_offset_mark": USE_OFFSET_MARK,
        "k1_delta_jobs": K1_DELTA_JOBS,
        "k1_delta_batch": K1_DELTA_BATCH,
        "k1_ops_per_attr": K1_OPS_PER_ATTR,
        "k1_multipath_policy": K1_MULTIPATH_POLICY,
        "k1_delta_mixed_batches": K1_DELTA_MIXED_BATCHES,
        "k3a_delta_jobs": K3A_DELTA_JOBS,
        "k3a_batch": K3A_BATCH,
        "k3a_ack_mode": K3A_ACK_MODE,
        "k3a_multipath_policy": K3A_MULTIPATH_POLICY,
        "mark_offsets_before_failure": MARK_OFFSETS_BEFORE_FAILURE,
        "full_write_jobs": FULL_WRITE_JOBS,
        "run_full_comparison": RUN_FULL_COMPARISON,
        "full_command_timeout": FULL_COMMAND_TIMEOUT,
    },
    "started_at": now_iso(),
    "ended_at": None,
    "duration_seconds": None,
    "success": None,
    "error": None,
    "stages": {},
}
started = time.monotonic()


def write_summary():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SUMMARY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(SUMMARY_PATH)


def stage_start(name, **info):
    summary["stages"][name] = {"started_at": now_iso(), "success": None, **info}
    summary["stages"][name]["_t0"] = time.monotonic()
    write_summary()


def stage_finish(name, success=True, **info):
    stage = summary["stages"].setdefault(name, {})
    t0 = stage.pop("_t0", None)
    if t0 is not None:
        stage["duration_seconds"] = round(time.monotonic() - t0, 3)
    stage.update(info)
    stage["success"] = success
    stage["ended_at"] = now_iso()
    write_summary()


def run_cmd(cmd, *, input_text=None, timeout=120, check=True):
    proc_obj = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc_obj.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc_obj.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(2)
        if proc_obj.poll() is None:
            try:
                os.killpg(proc_obj.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        stdout, stderr = proc_obj.communicate()
    proc = subprocess.CompletedProcess(
        cmd,
        124 if timed_out else proc_obj.returncode,
        stdout,
        (stderr or "") + (f"\nTIMEOUT after {timeout}s; killed process group {proc_obj.pid}" if timed_out else ""),
    )
    if check and proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip() or "command failed"
        raise LabError(f"{' '.join(cmd)}: {msg}")
    return proc


def docker_exec(name, shell, timeout=COMMAND_TIMEOUT, check=True):
    return run_cmd(["docker", "exec", name, "sh", "-lc", shell], timeout=timeout, check=check)


def running_brdnodes():
    proc = run_cmd([
        "docker", "ps",
        "--filter", f"label=com.docker.compose.project={PROJECT}",
        "--format", "{{.Names}}",
    ])
    nodes = []
    for line in proc.stdout.splitlines():
        name = line.strip()
        match = ROUTER_RE.match(name)
        if match:
            nodes.append({"name": name, "as": int(match.group(1))})
    nodes.sort(key=lambda item: item["name"])
    return nodes


def choose_victims(nodes):
    groups = defaultdict(list)
    for node in nodes:
        groups[node["as"]].append(node["name"])
    asns = sorted(groups)
    if DOWN_COUNT > len(asns):
        raise LabError(f"requested {DOWN_COUNT} ASes but only {len(asns)} running AS groups exist")
    if SELECT_STRATEGY == "first":
        chosen = asns[:DOWN_COUNT]
    elif SELECT_STRATEGY == "spread":
        chosen = []
        used = set()
        for idx in range(DOWN_COUNT):
            pos = 0 if DOWN_COUNT == 1 else round(idx * (len(asns) - 1) / (DOWN_COUNT - 1))
            while pos in used and pos + 1 < len(asns):
                pos += 1
            while pos in used and pos - 1 >= 0:
                pos -= 1
            used.add(pos)
            chosen.append(asns[pos])
    else:
        raise LabError(f"unsupported SELECT_STRATEGY={SELECT_STRATEGY!r}")
    return [{"as": asn, "name": sorted(groups[asn])[0]} for asn in chosen]


def map_nodes(nodes, fn, workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fn, nodes))


def mark_offset(node):
    start = time.monotonic()
    proc = docker_exec(
        node["name"],
        "mkdir -p /run/bird; stat -c %s /run/bird/delta-events.tsv > /run/bird/delta-offset-before-reconv; cat /run/bird/delta-offset-before-reconv",
        timeout=20,
        check=False,
    )
    return {
        "name": node["name"],
        "success": proc.returncode == 0,
        "offset": int(proc.stdout.strip() or 0) if proc.returncode == 0 else None,
        "duration_seconds": round(time.monotonic() - start, 6),
        "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip())[-1000:],
    }


def set_zero_offset(node):
    start = time.monotonic()
    proc = docker_exec(
        node["name"],
        "mkdir -p /run/bird; : > /run/bird/delta-offset-before-reconv; printf 0 > /run/bird/delta-offset-before-reconv; cat /run/bird/delta-offset-before-reconv",
        timeout=20,
        check=False,
    )
    return {
        "name": node["name"],
        "success": proc.returncode == 0,
        "offset": int(proc.stdout.strip() or 0) if proc.returncode == 0 else None,
        "duration_seconds": round(time.monotonic() - start, 6),
        "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip())[-1000:],
    }


def stop_victim(victim):
    start = time.monotonic()
    proc = run_cmd(["docker", "stop", "-t", str(STOP_TIMEOUT), victim["name"]], timeout=60, check=False)
    return {
        "as": victim["as"],
        "name": victim["name"],
        "success": proc.returncode == 0,
        "duration_seconds": round(time.monotonic() - start, 6),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def route_count(name):
    proc = docker_exec(
        name,
        "birdc -s /run/bird/bird.ctl show route primary count 2>/dev/null | awk '/master4/ {print $6; exit}'",
        timeout=20,
        check=False,
    )
    text = proc.stdout.strip()
    return int(text) if proc.returncode == 0 and text.isdigit() else None


def pick_samples(nodes, limit):
    if limit <= 0 or len(nodes) <= limit:
        return nodes
    if limit == 1:
        return [nodes[0]]
    out = []
    used = set()
    for idx in range(limit):
        pos = round(idx * (len(nodes) - 1) / (limit - 1))
        if pos in used:
            continue
        used.add(pos)
        out.append(nodes[pos])
    return out


def wait_stable(nodes):
    samples = pick_samples(nodes, STABILITY_SAMPLES)
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    stable_since = None
    last_counts = None
    history = []
    while time.monotonic() < deadline:
        load1 = os.getloadavg()[0]
        counts = {} if LOAD_ONLY_STABLE else {node["name"]: route_count(node["name"]) for node in samples}
        enough = True if LOAD_ONLY_STABLE else len([v for v in counts.values() if v is not None]) >= min(STABILITY_SAMPLES, len(samples))
        same = None if LOAD_ONLY_STABLE else (last_counts == counts if last_counts is not None else False)
        low = load1 < LOAD_THRESHOLD
        now = time.monotonic()
        item = {
            "at": now_iso(),
            "load1": load1,
            "same_counts": same,
            "low_load": low,
            "sample_count": len(counts),
            "route_count_check_skipped": LOAD_ONLY_STABLE,
            "stable_for_seconds": round(now - stable_since, 1) if stable_since else 0,
        }
        history.append(item)
        summary["stages"]["wait_rib_stable"]["latest"] = item
        summary["stages"]["wait_rib_stable"]["samples_tail"] = history[-10:]
        write_summary()
        if LOAD_ONLY_STABLE and low:
            return {
                "samples_tail": history[-10:],
                "final_load1": load1,
                "route_count_check_skipped": True,
            }
        if enough and same and low:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= STABLE_SECONDS:
                return {"samples_tail": history[-10:], "final_counts": counts, "stable_for_seconds": round(now - stable_since, 1)}
        else:
            stable_since = None
        last_counts = counts
        time.sleep(POLL_INTERVAL)
    raise LabError("RIB/load stability gate timed out")


def run_host_writer(stage_name, source, jobs, extra_args=None, timeout=None):
    writer_summary_path = RUN_DIR / f"{stage_name}_summary.json"
    writer_details_path = RUN_DIR / f"{stage_name}_details.json"
    cmd = [
        "sudo", "-n", "python3", "/home/lxl/bird/delta-assets/delta_fib_host_writer.py",
        "--project-name", PROJECT,
        "--metric", str(METRIC),
        "--jobs", str(jobs),
        "--executor", "process",
        "--write-mode", "setns-ip",
        "--source", source,
        "--summary-path", str(writer_summary_path),
        "--details-path", str(writer_details_path),
    ]
    if extra_args:
        cmd.extend(extra_args)
    proc = run_cmd(cmd, timeout=timeout or max(COMMAND_TIMEOUT, 1800), check=False)
    try:
        writer_summary = json.loads(writer_summary_path.read_text(encoding="utf-8"))
    except Exception:
        writer_summary = {"raw_stdout": proc.stdout[-4000:], "raw_stderr": proc.stderr[-4000:]}
    ok = proc.returncode == 0 and writer_summary.get("failures", 1) == 0 and writer_summary.get("batch_errors_total", 1) == 0
    return ok, writer_summary


def event_offset_args():
    if MARK_OFFSETS_BEFORE_FAILURE or USE_OFFSET_MARK:
        return ["--offset-mark", "/run/bird/delta-offset-before-reconv"]
    return []


def run_k1_delta_writer():
    commands_dir = RUN_DIR / "k1_delta_commands"
    dump_summary_path = RUN_DIR / "k1_delta_command_dump_summary.json"
    dump_details_path = RUN_DIR / "k1_delta_command_dump_details.json"
    k1_summary_path = RUN_DIR / "k1_delta_bulk_writer_summary.json"
    k1_details_path = RUN_DIR / "k1_delta_bulk_writer_details.json"

    dump_cmd = [
        "sudo", "-n", "python3", "/home/lxl/bird/delta-assets/delta_fib_host_writer.py",
        "--project-name", PROJECT,
        "--metric", str(METRIC),
        "--jobs", str(WRITE_JOBS),
        "--executor", "process",
        "--write-mode", "setns-ip",
        "--source", "event",
        "--dump-only",
        "--commands-dir", str(commands_dir),
        "--summary-path", str(dump_summary_path),
        "--details-path", str(dump_details_path),
    ]
    dump_cmd.extend(event_offset_args())
    dump_proc = run_cmd(dump_cmd, timeout=max(COMMAND_TIMEOUT, 1800), check=False)
    try:
        dump_summary = json.loads(dump_summary_path.read_text(encoding="utf-8"))
    except Exception:
        dump_summary = {"raw_stdout": dump_proc.stdout[-4000:], "raw_stderr": dump_proc.stderr[-4000:]}
    dump_ok = dump_proc.returncode == 0 and dump_summary.get("failures", 1) == 0
    if not dump_ok:
        return False, {"dump": dump_summary, "writer": None}

    k1_cmd = [
        "sudo", "-n", "python3", "/home/lxl/bird/delta-assets/kernel_bulk_command_writer.py",
        "--project-name", PROJECT,
        "--commands-dir", str(commands_dir),
        "--metric", str(METRIC),
        "--protocol", "12",
        "--jobs", str(K1_DELTA_JOBS),
        "--batch", str(K1_DELTA_BATCH),
        "--ops-per-attr", str(K1_OPS_PER_ATTR),
        "--multipath-policy", K1_MULTIPATH_POLICY,
        "--ignore-delete-miss",
        "--summary-path", str(k1_summary_path),
        "--details-path", str(k1_details_path),
    ]
    if K1_DELTA_MIXED_BATCHES:
        k1_cmd.append("--mixed-batches")
    k1_proc = run_cmd(k1_cmd, timeout=max(COMMAND_TIMEOUT, 1800), check=False)
    try:
        k1_summary = json.loads(k1_summary_path.read_text(encoding="utf-8"))
    except Exception:
        k1_summary = {"raw_stdout": k1_proc.stdout[-4000:], "raw_stderr": k1_proc.stderr[-4000:]}
    k1_ok = k1_proc.returncode == 0 and k1_summary.get("write_failures", 1) == 0 and k1_summary.get("fail_count", 1) == 0
    return k1_ok, {"dump": dump_summary, "writer": k1_summary}


def run_k3a_delta_writer():
    commands_dir = RUN_DIR / "k3a_delta_commands"
    dump_summary_path = RUN_DIR / "k3a_delta_command_dump_summary.json"
    dump_details_path = RUN_DIR / "k3a_delta_command_dump_details.json"
    k3a_summary_path = RUN_DIR / "k3a_delta_rtnl_writer_summary.json"
    k3a_details_path = RUN_DIR / "k3a_delta_rtnl_writer_details.json"

    dump_cmd = [
        "sudo", "-n", "python3", "/home/lxl/bird/delta-assets/delta_fib_host_writer.py",
        "--project-name", PROJECT,
        "--metric", str(METRIC),
        "--jobs", str(WRITE_JOBS),
        "--executor", "process",
        "--write-mode", "setns-ip",
        "--source", "event",
        "--dump-only",
        "--commands-dir", str(commands_dir),
        "--summary-path", str(dump_summary_path),
        "--details-path", str(dump_details_path),
    ]
    dump_cmd.extend(event_offset_args())
    dump_proc = run_cmd(dump_cmd, timeout=max(COMMAND_TIMEOUT, 1800), check=False)
    try:
        dump_summary = json.loads(dump_summary_path.read_text(encoding="utf-8"))
    except Exception:
        dump_summary = {"raw_stdout": dump_proc.stdout[-4000:], "raw_stderr": dump_proc.stderr[-4000:]}
    dump_ok = dump_proc.returncode == 0 and dump_summary.get("failures", 1) == 0
    if not dump_ok:
        return False, {"dump": dump_summary, "writer": None}

    k3a_cmd = [
        "sudo", "-n", "python3", "/home/lxl/bird/delta-assets/rtnetlink_route_writer.py",
        "--project-name", PROJECT,
        "--commands-dir", str(commands_dir),
        "--metric", str(METRIC),
        "--protocol", "12",
        "--jobs", str(K3A_DELTA_JOBS),
        "--batch", str(K3A_BATCH),
        "--ack-mode", K3A_ACK_MODE,
        "--multipath-policy", K3A_MULTIPATH_POLICY,
        "--ignore-delete-miss",
        "--summary-path", str(k3a_summary_path),
        "--details-path", str(k3a_details_path),
    ]
    k3a_proc = run_cmd(k3a_cmd, timeout=max(COMMAND_TIMEOUT, 1800), check=False)
    try:
        k3a_summary = json.loads(k3a_summary_path.read_text(encoding="utf-8"))
    except Exception:
        k3a_summary = {"raw_stdout": k3a_proc.stdout[-4000:], "raw_stderr": k3a_proc.stderr[-4000:]}
    k3a_ok = k3a_proc.returncode == 0 and k3a_summary.get("write_failures", 1) == 0 and k3a_summary.get("fail_count", 1) == 0
    return k3a_ok, {"dump": dump_summary, "writer": k3a_summary}


def write_delta(node):
    start = time.monotonic()
    shell = """
set -e
offset="$(cat /run/bird/delta-offset-before-reconv)"
/usr/local/bin/delta_fib_agent.py \
  --event-file /run/bird/delta-events.tsv \
  --since-byte-offset "${offset}" \
  --final-collapse \
  --stats-file /run/bird/delta-fib-reconv.json \
  --keep-going >/dev/null
"""
    proc = docker_exec(node["name"], shell, timeout=COMMAND_TIMEOUT, check=False)
    result = {
        "name": node["name"],
        "success": proc.returncode == 0,
        "duration_seconds": round(time.monotonic() - start, 6),
        "error": "" if proc.returncode == 0 else (proc.stderr.strip() or proc.stdout.strip())[-1000:],
    }
    try:
        stats = json.loads(docker_exec(node["name"], "cat /run/bird/delta-fib-reconv.json", timeout=10).stdout)
        result["stats"] = {
            "raw_route_events": int(stats.get("raw_route_events") or 0),
            "collapsed_events": int(stats.get("collapsed_events") or 0),
            "commands": int(stats.get("commands") or 0),
            "read_collapse_seconds": float(stats.get("read_collapse_seconds") or 0.0),
            "write_seconds": float(stats.get("write_seconds") or 0.0),
            "batch_errors": len(stats.get("batch_errors") or []),
            "start_byte_offset": int(stats.get("start_byte_offset") or 0),
            "end_byte_offset": int(stats.get("end_byte_offset") or 0),
        }
    except Exception as exc:
        result["stats_error"] = repr(exc)
    return result


def aggregate_write(results):
    stats_rows = [item.get("stats", {}) for item in results if item.get("success") and item.get("stats")]
    durations = [item["duration_seconds"] for item in results if item.get("success")]
    failed = [item for item in results if not item.get("success")]

    def total(key):
        return sum(float(row.get(key) or 0.0) for row in stats_rows)

    out = {
        "tasks": len(results),
        "successes": len(durations),
        "failures": len(failed),
        "failed_samples": failed[:20],
        "raw_route_events": int(total("raw_route_events")),
        "collapsed_events": int(total("collapsed_events")),
        "commands": int(total("commands")),
        "batch_errors_total": int(total("batch_errors")),
        "read_collapse_seconds_sum": total("read_collapse_seconds"),
        "write_seconds_sum": total("write_seconds"),
    }
    if durations:
        ordered = sorted(durations)
        out.update({
            "task_duration_min": min(durations),
            "task_duration_avg": statistics.fmean(durations),
            "task_duration_p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "task_duration_max": max(durations),
        })
    return out


def validate_one(node):
    shell = f"""
rib="$(birdc -s /run/bird/bird.ctl show route primary count 2>/dev/null | awk '/master4/ {{print $6; exit}}')"
fib="$(ip route show proto bird metric {METRIC} | awk '/^[[:space:]]/ {{next}} NF {{n++}} END {{print n+0}}')"
errors=0
if [ -f /run/bird/delta-fib-reconv.json ]; then
  errors="$(python3 - <<'PY2'
import json
try:
    data=json.load(open('/run/bird/delta-fib-reconv.json'))
    print(len(data.get('batch_errors', [])))
except Exception:
    print(999)
PY2
)"
fi
printf '%s\t%s\t%s\t%s\n' "$(hostname)" "$rib" "$fib" "$errors"
"""
    proc = docker_exec(node["name"], shell, timeout=30, check=False)
    fields = proc.stdout.strip().split("\t")
    item = {"container": node["name"], "raw": proc.stdout.strip(), "success": False}
    if proc.returncode == 0 and len(fields) == 4:
        name, rib, fib, errors = fields
        item.update({"name": name, "rib": int(rib or -1), "fib": int(fib or -1), "errors": int(errors or 999)})
        item["success"] = item["rib"] == item["fib"] and item["errors"] == 0
    return item


try:
    write_summary()
    nodes_before = running_brdnodes()

    if MARK_OFFSETS_BEFORE_FAILURE:
        stage_start("mark_offsets", brdnodes=len(nodes_before), mode="pre_failure_stat")
        marks = map_nodes(nodes_before, mark_offset, OFFSET_JOBS)
        marks_ok = [item for item in marks if item["success"]]
        stage_finish("mark_offsets", len(marks_ok) == len(marks), brdnodes=len(nodes_before), successes=len(marks_ok), failures=len(marks) - len(marks_ok), failed_samples=[item for item in marks if not item["success"]][:20])
        if len(marks_ok) != len(marks):
            raise LabError("failed to mark offsets in all brdnodes")
    else:
        stage_start("mark_offsets", brdnodes=len(nodes_before), mode="deferred_zero_offset")
        stage_finish("mark_offsets", True, skipped_pre_failure_mark=True, reason="delta protocol ignores initial feed; route events are post-enable changes")

    victims = choose_victims(nodes_before)
    stage_start("stop_victims", victims=victims)
    stop_results = [stop_victim(victim) for victim in victims]
    stage_finish("stop_victims", all(item["success"] for item in stop_results), results=stop_results)
    if not all(item["success"] for item in stop_results):
        raise LabError("failed to stop all selected brdnodes")

    nodes_after = running_brdnodes()
    stage_start("wait_rib_stable", running_brdnodes=len(nodes_after), sample_size=min(STABILITY_SAMPLES, len(nodes_after)))
    stable_info = wait_stable(nodes_after)
    stage_finish("wait_rib_stable", True, **stable_info)

    if not MARK_OFFSETS_BEFORE_FAILURE and USE_OFFSET_MARK:
        stage_start("prepare_delta_offsets", running_brdnodes=len(nodes_after), mode="offset_zero_after_stability")
        marks = map_nodes(nodes_after, set_zero_offset, OFFSET_JOBS)
        marks_ok = [item for item in marks if item["success"]]
        stage_finish("prepare_delta_offsets", len(marks_ok) == len(marks), brdnodes=len(nodes_after), successes=len(marks_ok), failures=len(marks) - len(marks_ok), failed_samples=[item for item in marks if not item["success"]][:20])
        if len(marks_ok) != len(marks):
            raise LabError("failed to prepare delta offsets in all running brdnodes")
    elif not MARK_OFFSETS_BEFORE_FAILURE:
        stage_start("prepare_delta_offsets", running_brdnodes=len(nodes_after), mode="implicit_zero_offset")
        stage_finish("prepare_delta_offsets", True, skipped_zero_offset_mark=True, reason="delta host writer defaults missing offset mark to byte offset 0")

    stage_start("write_delta_fib", running_brdnodes=len(nodes_after), jobs=WRITE_JOBS, write_backend=WRITE_BACKEND)
    if WRITE_BACKEND == "k1-bulk":
        write_ok, writer_summary = run_k1_delta_writer()
    elif WRITE_BACKEND == "k3a-rtnl":
        write_ok, writer_summary = run_k3a_delta_writer()
    else:
        write_ok, writer_summary = run_host_writer(
            "delta_host_writer",
            "event",
            WRITE_JOBS,
            event_offset_args(),
        )
    stage_finish("write_delta_fib", write_ok, writer=writer_summary)
    if not write_ok:
        raise LabError("delta FIB write failed")

    if RUN_FULL_COMPARISON:
        stage_start("write_full_fib_compare", running_brdnodes=len(nodes_after), jobs=FULL_WRITE_JOBS, write_mode="host-setns-birdc-primary")
        full_ok, full_writer_summary = run_host_writer(
            "full_birdc_primary_writer",
            "birdc-primary",
            FULL_WRITE_JOBS,
            ["--route-action", "replace"],
            timeout=FULL_COMMAND_TIMEOUT,
        )
        stage_finish("write_full_fib_compare", full_ok, writer=full_writer_summary)
        if not full_ok:
            raise LabError("full FIB comparison write failed")

    validate_nodes = pick_samples(nodes_after, VALIDATE_SAMPLES)
    stage_start("validate", samples=len(validate_nodes))
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(VALIDATE_SAMPLES, 16)) as executor:
        validations = list(executor.map(validate_one, validate_nodes))
    validation_ok = all(item["success"] for item in validations)
    stage_finish("validate", validation_ok, samples=validations)
    if not validation_ok:
        raise LabError("validation failed")

except Exception as exc:
    summary["error"] = str(exc)
finally:
    summary["ended_at"] = now_iso()
    summary["duration_seconds"] = round(time.monotonic() - started, 3)
    summary["success"] = summary.get("error") is None
    write_summary()
    print(json.dumps({"success": summary["success"], "summary": str(SUMMARY_PATH), "error": summary.get("error")}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if summary["success"] else 1)
PY
  local rc=$?
  set -e

  rsync_from_vm "${remote_dir}/summary.json" "${local_dir}/summary.json" || true
  echo "Local summary: ${local_dir}/summary.json"
  echo "Remote run dir: ${remote_dir}"
  return "${rc}"
}

phase="${1:-run}"
case "${phase}" in
  run) phase_run ;;
  status) phase_status ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
