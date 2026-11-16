#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VM_HOST="${VM_HOST:-192.168.122.60}"
VM_USER="${VM_USER:-lxl}"
PROJECT_NAME="${PROJECT_NAME:-topology-291e63c1}"
VM_RUN_ROOT="${VM_RUN_ROOT:-/home/lxl/bird/runs}"
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT:-${SCRIPT_DIR}/runs}"

DOWN_COUNT="${DOWN_COUNT:-10}"
STOP_TIMEOUT="${STOP_TIMEOUT:-1}"
SELECT_STRATEGY="${SELECT_STRATEGY:-spread}"
POST_DOWN_SLEEP_SECONDS="${POST_DOWN_SLEEP_SECONDS:-30}"
LIMIT_CONTAINERS="${LIMIT_CONTAINERS:-0}"

F_SHARDS="${F_SHARDS:-8}"
REAL_DUMP_WORKERS="${REAL_DUMP_WORKERS:-16}"
CLEANUP_WORKERS="${CLEANUP_WORKERS:-16}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-180}"
PRE_CLEANUP="${PRE_CLEANUP:-1}"
SUCCESS_LOAD_THRESHOLD="${SUCCESS_LOAD_THRESHOLD:-50}"
SUCCESS_STABLE_SECONDS="${SUCCESS_STABLE_SECONDS:-0}"
SUCCESS_LOAD_TIMEOUT="${SUCCESS_LOAD_TIMEOUT:-1800}"
SAMPLE_COUNT="${SAMPLE_COUNT:-10}"

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

Environment overrides:
  VM_HOST=${VM_HOST}
  VM_USER=${VM_USER}
  PROJECT_NAME=${PROJECT_NAME}
  DOWN_COUNT=${DOWN_COUNT}
  POST_DOWN_SLEEP_SECONDS=${POST_DOWN_SLEEP_SECONDS}
  PRE_CLEANUP=${PRE_CLEANUP}
  F_SHARDS=${F_SHARDS}

The run phase stops DOWN_COUNT running brdnode containers from different ASes,
waits POST_DOWN_SLEEP_SECONDS, then manually refreshes main FIB with real F.

PRE_CLEANUP=1 flushes stale proto bird routes before F writes. This is the
default for reconvergence experiments because route replace cannot withdraw
stale routes left by the previous FIB state.
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

now_epoch() {
  python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
}

phase_status() {
  ssh_vm 'date -Is; cat /proc/loadavg; printf "containers="; docker ps -aq | wc -l; printf "running="; docker ps -q | wc -l; printf "project_running="; docker ps --filter label=com.docker.compose.project='"${PROJECT_NAME}"' -q | wc -l; printf "brd_running="; docker ps --filter label=com.docker.compose.project='"${PROJECT_NAME}"' --format "{{.Names}}" | grep -Ec "^as[0-9]+brd-r" || true; printf "bird="; pgrep -fc "(^|/)bird( |$)" || true'
}

select_and_stop_remote() {
  local remote_dir="$1"
  local quoted_project quoted_dir
  quoted_project="$(printf '%q' "${PROJECT_NAME}")"
  quoted_dir="$(printf '%q' "${remote_dir}")"
  ssh_vm "PROJECT_NAME=${quoted_project} RUN_DIR=${quoted_dir} DOWN_COUNT=${DOWN_COUNT} STOP_TIMEOUT=${STOP_TIMEOUT} SELECT_STRATEGY=${SELECT_STRATEGY} python3 -" <<'PY'
import datetime as dt
import json
import os
import re
import socket
import subprocess
import time
from collections import defaultdict
from pathlib import Path

project = os.environ["PROJECT_NAME"]
run_dir = Path(os.environ["RUN_DIR"])
down_count = int(os.environ["DOWN_COUNT"])
stop_timeout = int(os.environ["STOP_TIMEOUT"])
strategy = os.environ["SELECT_STRATEGY"]
router_re = re.compile(r"^as(\d+)brd-r")

def now_iso():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")

def cmd(args, **kwargs):
    return subprocess.run(args, text=True, capture_output=True, **kwargs)

def loadavg():
    return Path("/proc/loadavg").read_text(encoding="utf-8").strip()

def running_brdnodes():
    ps = cmd([
        "docker", "ps",
        "--filter", f"label=com.docker.compose.project={project}",
        "--format", "{{.Names}}",
    ], check=True)
    names = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    groups = defaultdict(list)
    for name in names:
        match = router_re.match(name)
        if match:
            groups[int(match.group(1))].append(name)
    for names_for_as in groups.values():
        names_for_as.sort()
    return groups

def choose_asns(asns, count):
    if count <= 0:
        return []
    if count > len(asns):
        raise RuntimeError(f"requested {count} ASes but only {len(asns)} running AS groups exist")
    if strategy == "first":
        return asns[:count]
    if strategy != "spread":
        raise RuntimeError(f"unsupported SELECT_STRATEGY={strategy!r}")
    if count == 1:
        return [asns[0]]
    selected = []
    used = set()
    for idx in range(count):
        pos = round(idx * (len(asns) - 1) / (count - 1))
        while pos in used and pos + 1 < len(asns):
            pos += 1
        while pos in used and pos - 1 >= 0:
            pos -= 1
        used.add(pos)
        selected.append(asns[pos])
    return selected

run_dir.mkdir(parents=True, exist_ok=True)
event_path = run_dir / "event.json"
groups = running_brdnodes()
asns = sorted(groups)
chosen_asns = choose_asns(asns, down_count)
victims = [{"as": asn, "name": groups[asn][0]} for asn in chosen_asns]
summary = {
    "schema_version": 1,
    "host": socket.gethostname(),
    "project": project,
    "strategy": strategy,
    "requested_down_count": down_count,
    "unique_running_as_before": len(asns),
    "running_brdnodes_before": sum(len(v) for v in groups.values()),
    "load_before": loadavg(),
    "victims": victims,
    "stop_started_at": now_iso(),
    "stop_started_epoch": time.time(),
    "stop_results": [],
}
event_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

for victim in victims:
    start = time.monotonic()
    proc = cmd(["docker", "stop", "-t", str(stop_timeout), victim["name"]])
    summary["stop_results"].append({
        "as": victim["as"],
        "name": victim["name"],
        "success": proc.returncode == 0,
        "returncode": proc.returncode,
        "duration_seconds": round(time.monotonic() - start, 6),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    })
    event_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

after_groups = running_brdnodes()
summary["stop_completed_at"] = now_iso()
summary["stop_completed_epoch"] = time.time()
summary["running_brdnodes_after"] = sum(len(v) for v in after_groups.values())
summary["unique_running_as_after"] = len(after_groups)
summary["load_after"] = loadavg()
summary["success"] = all(item["success"] for item in summary["stop_results"])
event_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({
    "event_path": str(event_path),
    "success": summary["success"],
    "victims": victims,
    "running_brdnodes_before": summary["running_brdnodes_before"],
    "running_brdnodes_after": summary["running_brdnodes_after"],
}, ensure_ascii=False), flush=True)
raise SystemExit(0 if summary["success"] else 1)
PY
}

combine_summary() {
  local local_dir="$1"
  local remote_dir="$2"
  local rc="$3"
  local fib_started="$4"
  local fib_finished="$5"
  python3 - "${local_dir}" "${remote_dir}" "${rc}" "${fib_started}" "${fib_finished}" <<'PY'
import json
import sys
from pathlib import Path

local_dir = Path(sys.argv[1])
remote_dir = sys.argv[2]
rc = int(sys.argv[3])
fib_started = float(sys.argv[4])
fib_finished = float(sys.argv[5])
event_path = local_dir / "event.json"
f_path = local_dir / "f_writer" / "summary.json"
event = json.loads(event_path.read_text(encoding="utf-8")) if event_path.exists() else {}
f_summary = json.loads(f_path.read_text(encoding="utf-8")) if f_path.exists() else {}
f_result = f_summary.get("results", {}).get("F", {})
sharded = f_result.get("sharded_current_kernel", {})
fib_sample = f_result.get("fib_rib_sample", {})
stop_completed = event.get("stop_completed_epoch")
summary = {
    "schema_version": 1,
    "experiment": "reconfergence_F",
    "project": event.get("project"),
    "success": rc == 0 and bool(f_summary.get("success")),
    "remote_dir": remote_dir,
    "local_dir": str(local_dir),
    "rc": rc,
    "victims": event.get("victims", []),
    "running_brdnodes_before": event.get("running_brdnodes_before"),
    "running_brdnodes_after": event.get("running_brdnodes_after"),
    "post_down_sleep_seconds": max(0.0, fib_started - stop_completed) if stop_completed else None,
    "fib_command_wall_seconds": round(fib_finished - fib_started, 6),
    "event_to_fib_done_seconds": round(fib_finished - stop_completed, 6) if stop_completed else None,
    "f_route_ops": sharded.get("route_ops"),
    "f_duration_seconds": sharded.get("duration_seconds"),
    "f_route_ops_per_second": sharded.get("route_ops_per_second"),
    "f_task_summary": sharded.get("task_summary"),
    "fib_rib_sample": {
        "success": fib_sample.get("success"),
        "sample_count": fib_sample.get("sample_count"),
        "successes": fib_sample.get("successes"),
        "failures": fib_sample.get("failures"),
    },
    "event_summary_path": str(event_path),
    "f_summary_path": str(f_path),
}
out_path = local_dir / "summary.json"
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
PY
}

phase_run() {
  local run_stamp remote_dir local_dir pre_cleanup_arg rc fib_started fib_finished
  run_stamp="$(stamp)"
  remote_dir="${VM_RUN_ROOT}/reconfergence-F_${run_stamp}_${PROJECT_NAME}"
  local_dir="${LOCAL_RUN_ROOT}/reconfergence-F_${run_stamp}_${PROJECT_NAME}"
  mkdir -p "${local_dir}"
  ssh_vm "mkdir -p '${remote_dir}/f_writer'"
  rsync_to_vm "${SCRIPT_DIR}/rtnl_solution_lab.py" "/home/lxl/bird/rtnl_solution_lab.py"
  ssh_vm "chmod +x /home/lxl/bird/rtnl_solution_lab.py"

  echo "Stopping ${DOWN_COUNT} brdnode container(s) from different ASes..."
  select_and_stop_remote "${remote_dir}"

  if python3 - <<PY
import sys
sys.exit(0 if float("${POST_DOWN_SLEEP_SECONDS}") > 0 else 1)
PY
  then
    echo "Waiting ${POST_DOWN_SLEEP_SECONDS}s before manual F FIB refresh..."
    sleep "${POST_DOWN_SLEEP_SECONDS}"
  fi

  if [[ "${PRE_CLEANUP}" == "0" ]]; then
    pre_cleanup_arg="--no-pre-cleanup"
  else
    pre_cleanup_arg="--pre-cleanup"
  fi

  fib_started="$(now_epoch)"
  set +e
  ssh_vm "sudo -n python3 /home/lxl/bird/rtnl_solution_lab.py \
    --project '${PROJECT_NAME}' \
    --solution F \
    --real-bird \
    --no-real-prepare-bird \
    ${pre_cleanup_arg} \
    --limit-containers '${LIMIT_CONTAINERS}' \
    --work-dir '${remote_dir}/f_writer' \
    --timeout '${COMMAND_TIMEOUT}' \
    --no-pre-write-rib-stable \
    --success-load-threshold '${SUCCESS_LOAD_THRESHOLD}' \
    --success-stable-seconds '${SUCCESS_STABLE_SECONDS}' \
    --success-load-timeout '${SUCCESS_LOAD_TIMEOUT}' \
    --sample-count '${SAMPLE_COUNT}' \
    --fib-rib-min-delta 0 \
    --f-shards '${F_SHARDS}' \
    --real-dump-workers '${REAL_DUMP_WORKERS}' \
    --cleanup-workers '${CLEANUP_WORKERS}' \
    --no-cleanup-after"
  rc=$?
  set -e
  fib_finished="$(now_epoch)"

  rsync_from_vm "${remote_dir}/" "${local_dir}/" || true
  combine_summary "${local_dir}" "${remote_dir}" "${rc}" "${fib_started}" "${fib_finished}"
  echo "LOCAL_DIR=${local_dir}"
  echo "REMOTE_DIR=${remote_dir}"
  return "${rc}"
}

main() {
  local phase="${1:-run}"
  case "${phase}" in
    run) phase_run ;;
    status) phase_status ;;
    -h|--help|help) usage ;;
    *)
      usage >&2
      return 2
      ;;
  esac
}

main "$@"
