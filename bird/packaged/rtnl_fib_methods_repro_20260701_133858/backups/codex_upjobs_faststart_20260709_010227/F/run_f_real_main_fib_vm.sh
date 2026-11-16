#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOPOLOGY_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VM_HOST="${VM_HOST:-192.168.122.60}"
VM_USER="${VM_USER:-lxl}"
VM_DOMAIN="${VM_DOMAIN:-bird-rtnl-dev}"
PROJECT_NAME="${PROJECT_NAME:-topology-291e63c1}"
OUTPUT_DIR="${OUTPUT_DIR:-${TOPOLOGY_DIR}/output}"
VM_RUN_ROOT="${VM_RUN_ROOT:-/home/lxl/bird/runs}"
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT:-${SCRIPT_DIR}/runs}"

BUILD_JOBS="${BUILD_JOBS:-16}"
UP_JOBS="${UP_JOBS:-8}"
BUILD_BATCH="${BUILD_BATCH:-50}"
UP_BATCH="${UP_BATCH:-50}"

START_BIRD_JOBS="${START_BIRD_JOBS:-16}"
START_BIRD_TIMEOUT="${START_BIRD_TIMEOUT:-180}"
START_BIRD_DELAY="${START_BIRD_DELAY:-0}"

F_SHARDS="${F_SHARDS:-8}"
REAL_DUMP_WORKERS="${REAL_DUMP_WORKERS:-16}"
CLEANUP_WORKERS="${CLEANUP_WORKERS:-16}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-180}"
SUCCESS_LOAD_THRESHOLD="${SUCCESS_LOAD_THRESHOLD:-50}"
SUCCESS_STABLE_SECONDS="${SUCCESS_STABLE_SECONDS:-120}"
SUCCESS_LOAD_TIMEOUT="${SUCCESS_LOAD_TIMEOUT:-7200}"
SAMPLE_COUNT="${SAMPLE_COUNT:-10}"
IMPORT_SEEDEMU_IMAGES_FROM_HOST="${IMPORT_SEEDEMU_IMAGES_FROM_HOST:-auto}"
LIMIT_CONTAINERS="${LIMIT_CONTAINERS:-0}"

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ConnectTimeout=10
)

usage() {
  cat <<EOF
Usage:
  $0 run
  $0 start-bird
  $0 writeFIB_F
  $0 all
  $0 reset
  $0 status

Environment overrides:
  VM_HOST=${VM_HOST}
  VM_USER=${VM_USER}
  PROJECT_NAME=${PROJECT_NAME}
  OUTPUT_DIR=${OUTPUT_DIR}
  F_SHARDS=${F_SHARDS}
  LIMIT_CONTAINERS=${LIMIT_CONTAINERS}
  IMPORT_SEEDEMU_IMAGES_FROM_HOST=${IMPORT_SEEDEMU_IMAGES_FROM_HOST}

Stages:
  run         Sync OUTPUT_DIR to the VM, build images, and docker compose up.
  start-bird Start BIRD inside brdnode containers using the configs already
              generated in the image. This stage does not rewrite kernel.conf.
  writeFIB_F Write main FIB with the real F sharded writer, validate samples,
              and keep the written FIB for manual inspection.
  all         Run run, start-bird, then writeFIB_F.
  reset       Destroy/start the VM and reset Docker inside the VM.
  status      Print VM Docker/load status.
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
  ssh_vm 'date -Is; cat /proc/loadavg; printf "containers="; docker ps -aq | wc -l; printf "running="; docker ps -q | wc -l; printf "project_running="; docker ps --filter label=com.docker.compose.project='"${PROJECT_NAME}"' -q | wc -l; printf "bird="; pgrep -fc "(^|/)bird( |$)" || true; printf "images="; docker images -q | wc -l; printf "networks="; docker network ls -q | wc -l'
}

host_has_seedemu_images() {
  docker image inspect \
    handsonsecurity/seedemu-base:2.0 \
    handsonsecurity/seedemu-router:2.0 >/dev/null 2>&1
}

import_seedemu_images_to_vm() {
  echo "Importing cached SeedEMU images from host into ${VM_HOST}..."
  docker save \
    handsonsecurity/seedemu-base:2.0 \
    handsonsecurity/seedemu-router:2.0 \
    | ssh "${SSH_OPTS[@]}" "${VM_USER}@${VM_HOST}" 'sudo -n docker load'
  ssh_vm 'sudo -n docker tag handsonsecurity/seedemu-base:2.0 handsonsecurity/seedemu-base:latest && sudo -n docker tag handsonsecurity/seedemu-router:2.0 handsonsecurity/seedemu-router:latest'
}

phase_reset() {
  virsh destroy "${VM_DOMAIN}" || true
  virsh start "${VM_DOMAIN}"
  for i in $(seq 1 60); do
    if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=5 "${VM_USER}@${VM_HOST}" 'true' >/dev/null 2>&1; then
      echo "ssh-ready-at=${i}"
      break
    fi
    sleep 5
    if [[ "${i}" == "60" ]]; then
      echo "ERROR: SSH did not become ready on ${VM_HOST}" >&2
      return 1
    fi
  done
  if [[ "${IMPORT_SEEDEMU_IMAGES_FROM_HOST}" == "1" ]] || { [[ "${IMPORT_SEEDEMU_IMAGES_FROM_HOST}" == "auto" ]] && host_has_seedemu_images; }; then
    ssh_vm '/home/lxl/bird/reset_docker_vm.sh --skip-rebuild-seedemu-images'
    import_seedemu_images_to_vm
  else
    ssh_vm '/home/lxl/bird/reset_docker_vm.sh'
  fi
  phase_status
}

phase_run() {
  python3 "${SCRIPT_DIR}/run.py" "${OUTPUT_DIR}" \
    --vm "${VM_HOST}" \
    --vm-user "${VM_USER}" \
    --project-name "${PROJECT_NAME}" \
    --build-jobs "${BUILD_JOBS}" \
    --build-batch "${BUILD_BATCH}" \
    --up-jobs "${UP_JOBS}" \
    --up-batch "${UP_BATCH}"
}

phase_start_bird() {
  local run_stamp remote_dir local_dir
  run_stamp="$(stamp)"
  remote_dir="${VM_RUN_ROOT}/start-bird_${run_stamp}_${PROJECT_NAME}"
  local_dir="${LOCAL_RUN_ROOT}/start-bird_${run_stamp}_${PROJECT_NAME}"
  mkdir -p "${local_dir}"
  ssh_vm "mkdir -p '${remote_dir}'"

  local quoted_project quoted_dir
  quoted_project="$(printf '%q' "${PROJECT_NAME}")"
  quoted_dir="$(printf '%q' "${remote_dir}")"
  ssh_vm "PROJECT_NAME=${quoted_project} START_BIRD_JOBS=${START_BIRD_JOBS} START_BIRD_TIMEOUT=${START_BIRD_TIMEOUT} START_BIRD_DELAY=${START_BIRD_DELAY} START_DIR=${quoted_dir} python3 -" <<'PY'
import concurrent.futures
import datetime as dt
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

project = os.environ["PROJECT_NAME"]
workers = int(os.environ["START_BIRD_JOBS"])
timeout = float(os.environ["START_BIRD_TIMEOUT"])
serial_delay = float(os.environ.get("START_BIRD_DELAY", "0") or "0")
start_dir = Path(os.environ["START_DIR"])
summary_path = start_dir / "summary.json"
router_re = re.compile(r"^as\d+brd-r")

def now():
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")

def run(cmd, **kwargs):
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)

def write_summary(data):
    start_dir.mkdir(parents=True, exist_ok=True)
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(summary_path)

summary = {
    "schema_version": 1,
    "started_at": now(),
    "ended_at": None,
    "success": None,
    "error": None,
    "host": socket.gethostname(),
    "project": project,
    "stage": "start-bird",
    "description": "Start BIRD with the existing container BIRD/kernel.conf configuration; do not rewrite kernel.conf.",
    "containers": [],
    "results": [],
}
write_summary(summary)

try:
    ps = run(["docker", "ps", "--filter", f"label=com.docker.compose.project={project}", "--format", "{{.ID}}"], check=False)
    ids = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    if not ids:
        raise RuntimeError(f"no running containers found for project {project}")
    inspect = run(["docker", "inspect", "-f", "{{.Name}} {{.State.Running}}"] + ids, check=False)
    containers = []
    for line in inspect.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].lstrip("/")
        if parts[1].lower() == "true" and router_re.match(name):
            containers.append({"name": name})
    containers.sort(key=lambda item: item["name"])
    if not containers:
        raise RuntimeError(f"no brdnode containers found for project {project}")
    summary["containers"] = containers
    write_summary(summary)

    def task(item):
        started = time.monotonic()
        name = item["name"]
        status = run(["docker", "exec", name, "sh", "-lc", "birdc show status >/dev/null 2>&1"], timeout=timeout)
        if status.returncode == 0:
            return {
                "name": name,
                "success": True,
                "already_running": True,
                "duration_seconds": round(time.monotonic() - started, 6),
                "error": "",
            }
        cmd = "rm -f /run/bird/bird.pid /run/bird/bird.ctl; bird -p -c /etc/bird/bird.conf && bird"
        proc = run(["docker", "exec", name, "sh", "-lc", cmd], timeout=timeout)
        if serial_delay > 0:
            time.sleep(serial_delay)
        return {
            "name": name,
            "success": proc.returncode == 0,
            "already_running": False,
            "duration_seconds": round(time.monotonic() - started, 6),
            "error": "" if proc.returncode == 0 else ((proc.stderr or proc.stdout or "").strip())[-2000:],
        }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(task, item): item for item in containers}
        for fut in concurrent.futures.as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:
                item = futures[fut]
                results.append({"name": item["name"], "success": False, "error": str(exc), "duration_seconds": 0})
    failures = [row for row in results if not row["success"]]
    summary["results"] = sorted(results, key=lambda item: item["name"])
    summary["task_summary"] = {
        "tasks": len(results),
        "successes": len(results) - len(failures),
        "failures": len(failures),
        "already_running": sum(1 for row in results if row.get("already_running")),
        "errors": failures[:20],
    }
    if failures:
        raise RuntimeError(f"start-bird failed for {len(failures)} containers")
    summary["success"] = True
except Exception as exc:
    summary["success"] = False
    summary["error"] = str(exc)
    print(f"ERROR: {exc}", flush=True)
finally:
    summary["ended_at"] = now()
    write_summary(summary)

print(f"Summary: {summary_path}", flush=True)
raise SystemExit(0 if summary["success"] else 1)
PY

  rsync_from_vm "${remote_dir}/" "${local_dir}/"
  echo "LOCAL_DIR=${local_dir}"
  echo "REMOTE_DIR=${remote_dir}"
}

phase_write_fib_f() {
  local run_stamp remote_dir local_dir rc
  run_stamp="$(stamp)"
  remote_dir="${VM_RUN_ROOT}/realfib-F_${run_stamp}_${PROJECT_NAME}"
  local_dir="${LOCAL_RUN_ROOT}/realfib-F_${run_stamp}_${PROJECT_NAME}"
  mkdir -p "${local_dir}"
  rsync_to_vm "${SCRIPT_DIR}/rtnl_solution_lab.py" "/home/lxl/bird/rtnl_solution_lab.py"
  ssh_vm "chmod +x /home/lxl/bird/rtnl_solution_lab.py"

  set +e
  local limit_arg=""
  if [[ "${LIMIT_CONTAINERS}" != "0" ]]; then
    limit_arg="--limit-containers '${LIMIT_CONTAINERS}'"
  fi
  ssh_vm "sudo -n python3 /home/lxl/bird/rtnl_solution_lab.py \
    --project '${PROJECT_NAME}' \
    --solution F \
    --real-bird \
    --no-real-prepare-bird \
    --no-pre-cleanup \
    ${limit_arg} \
    --work-dir '${remote_dir}' \
    --timeout '${COMMAND_TIMEOUT}' \
    --no-pre-write-rib-stable \
    --success-load-threshold '${SUCCESS_LOAD_THRESHOLD}' \
    --success-stable-seconds '${SUCCESS_STABLE_SECONDS}' \
    --success-load-timeout '${SUCCESS_LOAD_TIMEOUT}' \
    --sample-count '${SAMPLE_COUNT}' \
    --f-shards '${F_SHARDS}' \
    --real-dump-workers '${REAL_DUMP_WORKERS}' \
    --cleanup-workers '${CLEANUP_WORKERS}' \
    --no-cleanup-after"
  rc=$?
  set -e

  rsync_from_vm "${remote_dir}/" "${local_dir}/" || true
  echo "LOCAL_DIR=${local_dir}"
  echo "REMOTE_DIR=${remote_dir}"
  echo "RC=${rc}"
  return "${rc}"
}

phase_all() {
  phase_run
  phase_start_bird
  phase_write_fib_f
}

main() {
  local phase="${1:-}"
  case "${phase}" in
    run) phase_run ;;
    start-bird) phase_start_bird ;;
    writeFIB_F) phase_write_fib_f ;;
    all) phase_all ;;
    reset) phase_reset ;;
    status) phase_status ;;
    -h|--help|help|"") usage ;;
    *)
      usage >&2
      return 2
      ;;
  esac
}

main "$@"
