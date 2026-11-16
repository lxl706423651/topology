#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VM_HOST="${VM_HOST:-192.168.122.60}"
VM_USER="${VM_USER:-lxl}"
VM_DOMAIN="${VM_DOMAIN:-bird-rtnl-dev}"
PROJECT_NAME="${PROJECT_NAME:-topology-3083}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/F_reconf_multiscale_20260614_094101/scale_3083/output}"
RUN_NAME="${RUN_NAME:-loadonly_3083_delta_vs_full_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/runs/${RUN_NAME}}"

RESET_VM="${RESET_VM:-1}"
LOAD_THRESHOLD="${LOAD_THRESHOLD:-40}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-7200}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
RESTART_DELAY="${RESTART_DELAY:-0.1}"
RESTART_JOBS="${RESTART_JOBS:-1}"

INITIAL_WRITE_JOBS="${INITIAL_WRITE_JOBS:-8}"
DELTA_WRITE_JOBS="${DELTA_WRITE_JOBS:-8}"
FULL_WRITE_JOBS="${FULL_WRITE_JOBS:-8}"
DELTA_JOBS="${DELTA_JOBS:-16}"
AGENT_JOBS="${AGENT_JOBS:-16}"
DOWN_COUNT="${DOWN_COUNT:-10}"
METRIC="${METRIC:-232}"

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ConnectTimeout=10
)

log() {
  printf '%(%F %T)T %s\n' -1 "$*"
}

ssh_vm() {
  ssh "${SSH_OPTS[@]}" "${VM_USER}@${VM_HOST}" "$@"
}

wait_ssh() {
  local i
  for i in $(seq 1 90); do
    if ssh_vm "echo ready" >/dev/null 2>&1; then
      log "VM SSH is ready"
      return 0
    fi
    sleep 5
  done
  log "ERROR: VM SSH did not become ready"
  return 1
}

reset_vm() {
  log "Resetting VM ${VM_DOMAIN} and Docker runtime"
  virsh destroy "${VM_DOMAIN}" >/dev/null 2>&1 || true
  virsh start "${VM_DOMAIN}"
  wait_ssh
  ssh_vm "/home/${VM_USER}/bird/reset_docker_vm.sh"
  ssh_vm "printf 'containers='; docker ps -aq | wc -l; printf 'volumes='; docker volume ls -q | wc -l; printf 'custom_networks='; docker network ls --filter type=custom -q | wc -l"
}

summarize() {
  python3 - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
main_path = run_dir / "main_summary.json"
reconv_path = run_dir / "reconvergence" / "summary.json"
out_path = run_dir / "comparison_summary.json"

def load(path):
    if not path.exists():
        return {}
    return json.load(open(path, encoding="utf-8"))

main = load(main_path)
reconv = load(reconv_path)
main_stages = main.get("stages", {})
reconv_stages = reconv.get("stages", {})
delta_writer = reconv_stages.get("write_delta_fib", {}).get("writer", {})
full_writer = reconv_stages.get("write_full_fib_compare", {}).get("writer", {})
initial_writer = main_stages.get("write_initial_fib", {}).get("writer_stats", {})

summary = {
    "main_summary": str(main_path),
    "reconvergence_summary": str(reconv_path),
    "success": bool(main.get("success")) and bool(reconv.get("success")),
    "main": {
        "duration_seconds": main.get("duration_seconds"),
        "build_seconds": main_stages.get("build", {}).get("duration_seconds"),
        "up_seconds": main_stages.get("up", {}).get("duration_seconds"),
        "restart_bird_seconds": main_stages.get("restart_bird", {}).get("duration_seconds"),
        "wait_load40_seconds": main_stages.get("wait_rib_stable", {}).get("duration_seconds"),
        "initial_full_fib_wall_seconds": initial_writer.get("wall_seconds"),
        "initial_full_fib_commands": initial_writer.get("commands"),
    },
    "reconvergence": {
        "duration_seconds": reconv.get("duration_seconds"),
        "mark_offsets_seconds": reconv_stages.get("mark_offsets", {}).get("duration_seconds"),
        "stop_victims_seconds": reconv_stages.get("stop_victims", {}).get("duration_seconds"),
        "wait_load40_seconds": reconv_stages.get("wait_rib_stable", {}).get("duration_seconds"),
        "delta": {
            "wall_seconds": delta_writer.get("wall_seconds"),
            "raw_route_events": delta_writer.get("raw_route_events"),
            "collapsed_events": delta_writer.get("collapsed_events"),
            "commands": delta_writer.get("commands"),
            "read_collapse_seconds": delta_writer.get("read_collapse_seconds"),
            "write_seconds": delta_writer.get("write_seconds"),
            "jobs": delta_writer.get("jobs"),
        },
        "full_birdc_primary": {
            "wall_seconds": full_writer.get("wall_seconds"),
            "raw_route_events": full_writer.get("raw_route_events"),
            "collapsed_events": full_writer.get("collapsed_events"),
            "commands": full_writer.get("commands"),
            "read_collapse_seconds": full_writer.get("read_collapse_seconds"),
            "write_seconds": full_writer.get("write_seconds"),
            "jobs": full_writer.get("jobs"),
        },
    },
}

out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

mkdir -p "${RUN_DIR}"

log "Run dir: ${RUN_DIR}"
log "Output dir: ${OUTPUT_DIR}"

if [ "${RESET_VM}" = "1" ]; then
  reset_vm | tee "${RUN_DIR}/reset_vm.log"
else
  wait_ssh
fi

log "Running initial 3083 flow: build/up, serial restart-bird, load-only wait, full FIB write"
python3 "${SCRIPT_DIR}/run_bird_delta_experiment.py" "${OUTPUT_DIR}" \
  --vm "${VM_HOST}" \
  --vm-user "${VM_USER}" \
  --project-name "${PROJECT_NAME}" \
  --build-mode delta-fast \
  --defer-delta \
  --restart-jobs "${RESTART_JOBS}" \
  --restart-delay "${RESTART_DELAY}" \
  --load-only-stable \
  --load-threshold "${LOAD_THRESHOLD}" \
  --stable-seconds 0 \
  --max-wait-seconds "${MAX_WAIT_SECONDS}" \
  --poll-interval "${POLL_INTERVAL}" \
  --delta-jobs "${DELTA_JOBS}" \
  --agent-jobs "${AGENT_JOBS}" \
  --write-jobs "${INITIAL_WRITE_JOBS}" \
  --fib-writer host-setns \
  --initial-source birdc-primary \
  --initial-route-action replace \
  --summary-path "${RUN_DIR}/main_summary.json" \
  2>&1 | tee "${RUN_DIR}/main.log"

log "Running reconvergence comparison: delta event writer first, then full birdc-primary writer"
VM_HOST="${VM_HOST}" \
VM_USER="${VM_USER}" \
PROJECT_NAME="${PROJECT_NAME}" \
RUN_NAME="${RUN_NAME}_reconvergence" \
LOCAL_RUN_DIR="${RUN_DIR}/reconvergence" \
REMOTE_RUN_DIR="/home/${VM_USER}/bird/runs/${RUN_NAME}/reconvergence" \
DOWN_COUNT="${DOWN_COUNT}" \
METRIC="${METRIC}" \
LOAD_THRESHOLD="${LOAD_THRESHOLD}" \
LOAD_ONLY_STABLE=1 \
STABLE_SECONDS=0 \
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS}" \
POLL_INTERVAL="${POLL_INTERVAL}" \
WRITE_JOBS="${DELTA_WRITE_JOBS}" \
FULL_WRITE_JOBS="${FULL_WRITE_JOBS}" \
RUN_FULL_COMPARISON=1 \
COMMAND_TIMEOUT=240 \
"${SCRIPT_DIR}/reconvergence_bird_delta.sh" run \
  2>&1 | tee "${RUN_DIR}/reconvergence.log"

log "Writing comparison summary"
summarize | tee "${RUN_DIR}/comparison_summary.pretty.json"

log "Done: ${RUN_DIR}"
