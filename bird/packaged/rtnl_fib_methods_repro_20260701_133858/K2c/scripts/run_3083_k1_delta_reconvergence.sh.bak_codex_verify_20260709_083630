#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VM_HOST="${VM_HOST:-192.168.122.60}"
VM_USER="${VM_USER:-lxl}"
PROJECT_NAME="${PROJECT_NAME:-topology-3083}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/F_reconf_multiscale_20260614_094101/scale_3083/output}"
RUN_NAME="${RUN_NAME:-k1-delta-reconv-3083_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/runs/${RUN_NAME}}"

RESET_VM="${RESET_VM:-1}"
LOAD_THRESHOLD="${LOAD_THRESHOLD:-50}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-7200}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
RESTART_JOBS="${RESTART_JOBS:-1}"
RESTART_DELAY="${RESTART_DELAY:-0.1}"
BUILD_MODE="${BUILD_MODE:-delta-fast}"
SKIP_BUILD_UP="${SKIP_BUILD_UP:-0}"
FORCE_INSTALL="${FORCE_INSTALL:-0}"
RESUME_AFTER_INITIAL="${RESUME_AFTER_INITIAL:-0}"

INITIAL_WRITE_JOBS="${INITIAL_WRITE_JOBS:-8}"
K1_DUMP_JOBS="${K1_DUMP_JOBS:-16}"
K1_BATCH="${K1_BATCH:-2000}"
K1_COMMAND_TIMEOUT="${K1_COMMAND_TIMEOUT:-240}"
K1_MULTIPATH_POLICY="${K1_MULTIPATH_POLICY:-first}"

DELTA_JOBS="${DELTA_JOBS:-16}"
DOWN_COUNT="${DOWN_COUNT:-10}"
DELTA_WRITE_JOBS="${DELTA_WRITE_JOBS:-8}"
OFFSET_JOBS="${OFFSET_JOBS:-${DELTA_WRITE_JOBS}}"
K1_DELTA_BATCH="${K1_DELTA_BATCH:-2000}"
K1_DELTA_MIXED_BATCHES="${K1_DELTA_MIXED_BATCHES:-0}"
RUN_FULL_COMPARISON="${RUN_FULL_COMPARISON:-0}"
FULL_WRITE_JOBS="${FULL_WRITE_JOBS:-8}"
FULL_COMMAND_TIMEOUT="${FULL_COMMAND_TIMEOUT:-1800}"
SKIP_AGENT_INSTALL="${SKIP_AGENT_INSTALL:-1}"
USE_OFFSET_MARK="${USE_OFFSET_MARK:-0}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-30}"

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}"
)

log() {
  printf '%(%F %T)T %s\n' -1 "$*"
}

ssh_vm() {
  ssh "${SSH_OPTS[@]}" "${VM_USER}@${VM_HOST}" "$@"
}

rsync_from_vm() {
  rsync -a -e "ssh ${SSH_OPTS[*]}" "${VM_USER}@${VM_HOST}:$1" "$2"
}

sample_vm_load1() {
  local output
  if ! output="$(timeout "$((SSH_CONNECT_TIMEOUT + 20))" ssh "${SSH_OPTS[@]}" "${VM_USER}@${VM_HOST}" "awk '{print \$1}' /proc/loadavg" 2>&1)"; then
    printf '%s\n' "${output}" >&2
    return 1
  fi
  printf '%s\n' "${output}" | awk 'NF {print $1; exit}'
}

reset_k1_vm() {
  log "Booting/resetting VM into K1 kernel and clearing Docker"
  VM_HOST="${VM_HOST}" VM_USER="${VM_USER}" \
  PROJECT_NAME="${PROJECT_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" \
  RUN_ID="${RUN_NAME}_prepare" LOCAL_RUN_DIR="${RUN_DIR}/prepare" \
    "${SCRIPT_DIR}/run_3083_k1_bulk_real.sh" prepare-k1-reset
}

run_initial_full_k1() {
  log "Initial 3083 flow: build/up, deferred delta start, K1 full FIB write"
  local build_args=()
  if [[ "${SKIP_BUILD_UP}" == "1" ]]; then
    build_args+=(--skip-build-up)
  fi
  if [[ "${FORCE_INSTALL}" == "1" ]]; then
    build_args+=(--force-install)
  fi
  python3 "${SCRIPT_DIR}/run_bird_delta_experiment.py" "${OUTPUT_DIR}" \
    --vm "${VM_HOST}" \
    --vm-user "${VM_USER}" \
    --project-name "${PROJECT_NAME}" \
    --build-mode "${BUILD_MODE}" \
    "${build_args[@]}" \
    --defer-delta \
    --skip-enable-delta-after-initial \
    --skip-start-agents \
    --skip-final-validate \
    --restart-jobs "${RESTART_JOBS}" \
    --restart-delay "${RESTART_DELAY}" \
    --load-only-stable \
    --load-threshold "${LOAD_THRESHOLD}" \
    --stable-seconds 0 \
    --max-wait-seconds "${MAX_WAIT_SECONDS}" \
    --poll-interval "${POLL_INTERVAL}" \
    --delta-jobs "${DELTA_JOBS}" \
    --write-jobs "${INITIAL_WRITE_JOBS}" \
    --fib-writer k1-bulk \
    --k1-dump-jobs "${K1_DUMP_JOBS}" \
    --k1-batch "${K1_BATCH}" \
    --k1-command-timeout "${K1_COMMAND_TIMEOUT}" \
    --k1-multipath-policy "${K1_MULTIPATH_POLICY}" \
    --summary-path "${RUN_DIR}/main_summary.json" \
    2>&1 | tee "${RUN_DIR}/main.log"
}

verify_initial_fib() {
  log "Verifying initial K1 full FIB write"
  local remote_verify="/home/${VM_USER}/bird/runs/${RUN_NAME}/verify_initial_fib.json"
  ssh_vm "mkdir -p /home/${VM_USER}/bird/runs/${RUN_NAME} && python3 /home/${VM_USER}/bird/verify_bird_fib_vm.py \
    --project-name '${PROJECT_NAME}' \
    --stage fib-write \
    --load-threshold '${LOAD_THRESHOLD}' \
    --load-timeout '${MAX_WAIT_SECONDS}' \
    --poll-interval '${POLL_INTERVAL}' \
    --summary-path '${remote_verify}'"
  rsync_from_vm "${remote_verify}" "${RUN_DIR}/verify_initial_fib.json"
}

wait_vm_load_below() {
  local stage_name="$1"
  local out_path="$2"
  local start_ts
  local start_epoch
  local samples=()
  local elapsed=0
  start_ts="$(date -Is)"
  start_epoch="$(date +%s)"
  log "${stage_name}: waiting for VM load1 < ${LOAD_THRESHOLD}"
  while true; do
    local sample_ts
    local load1
    sample_ts="$(date -Is)"
    if load1="$(sample_vm_load1)"; then
      samples+=("{\"time\":\"${sample_ts}\",\"load1\":${load1}}")
      log "${stage_name}: load1=${load1}"
      if python3 - "${load1}" "${LOAD_THRESHOLD}" <<'PY'
import sys
sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)
PY
      then
        break
      fi
    else
      samples+=("{\"time\":\"${sample_ts}\",\"error\":\"ssh_failed\"}")
      log "${stage_name}: load sample failed; retrying after ${POLL_INTERVAL}s"
    fi
    elapsed=$(( $(date +%s) - start_epoch ))
    if (( elapsed >= MAX_WAIT_SECONDS )); then
      {
        printf '{\n'
        printf '  "success": false,\n'
        printf '  "stage": "%s",\n' "${stage_name}"
        printf '  "start_time": "%s",\n' "${start_ts}"
        printf '  "duration_seconds": %s,\n' "${elapsed}"
        printf '  "load_threshold": %s,\n' "${LOAD_THRESHOLD}"
        printf '  "samples": [%s]\n' "$(IFS=,; echo "${samples[*]}")"
        printf '}\n'
      } > "${out_path}"
      return 1
    fi
    sleep "${POLL_INTERVAL}"
  done
  elapsed=$(( $(date +%s) - start_epoch ))
  {
    printf '{\n'
    printf '  "success": true,\n'
    printf '  "stage": "%s",\n' "${stage_name}"
    printf '  "start_time": "%s",\n' "${start_ts}"
    printf '  "duration_seconds": %s,\n' "${elapsed}"
    printf '  "load_threshold": %s,\n' "${LOAD_THRESHOLD}"
    printf '  "samples": [%s]\n' "$(IFS=,; echo "${samples[*]}")"
    printf '}\n'
  } > "${out_path}"
}

enable_delta_after_initial_verify() {
  log "Enabling BIRD delta after initial FIB verification"
  local started
  local ended
  local duration
  started="$(date +%s)"
  env VM_HOST="${VM_HOST}" \
    VM_USER="${VM_USER}" \
    PROJECT_NAME="${PROJECT_NAME}" \
    LIMIT_CONTAINERS=0 \
    DELTA_FIB_METRIC=232 \
    DELTA_JOBS="${DELTA_JOBS}" \
    DELTA_IGNORE_INITIAL_FEED=1 \
      "${SCRIPT_DIR}/install_bird_delta_vm.sh" enable-delta \
      2>&1 | tee "${RUN_DIR}/enable_delta_after_verify.log"
  ended="$(date +%s)"
  duration=$((ended - started))
  python3 - "${RUN_DIR}/enable_delta_after_verify.json" "${duration}" "${DELTA_JOBS}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.write_text(json.dumps({
    "success": True,
    "duration_seconds": int(sys.argv[2]),
    "jobs": int(sys.argv[3]),
    "ignore_initial_feed": True,
}, indent=2) + "\n", encoding="utf-8")
PY
  wait_vm_load_below "wait_after_enable_delta" "${RUN_DIR}/wait_after_enable_delta.json"
}

run_reconvergence_k1_delta() {
  log "Reconvergence: stop brdnodes and write delta routes through K1 bulk"
  VM_HOST="${VM_HOST}" \
  VM_USER="${VM_USER}" \
  PROJECT_NAME="${PROJECT_NAME}" \
  RUN_NAME="${RUN_NAME}_reconvergence" \
  LOCAL_RUN_DIR="${RUN_DIR}/reconvergence" \
  REMOTE_RUN_DIR="/home/${VM_USER}/bird/runs/${RUN_NAME}/reconvergence" \
  DOWN_COUNT="${DOWN_COUNT}" \
  LOAD_THRESHOLD="${LOAD_THRESHOLD}" \
  LOAD_ONLY_STABLE=1 \
  STABLE_SECONDS=0 \
  MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS}" \
  POLL_INTERVAL="${POLL_INTERVAL}" \
  WRITE_BACKEND=k1-bulk \
  SKIP_AGENT_INSTALL="${SKIP_AGENT_INSTALL}" \
  USE_OFFSET_MARK="${USE_OFFSET_MARK}" \
  WRITE_JOBS="${DELTA_WRITE_JOBS}" \
  OFFSET_JOBS="${OFFSET_JOBS}" \
  K1_DELTA_JOBS="${DELTA_WRITE_JOBS}" \
  K1_DELTA_BATCH="${K1_DELTA_BATCH}" \
  K1_DELTA_MIXED_BATCHES="${K1_DELTA_MIXED_BATCHES}" \
  K1_MULTIPATH_POLICY="${K1_MULTIPATH_POLICY}" \
  RUN_FULL_COMPARISON="${RUN_FULL_COMPARISON}" \
  FULL_WRITE_JOBS="${FULL_WRITE_JOBS}" \
  FULL_COMMAND_TIMEOUT="${FULL_COMMAND_TIMEOUT}" \
    "${SCRIPT_DIR}/reconvergence_bird_delta.sh" run \
    2>&1 | tee "${RUN_DIR}/reconvergence.log"
}

summarize() {
  log "Writing run summary"
  python3 - "${RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
main_path = run_dir / "main_summary.json"
wait_initial_path = run_dir / "wait_after_initial_k1_fib.json"
verify_path = run_dir / "verify_initial_fib.json"
enable_path = run_dir / "enable_delta_after_verify.json"
wait_after_enable_path = run_dir / "wait_after_enable_delta.json"
reconv_path = run_dir / "reconvergence" / "summary.json"
out_path = run_dir / "comparison_summary.json"

def load(path):
    return json.load(open(path, encoding="utf-8")) if path.exists() else {}

main = load(main_path)
wait_initial = load(wait_initial_path)
verify = load(verify_path)
enable = load(enable_path)
wait_after_enable = load(wait_after_enable_path)
reconv = load(reconv_path)
main_stages = main.get("stages", {})
reconv_stages = reconv.get("stages", {})
initial_writer = main_stages.get("write_initial_fib", {}).get("writer_stats", {})
delta_stage_writer = reconv_stages.get("write_delta_fib", {}).get("writer", {})
delta_dump = delta_stage_writer.get("dump", delta_stage_writer)
delta_writer = delta_stage_writer.get("writer", delta_stage_writer)

summary = {
    "main_summary": str(main_path),
    "wait_after_initial_k1_fib": str(wait_initial_path),
    "verify_initial_fib": str(verify_path),
    "reconvergence_summary": str(reconv_path),
    "success": bool(main.get("success")) and bool(verify.get("success", True)) and bool(reconv.get("success")),
    "main": {
        "duration_seconds": main.get("duration_seconds"),
        "build_seconds": main_stages.get("build", {}).get("duration_seconds"),
        "up_seconds": main_stages.get("up", {}).get("duration_seconds"),
        "restart_bird_seconds": main_stages.get("restart_bird", {}).get("duration_seconds"),
        "wait_load_seconds": main_stages.get("wait_rib_stable", {}).get("duration_seconds"),
        "enable_delta_seconds": main_stages.get("enable_delta", {}).get("duration_seconds"),
        "initial_k1_full_fib_wall_seconds": initial_writer.get("wall_seconds"),
        "initial_k1_full_fib_dump_seconds": initial_writer.get("dump_wall_seconds"),
        "initial_k1_full_fib_write_seconds": initial_writer.get("write_wall_seconds"),
        "initial_k1_full_fib_ops": initial_writer.get("ops_requested") or initial_writer.get("commands"),
        "initial_k1_full_fib_success_count": initial_writer.get("success_count"),
        "initial_k1_full_fib_failed_ops": initial_writer.get("fail_count") or initial_writer.get("failed_ops"),
    },
    "verify": {
        "success": verify.get("success"),
        "load1": verify.get("load", {}).get("load1"),
        "samples": verify.get("samples"),
    },
    "wait_after_initial_k1_fib": {
        "success": wait_initial.get("success"),
        "duration_seconds": wait_initial.get("duration_seconds"),
        "final_load1": (
            wait_initial.get("samples", [{}])[-1].get("load1")
            if wait_initial.get("samples") else None
        ),
    },
    "enable_delta_after_verify": {
        "success": enable.get("success"),
        "duration_seconds": enable.get("duration_seconds"),
        "jobs": enable.get("jobs"),
        "ignore_initial_feed": enable.get("ignore_initial_feed"),
    },
    "wait_after_enable_delta": {
        "success": wait_after_enable.get("success"),
        "duration_seconds": wait_after_enable.get("duration_seconds"),
        "final_load1": (
            wait_after_enable.get("samples", [{}])[-1].get("load1")
            if wait_after_enable.get("samples") else None
        ),
    },
    "reconvergence": {
        "duration_seconds": reconv.get("duration_seconds"),
        "mark_offsets_seconds": reconv_stages.get("mark_offsets", {}).get("duration_seconds"),
        "mark_offsets_mode": reconv_stages.get("mark_offsets", {}).get("mode"),
        "prepare_delta_offsets_seconds": reconv_stages.get("prepare_delta_offsets", {}).get("duration_seconds"),
        "stop_victims_seconds": reconv_stages.get("stop_victims", {}).get("duration_seconds"),
        "wait_load_seconds": reconv_stages.get("wait_rib_stable", {}).get("duration_seconds"),
        "delta_dump_wall_seconds": delta_dump.get("wall_seconds"),
        "delta_raw_route_events": delta_dump.get("raw_route_events"),
        "delta_collapsed_events": delta_dump.get("collapsed_events"),
        "delta_commands": delta_dump.get("commands") or delta_writer.get("commands"),
        "delta_writer_wall_seconds": delta_writer.get("wall_seconds"),
        "delta_writer_success_count": delta_writer.get("success_count"),
        "delta_writer_fail_count": delta_writer.get("fail_count"),
        "delta_writer_ignored_delete_miss": delta_writer.get("ignored_delete_miss"),
        "delta_writer_kernel_apply_seconds_sum": delta_writer.get("kernel_apply_seconds_sum"),
        "delta_writer_jobs": delta_writer.get("jobs"),
    },
}

out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

main() {
  mkdir -p "${RUN_DIR}"
  log "Run dir: ${RUN_DIR}"
  log "Output dir: ${OUTPUT_DIR}"
  if [[ "${RESUME_AFTER_INITIAL}" == "1" ]]; then
    log "RESUME_AFTER_INITIAL=1: using existing initial full-FIB result in ${RUN_DIR}"
  else
    if [[ "${RESET_VM}" == "1" ]]; then
      reset_k1_vm | tee "${RUN_DIR}/reset_k1_vm.log"
    fi
    run_initial_full_k1
  fi
  wait_vm_load_below "wait_after_initial_k1_fib" "${RUN_DIR}/wait_after_initial_k1_fib.json"
  verify_initial_fib
  enable_delta_after_initial_verify
  run_reconvergence_k1_delta
  summarize | tee "${RUN_DIR}/comparison_summary.pretty.json"
  log "Done: ${RUN_DIR}"
}

main "$@"
