#!/usr/bin/env bash
set -Eeuo pipefail

PKG="${PKG:-/home/seed/rtnl_packages/rtnl_fib_methods_repro_20260701_133858_seedlocal}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/seed/lxl_topology/autocoder_test/outputttt}"
PROJECT_NAME="${PROJECT_NAME:-rtnl-k3c-fibref-7001}"
RUN_NAME="${RUN_NAME:-k3c-fibref-7001-continue_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${PKG}/K3c/runs/${RUN_NAME}}"

MEMLOG="${RUN_DIR}/memory_sample.csv"
LOG="${RUN_DIR}/tmux.log"
STATUS="${RUN_DIR}/exit.status"
ABORT_FILE="${RUN_DIR}/memory_abort.txt"
START_TIME="${RUN_DIR}/start.time"
END_TIME="${RUN_DIR}/end.time"

MIN_AVAILABLE_KB="${MIN_AVAILABLE_KB:-31457280}" # 30 GiB
MAX_SWAP_USED_KB="${MAX_SWAP_USED_KB:-2097152}"  # 2 GiB
ABORT_ON_LOW_MEMORY="${ABORT_ON_LOW_MEMORY:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-10}"

mkdir -p "${RUN_DIR}"

date -Is | tee "${START_TIME}"
{
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "kernel=$(uname -r)"
  echo "output_dir=${OUTPUT_DIR}"
  echo "project_name=${PROJECT_NAME}"
  echo "loadavg=$(cat /proc/loadavg)"
  echo "min_available_kb=${MIN_AVAILABLE_KB}"
  echo "max_swap_used_kb=${MAX_SWAP_USED_KB}"
  echo "skip_build_up=1"
  echo "project_containers=$(docker ps -a --filter label=com.docker.compose.project=${PROJECT_NAME} -q | wc -l)"
  echo "project_networks=$(docker network ls --filter label=com.docker.compose.project=${PROJECT_NAME} -q | wc -l)"
} | tee -a "${LOG}"

sample_mem() {
  echo 'ts,total_kb,available_kb,free_kb,buffers_kb,cached_kb,used_kb,swap_total_kb,swap_free_kb,swap_used_kb,loadavg,docker_containers,docker_running,docker_images,project_containers,project_networks' > "${MEMLOG}"
  while true; do
    local ts total avail free buffers cached swapt swapf swap_used used load dinfo pc pn
    ts="$(date -Is)"
    total="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
    avail="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    free="$(awk '/^MemFree:/ {print $2}' /proc/meminfo)"
    buffers="$(awk '/^Buffers:/ {print $2}' /proc/meminfo)"
    cached="$(awk '/^Cached:/ {print $2; exit}' /proc/meminfo)"
    swapt="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
    swapf="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
    used=$((total - avail))
    swap_used=$((swapt - swapf))
    load="$(tr ' ' '_' < /proc/loadavg)"
    dinfo="$(docker info --format '{{.Containers}},{{.ContainersRunning}},{{.Images}}' 2>/dev/null || echo 'NA,NA,NA')"
    pc="$(docker ps -a --filter label=com.docker.compose.project=${PROJECT_NAME} -q 2>/dev/null | wc -l)"
    pn="$(docker network ls --filter label=com.docker.compose.project=${PROJECT_NAME} -q 2>/dev/null | wc -l)"
    echo "${ts},${total},${avail},${free},${buffers},${cached},${used},${swapt},${swapf},${swap_used},${load},${dinfo},${pc},${pn}" >> "${MEMLOG}"

    if [[ "${ABORT_ON_LOW_MEMORY}" == "1" ]] && { (( avail < MIN_AVAILABLE_KB )) || (( swap_used > MAX_SWAP_USED_KB )); }; then
      {
        echo "abort_time=${ts}"
        echo "available_kb=${avail}"
        echo "swap_used_kb=${swap_used}"
        echo "loadavg=${load}"
        echo "reason=memory_guard"
      } | tee "${ABORT_FILE}" | tee -a "${LOG}"
      if [[ -n "${EXP_PGID:-}" ]]; then
        kill -TERM -- "-${EXP_PGID}" 2>/dev/null || true
        sleep 30
        kill -KILL -- "-${EXP_PGID}" 2>/dev/null || true
      fi
      break
    fi
    sleep "${SAMPLE_INTERVAL}"
  done
}

export VM_HOST=127.0.0.1
export VM_USER=seed
unset VM_DOMAIN || true
export RESET_VM=0
export K3C_KERNEL=6.8.12-birdrtbulk-k3c-fibref
export OUTPUT_DIR PROJECT_NAME RUN_NAME RUN_DIR
export SKIP_BUILD_UP=1
export RESTART_JOBS="${RESTART_JOBS:-1}"
export RESTART_DELAY="${RESTART_DELAY:-0.1}"
export COMPOSE_PARALLEL_LIMIT="${COMPOSE_PARALLEL_LIMIT:-8}"
export INITIAL_WRITE_JOBS="${INITIAL_WRITE_JOBS:-16}"
export K1_DUMP_JOBS="${K1_DUMP_JOBS:-16}"
export K1_BATCH="${K1_BATCH:-16000}"
export K1_OPS_PER_ATTR="${K1_OPS_PER_ATTR:-2000}"
export DELTA_WRITE_JOBS="${DELTA_WRITE_JOBS:-8}"
export OFFSET_JOBS="${OFFSET_JOBS:-16}"
export K1_DELTA_BATCH="${K1_DELTA_BATCH:-16000}"
export RUN_FULL_COMPARISON="${RUN_FULL_COMPARISON:-0}"
export MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-14400}"

setsid bash -c 'cd "$1"; shift; exec "$@"' bash "${PKG}/K3c" /usr/bin/time -v ./run_3083_k3c_delta_experiment.sh >> "${LOG}" 2>&1 &
EXP_PGID=$!

sample_mem &
sampler_pid=$!

cleanup() {
  kill "${sampler_pid}" 2>/dev/null || true
  wait "${sampler_pid}" 2>/dev/null || true
}
trap cleanup EXIT

set +e
wait "${EXP_PGID}"
rc=$?
set -e
echo "${rc}" > "${STATUS}"
date -Is | tee "${END_TIME}"
echo "exit_status=${rc}" | tee -a "${LOG}"
exit "${rc}"
