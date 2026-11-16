#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VM_HOST="${VM_HOST:-192.168.122.60}"
VM_USER="${VM_USER:-lxl}"
K3C_KERNEL="${K3C_KERNEL:-6.8.12-birdrtbulk-k3c}"
PROJECT_NAME="${PROJECT_NAME:-topology-3083}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/F_reconf_multiscale_20260614_094101/scale_3083/output}"

RUN_NAME="${RUN_NAME:-k3c-delta-reconv-3083_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/runs/${RUN_NAME}}"

# The K3c fast path works best when each generic-netlink attr contains routes
# that share the same fib_info fields. The writers default to grouping by those
# fields; keep mixed batches disabled so replace/delete do not share an attr.
export VM_HOST VM_USER PROJECT_NAME OUTPUT_DIR RUN_NAME RUN_DIR
export RESET_VM="${RESET_VM:-0}"
export K1_KERNEL="${K1_KERNEL:-${K3C_KERNEL}}"
export LOAD_THRESHOLD="${LOAD_THRESHOLD:-50}"
export MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-7200}"
export POLL_INTERVAL="${POLL_INTERVAL:-30}"
export RESTART_JOBS="${RESTART_JOBS:-1}"
export RESTART_DELAY="${RESTART_DELAY:-0.1}"
export BUILD_MODE="${BUILD_MODE:-delta-fast}"
export SKIP_BUILD_UP="${SKIP_BUILD_UP:-0}"
export FORCE_INSTALL="${FORCE_INSTALL:-0}"
export INITIAL_WRITE_JOBS="${INITIAL_WRITE_JOBS:-16}"
export K1_DUMP_JOBS="${K1_DUMP_JOBS:-16}"
export K1_BATCH="${K1_BATCH:-16000}"
export K1_COMMAND_TIMEOUT="${K1_COMMAND_TIMEOUT:-900}"
export K1_MULTIPATH_POLICY="${K1_MULTIPATH_POLICY:-first}"
export DELTA_JOBS="${DELTA_JOBS:-16}"
export DOWN_COUNT="${DOWN_COUNT:-10}"
export DELTA_WRITE_JOBS="${DELTA_WRITE_JOBS:-8}"
export OFFSET_JOBS="${OFFSET_JOBS:-16}"
export K1_DELTA_JOBS="${K1_DELTA_JOBS:-${DELTA_WRITE_JOBS}}"
export K1_DELTA_BATCH="${K1_DELTA_BATCH:-16000}"
export K1_OPS_PER_ATTR="${K1_OPS_PER_ATTR:-2000}"
export K1_DELTA_MIXED_BATCHES="${K1_DELTA_MIXED_BATCHES:-0}"
export RUN_FULL_COMPARISON="${RUN_FULL_COMPARISON:-0}"
export FULL_WRITE_JOBS="${FULL_WRITE_JOBS:-8}"
export FULL_COMMAND_TIMEOUT="${FULL_COMMAND_TIMEOUT:-1800}"
export SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-30}"

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ConnectTimeout="${SSH_CONNECT_TIMEOUT}"
)

current_kernel="$(ssh "${SSH_OPTS[@]}" "${VM_USER}@${VM_HOST}" 'uname -r' | tr -d '\r\n')"
if [[ "${RESET_VM}" == "0" && "${current_kernel}" != "${K3C_KERNEL}" ]]; then
  echo "ERROR: VM kernel is ${current_kernel}, expected ${K3C_KERNEL}." >&2
  echo "Boot the VM into K3c first, set RESET_VM=1 to let the runner reboot into K3c, or set K3C_KERNEL to the running kernel." >&2
  exit 1
fi

echo "K3c experiment run dir: ${RUN_DIR}"
echo "VM kernel: ${current_kernel}; target K3c kernel: ${K3C_KERNEL}; RESET_VM=${RESET_VM}"
exec "${SCRIPT_DIR}/run_3083_k1_delta_reconvergence.sh"
