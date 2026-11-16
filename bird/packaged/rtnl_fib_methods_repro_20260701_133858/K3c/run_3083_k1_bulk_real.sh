#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOPOLOGY_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VM_HOST="${VM_HOST:-192.168.122.60}"
VM_USER="${VM_USER:-lxl}"
VM_DOMAIN="${VM_DOMAIN:-bird-rtnl-dev}"
PROJECT_NAME="${PROJECT_NAME:-topology-3083}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/F_reconf_multiscale_20260614_094101/scale_3083/output}"
K1_KERNEL="${K1_KERNEL:-6.8.12-birdrtbulk}"
K1_GRUB_ENTRY="${K1_GRUB_ENTRY:-Advanced options for Ubuntu>Ubuntu, with Linux 6.8.12-birdrtbulk}"
K1_GRUB_ENTRY_ID="${K1_GRUB_ENTRY_ID:-}"

BUILD_JOBS="${BUILD_JOBS:-16}"
UP_JOBS="${UP_JOBS:-8}"
BUILD_BATCH="${BUILD_BATCH:-50}"
UP_BATCH="${UP_BATCH:-50}"

START_BIRD_JOBS="${START_BIRD_JOBS:-1}"
START_BIRD_DELAY="${START_BIRD_DELAY:-0.1}"
START_BIRD_TIMEOUT="${START_BIRD_TIMEOUT:-180}"

K1_JOBS="${K1_JOBS:-8}"
K1_DUMP_JOBS="${K1_DUMP_JOBS:-16}"
K1_BATCH="${K1_BATCH:-2000}"
K1_COMMAND_TIMEOUT="${K1_COMMAND_TIMEOUT:-240}"
K1_KEEP_DUMPS="${K1_KEEP_DUMPS:-0}"
K1_MULTIPATH_POLICY="${K1_MULTIPATH_POLICY:-first}"

LOAD_THRESHOLD="${LOAD_THRESHOLD:-50}"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-7200}"
LOAD_POLL_INTERVAL="${LOAD_POLL_INTERVAL:-30}"
IMPORT_SEEDEMU_IMAGES_FROM_HOST="${IMPORT_SEEDEMU_IMAGES_FROM_HOST:-auto}"

RUN_ID="${RUN_ID:-k1-real-3083_$(date +%Y%m%d_%H%M%S)_${PROJECT_NAME}}"
LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-${SCRIPT_DIR}/runs/${RUN_ID}}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-/home/${VM_USER}/bird/runs/${RUN_ID}}"

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ConnectTimeout=10
)

usage() {
  cat <<EOF
Usage:
  $0 prepare-k1-reset
  $0 run
  $0 start-bird
  $0 verify-start
  $0 writeK1
  $0 verify-fib
  $0 all
  $0 status

Default topology:
  PROJECT_NAME=${PROJECT_NAME}
  OUTPUT_DIR=${OUTPUT_DIR}

Notes:
  all = prepare-k1-reset, run, start-bird, verify-start, writeK1, verify-fib.
  writeK1 uses BIRD_RT_BULK on kernel ${K1_KERNEL}.
EOF
}

ssh_vm() {
  ssh "${SSH_OPTS[@]}" "${VM_USER}@${VM_HOST}" "$@"
}

rsync_to_vm() {
  rsync -a -e "ssh ${SSH_OPTS[*]}" "$1" "${VM_USER}@${VM_HOST}:$2"
}

rsync_from_vm() {
  rsync -a -e "ssh ${SSH_OPTS[*]}" "${VM_USER}@${VM_HOST}:$1" "$2"
}

wait_ssh() {
  for i in $(seq 1 90); do
    if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=5 "${VM_USER}@${VM_HOST}" 'true' >/dev/null 2>&1; then
      echo "ssh-ready-at=${i}"
      return 0
    fi
    sleep 5
  done
  echo "ERROR: SSH did not become ready on ${VM_HOST}" >&2
  return 1
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

sync_k1_scripts() {
  ssh_vm "mkdir -p /home/${VM_USER}/bird/kernel/tools ${REMOTE_RUN_DIR}"
  rsync_to_vm "${SCRIPT_DIR}/kernel/tools/kernel_bulk_writer.py" "/home/${VM_USER}/bird/kernel/tools/kernel_bulk_writer.py"
  rsync_to_vm "${SCRIPT_DIR}/kernel/tools/kernel_bulk_birdc_writer.py" "/home/${VM_USER}/bird/kernel/tools/kernel_bulk_birdc_writer.py"
  rsync_to_vm "${SCRIPT_DIR}/verify_bird_fib_vm.py" "/home/${VM_USER}/bird/verify_bird_fib_vm.py"
  ssh_vm "chmod +x /home/${VM_USER}/bird/kernel/tools/kernel_bulk_writer.py /home/${VM_USER}/bird/kernel/tools/kernel_bulk_birdc_writer.py /home/${VM_USER}/bird/verify_bird_fib_vm.py"
}

resolve_k1_grub_entry() {
  if [[ -n "${K1_GRUB_ENTRY_ID}" ]]; then
    printf '%s\n' "${K1_GRUB_ENTRY_ID}"
    return 0
  fi
  local resolved
  resolved="$(
    ssh "${SSH_OPTS[@]}" "${VM_USER}@${VM_HOST}" "sudo -n python3 - '${K1_KERNEL}'" <<'PY'
import re
import sys
kernel = sys.argv[1]
submenu_id = None
entry_id = None
with open("/boot/grub/grub.cfg", "r", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        if "submenu 'Advanced options for Ubuntu'" in line:
            match = re.search(r"\$menuentry_id_option '([^']+)'", line)
            if match:
                submenu_id = match.group(1)
        if f"menuentry 'Ubuntu, with Linux {kernel}'" in line and "recovery mode" not in line:
            match = re.search(r"\$menuentry_id_option '([^']+)'", line)
            if match:
                entry_id = match.group(1)
                break
if submenu_id and entry_id:
    print(f"{submenu_id}>{entry_id}")
elif entry_id:
    print(entry_id)
else:
    raise SystemExit(f"could not resolve grub entry id for {kernel}")
PY
  )"
  if [[ -n "${resolved}" ]]; then
    printf '%s\n' "${resolved}"
  else
    printf '%s\n' "${K1_GRUB_ENTRY}"
  fi
}

write_manifest() {
  mkdir -p "${LOCAL_RUN_DIR}"
  cat > "${LOCAL_RUN_DIR}/manifest.json" <<EOF
{
  "run_id": "${RUN_ID}",
  "project_name": "${PROJECT_NAME}",
  "output_dir": "${OUTPUT_DIR}",
  "local_run_dir": "${LOCAL_RUN_DIR}",
  "remote_run_dir": "${REMOTE_RUN_DIR}",
  "vm_host": "${VM_HOST}",
  "k1_kernel": "${K1_KERNEL}",
  "settings": {
    "build_jobs": ${BUILD_JOBS},
    "up_jobs": ${UP_JOBS},
    "start_bird_jobs": ${START_BIRD_JOBS},
    "start_bird_delay": ${START_BIRD_DELAY},
    "k1_jobs": ${K1_JOBS},
    "k1_dump_jobs": ${K1_DUMP_JOBS},
    "k1_batch": ${K1_BATCH},
    "k1_multipath_policy": "${K1_MULTIPATH_POLICY}",
    "load_threshold": ${LOAD_THRESHOLD}
  }
}
EOF
}

phase_status() {
  ssh_vm 'date -Is; uname -r; cat /proc/loadavg; printf "containers="; docker ps -aq | wc -l; printf "running="; docker ps -q | wc -l; printf "project_running="; docker ps --filter label=com.docker.compose.project='"${PROJECT_NAME}"' -q | wc -l; printf "images="; docker images -q | wc -l'
}

phase_prepare_k1_reset() {
  write_manifest
  local grub_entry
  grub_entry="$(resolve_k1_grub_entry)"
  echo "Setting default boot into ${K1_KERNEL}: ${grub_entry}"
  ssh_vm "sudo -n grub-set-default '${grub_entry}'"
  virsh destroy "${VM_DOMAIN}" || true
  virsh start "${VM_DOMAIN}"
  wait_ssh
  local kernel
  kernel="$(ssh_vm 'uname -r' | tr -d '\r\n')"
  echo "vm-kernel=${kernel}"
  if [[ "${kernel}" != "${K1_KERNEL}" ]]; then
    echo "ERROR: VM booted ${kernel}, expected ${K1_KERNEL}" >&2
    return 1
  fi
  sync_k1_scripts
  if [[ "${IMPORT_SEEDEMU_IMAGES_FROM_HOST}" == "1" ]] || { [[ "${IMPORT_SEEDEMU_IMAGES_FROM_HOST}" == "auto" ]] && host_has_seedemu_images; }; then
    ssh_vm '/home/lxl/bird/reset_docker_vm.sh --skip-rebuild-seedemu-images'
    import_seedemu_images_to_vm
  else
    ssh_vm '/home/lxl/bird/reset_docker_vm.sh'
  fi
  phase_status | tee "${LOCAL_RUN_DIR}/prepare_status.txt"
}

phase_run() {
  write_manifest
  sync_k1_scripts
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
  write_manifest
  PROJECT_NAME="${PROJECT_NAME}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  VM_HOST="${VM_HOST}" \
  VM_USER="${VM_USER}" \
  START_BIRD_JOBS="${START_BIRD_JOBS}" \
  START_BIRD_DELAY="${START_BIRD_DELAY}" \
  START_BIRD_TIMEOUT="${START_BIRD_TIMEOUT}" \
    "${SCRIPT_DIR}/run_f_real_main_fib_vm.sh" start-bird | tee "${LOCAL_RUN_DIR}/start_bird_driver.log"
}

phase_verify_start() {
  write_manifest
  sync_k1_scripts
  ssh_vm "python3 /home/${VM_USER}/bird/verify_bird_fib_vm.py \
    --project-name '${PROJECT_NAME}' \
    --stage start-bird \
    --load-threshold '${LOAD_THRESHOLD}' \
    --load-timeout '${LOAD_TIMEOUT}' \
    --poll-interval '${LOAD_POLL_INTERVAL}' \
    --summary-path '${REMOTE_RUN_DIR}/verify_after_start_bird.json'"
  rsync_from_vm "${REMOTE_RUN_DIR}/verify_after_start_bird.json" "${LOCAL_RUN_DIR}/verify_after_start_bird.json"
}

phase_write_k1() {
  write_manifest
  sync_k1_scripts
  local keep_arg=()
  if [[ "${K1_KEEP_DUMPS}" == "1" ]]; then
    keep_arg=(--keep-dumps)
  fi
  ssh_vm "sudo -n python3 /home/${VM_USER}/bird/kernel/tools/kernel_bulk_birdc_writer.py \
    --project-name '${PROJECT_NAME}' \
    --jobs '${K1_JOBS}' \
    --dump-jobs '${K1_DUMP_JOBS}' \
    --batch '${K1_BATCH}' \
    --metric 232 \
    --protocol 12 \
    --route-action replace \
    --multipath-policy '${K1_MULTIPATH_POLICY}' \
    --work-dir '${REMOTE_RUN_DIR}/k1_bulk_writer' \
    --command-timeout '${K1_COMMAND_TIMEOUT}' \
    --summary-path '${REMOTE_RUN_DIR}/k1_bulk_writer_summary.json' \
    --details-path '${REMOTE_RUN_DIR}/k1_bulk_writer_details.json' \
    ${keep_arg[*]}"
  rsync_from_vm "${REMOTE_RUN_DIR}/" "${LOCAL_RUN_DIR}/"
}

phase_verify_fib() {
  write_manifest
  sync_k1_scripts
  ssh_vm "python3 /home/${VM_USER}/bird/verify_bird_fib_vm.py \
    --project-name '${PROJECT_NAME}' \
    --stage fib-write \
    --load-threshold '${LOAD_THRESHOLD}' \
    --load-timeout '${LOAD_TIMEOUT}' \
    --poll-interval '${LOAD_POLL_INTERVAL}' \
    --summary-path '${REMOTE_RUN_DIR}/verify_after_fib_write.json'"
  rsync_from_vm "${REMOTE_RUN_DIR}/verify_after_fib_write.json" "${LOCAL_RUN_DIR}/verify_after_fib_write.json"
}

phase_all() {
  phase_prepare_k1_reset
  phase_run
  phase_start_bird
  phase_verify_start
  phase_write_k1
  phase_verify_fib
}

main() {
  local phase="${1:-}"
  case "${phase}" in
    prepare-k1-reset) phase_prepare_k1_reset ;;
    run) phase_run ;;
    start-bird) phase_start_bird ;;
    verify-start) phase_verify_start ;;
    writeK1) phase_write_k1 ;;
    verify-fib) phase_verify_fib ;;
    all) phase_all ;;
    status) phase_status ;;
    -h|--help|help|"") usage ;;
    *)
      usage >&2
      return 2
      ;;
  esac
}

main "$@"
