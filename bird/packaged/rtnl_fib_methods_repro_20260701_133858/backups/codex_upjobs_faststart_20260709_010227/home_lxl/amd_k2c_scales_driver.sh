#!/usr/bin/env bash
set -Eeuo pipefail

BIRD="${BIRD:-/home/lxl/seed-emulator/topology/bird}"
TOPO="${TOPO:-/home/lxl/seed-emulator/topology}"
PKG="${PKG:-${BIRD}/packaged/rtnl_fib_methods_repro_20260701_133858}"
GENERATE_PYTHON="${GENERATE_PYTHON:-/home/lxl/anaconda3/bin/python}"
RUN_BATCH="${RUN_BATCH:-${PKG}/K2c/runs/k2c-scales_$(date +%Y%m%d_%H%M%S)}"
SCALES_STR="${SCALES:-4954 7001 9955}"

mkdir -p "${RUN_BATCH}"
LOG="${RUN_BATCH}/driver.log"
exec > >(tee -a "${LOG}") 2>&1

echo "started_at=$(date -Is)"
echo "host=$(hostname)"
echo "kernel=$(uname -r)"
echo "run_batch=${RUN_BATCH}"
echo "scales=${SCALES_STR}"
echo "docker=$(docker info --format 'Containers={{.Containers}} Running={{.ContainersRunning}} Images={{.Images}} NFd={{.NFd}} NGoroutines={{.NGoroutines}}')"

require_kernel() {
  local expected="6.8.12-birdrtbulk-k2c"
  local current
  current="$(uname -r)"
  if [[ "${current}" != "${expected}" ]]; then
    echo "ERROR: kernel=${current}, expected=${expected}" >&2
    exit 1
  fi
}

cleanup_project() {
  local project="$1"
  local c n attempt
  echo "cleanup_start project=${project} at=$(date -Is)"
  echo "cleanup_before_containers=$(docker ps -a --filter label=com.docker.compose.project="${project}" -q | wc -l)"
  echo "cleanup_before_networks=$(docker network ls --filter label=com.docker.compose.project="${project}" -q | wc -l)"

  for attempt in 1 2 3; do
    docker ps -a --filter label=com.docker.compose.project="${project}" -q \
      | xargs -r -n 50 docker rm -f >/dev/null || true
    c=$(docker ps -a --filter label=com.docker.compose.project="${project}" -q | wc -l)
    echo "cleanup_container_attempt_${attempt} remaining=${c}"
    [[ "${c}" == "0" ]] && break
    sleep 10
  done

  for attempt in 1 2 3; do
    docker network ls --filter label=com.docker.compose.project="${project}" -q \
      | xargs -r -n 50 docker network rm >/dev/null || true
    n=$(docker network ls --filter label=com.docker.compose.project="${project}" -q | wc -l)
    echo "cleanup_network_attempt_${attempt} remaining=${n}"
    [[ "${n}" == "0" ]] && break
    docker network prune -f --filter label=com.docker.compose.project="${project}" >/dev/null || true
    n=$(docker network ls --filter label=com.docker.compose.project="${project}" -q | wc -l)
    echo "cleanup_network_prune_${attempt} remaining=${n}"
    [[ "${n}" == "0" ]] && break
    sleep 10
  done

  c=$(docker ps -a --filter label=com.docker.compose.project="${project}" -q | wc -l)
  n=$(docker network ls --filter label=com.docker.compose.project="${project}" -q | wc -l)
  echo "cleanup_after_containers=${c}"
  echo "cleanup_after_networks=${n}"
  test "${c}" = "0"
  test "${n}" = "0"
  echo "cleanup_end project=${project} at=$(date -Is)"
}

service_count() {
  local compose="$1"
  python3 - "${compose}" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as f:
    data = yaml.safe_load(f)
print(len(data.get("services", {})), len(data.get("networks", {})))
PY
}

fix_compose_gateways() {
  local compose="$1"
  python3 - "${compose}" <<'PY'
import ipaddress
import shutil
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text()) or {}
services = data.get("services") or {}
networks = data.get("networks") or {}
used = {name: set() for name in networks}

for svc in services.values():
    svc_networks = svc.get("networks") or {}
    if not isinstance(svc_networks, dict):
        continue
    for net_name, cfg in svc_networks.items():
        if isinstance(cfg, dict):
            ip = cfg.get("ipv4_address")
            if ip:
                used.setdefault(net_name, set()).add(ip)

changed = 0
for net_name, net_cfg in networks.items():
    if not isinstance(net_cfg, dict):
        continue
    ipam = net_cfg.get("ipam") or {}
    configs = ipam.get("config") or []
    for cfg in configs:
        if not isinstance(cfg, dict) or cfg.get("gateway") or "subnet" not in cfg:
            continue
        try:
            subnet = ipaddress.ip_network(cfg["subnet"], strict=False)
        except ValueError:
            continue
        if subnet.version != 4 or subnet.num_addresses < 4:
            continue
        used_ips = used.get(net_name, set())
        for value in range(int(subnet.broadcast_address) - 1, int(subnet.network_address), -1):
            candidate = str(ipaddress.ip_address(value))
            if candidate not in used_ips:
                cfg["gateway"] = candidate
                changed += 1
                break

if changed:
    backup = path.with_suffix(path.suffix + ".gatewayfix.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
print(f"gateway_fix_changed={changed}")
PY
}

ensure_output() {
  local scale="$1"
  local scale_dir="${BIRD}/runs/F_reconf_multiscale_20260614_094101/scale_${scale}"
  local output_dir="${scale_dir}/output"
  local compose="${output_dir}/docker-compose.yml"
  mkdir -p "${scale_dir}"
  if [[ -f "${compose}" ]]; then
    fix_compose_gateways "${compose}"
    echo "output_exists scale=${scale} output_dir=${output_dir} counts=$(service_count "${compose}")"
    return 0
  fi

  echo "generate_start scale=${scale} at=$(date -Is)"
  if [[ -d "${TOPO}/output" ]]; then
    local backup="${scale_dir}/host_output_backup_$(date +%Y%m%d_%H%M%S)"
    echo "backing_up_existing_topology_output=${backup}"
    mv "${TOPO}/output" "${backup}"
  fi
  (
    cd "${TOPO}"
    PYTHONPATH="$(dirname "${TOPO}")${PYTHONPATH:+:${PYTHONPATH}}" /usr/bin/time -v "${GENERATE_PYTHON}" RR_1897.py "${scale}"
  ) > "${scale_dir}/generate_output.log" 2>&1
  if [[ ! -f "${TOPO}/output/docker-compose.yml" ]]; then
    echo "ERROR: RR_1897.py did not produce ${TOPO}/output/docker-compose.yml" >&2
    exit 1
  fi
  rm -rf "${output_dir}"
  mv "${TOPO}/output" "${output_dir}"
  fix_compose_gateways "${compose}"
  echo "generate_done scale=${scale} output_dir=${output_dir} counts=$(service_count "${compose}") at=$(date -Is)"
}

collect_after_run() {
  local run_dir="$1"
  local since="$2"
  {
    echo "kernel=$(uname -r)"
    echo "load=$(cat /proc/loadavg)"
    free -h
    docker info --format 'Containers={{.Containers}} Running={{.ContainersRunning}} Images={{.Images}} NFd={{.NFd}} NGoroutines={{.NGoroutines}}'
  } > "${run_dir}/post_status.txt" 2>&1 || true
  journalctl -k --since "${since}" --no-pager \
    | egrep -i 'BUG:|Oops|panic|WARNING:|lockdep|KASAN|refcount|use-after-free|general protection|unregister_netdevice|bird_rtbulk|fib_create_info' \
    > "${run_dir}/kernel_risk.log" 2>&1 || true
}

run_scale() {
  local scale="$1"
  local output_dir="${BIRD}/runs/F_reconf_multiscale_20260614_094101/scale_${scale}/output"
  local project="rtnl-amd-k2c-${scale}"
  local run_name="k2c-amd-${scale}_$(date +%Y%m%d_%H%M%S)"
  local run_dir="${PKG}/K2c/runs/${run_name}"
  local since
  since="$(date -Is)"
  mkdir -p "${run_dir}"

  echo "scale_start scale=${scale} project=${project} run_name=${run_name} at=${since}"
  require_kernel
  ensure_output "${scale}"
  cleanup_project "${project}"

  set +e
  (
    cd "${PKG}/K2c"
    env \
      VM_HOST=127.0.0.1 \
      VM_USER=lxl \
      RESET_VM=0 \
      VM_DOMAIN= \
      OUTPUT_DIR="${output_dir}" \
      PROJECT_NAME="${project}" \
      RUN_NAME="${run_name}" \
      RUN_DIR="${run_dir}" \
      K2C_KERNEL=6.8.12-birdrtbulk-k2c \
      K1_KERNEL=6.8.12-birdrtbulk-k2c \
      UP_BATCH="${UP_BATCH:-25}" \
      UP_JOBS="${UP_JOBS:-2}" \
      COMPOSE_PARALLEL_LIMIT="${COMPOSE_PARALLEL_LIMIT:-8}" \
      RESTART_JOBS="${RESTART_JOBS:-1}" \
      RESTART_DELAY="${RESTART_DELAY:-0.1}" \
      LOAD_THRESHOLD="${LOAD_THRESHOLD:-50}" \
      MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-14400}" \
      INITIAL_WRITE_JOBS="${INITIAL_WRITE_JOBS:-8}" \
      K1_DUMP_JOBS="${K1_DUMP_JOBS:-16}" \
      K1_BATCH="${K1_BATCH:-2000}" \
      K1_COMMAND_TIMEOUT="${K1_COMMAND_TIMEOUT:-1800}" \
      K1_OPS_PER_ATTR="${K1_OPS_PER_ATTR:-2000}" \
      DELTA_WRITE_JOBS="${DELTA_WRITE_JOBS:-1}" \
      OFFSET_JOBS="${OFFSET_JOBS:-16}" \
      K1_DELTA_BATCH="${K1_DELTA_BATCH:-2000}" \
      K1_DELTA_MIXED_BATCHES="${K1_DELTA_MIXED_BATCHES:-1}" \
      RUN_FULL_COMPARISON="${RUN_FULL_COMPARISON:-0}" \
      SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-60}" \
      /usr/bin/time -v ./run_3083_k2c_delta_optimized.sh
  ) 2>&1 | tee "${run_dir}/outer.log"
  local rc=${PIPESTATUS[0]}
  set -e

  echo "${rc}" > "${run_dir}/exit.status"
  collect_after_run "${run_dir}" "${since}"
  echo "scale_end scale=${scale} rc=${rc} run_dir=${run_dir} at=$(date -Is)"
  if [[ "${rc}" != "0" ]]; then
    echo "ERROR: scale ${scale} failed; leaving project ${project} for inspection" >&2
    exit "${rc}"
  fi
  cleanup_project "${project}"
}

for scale in ${SCALES_STR}; do
  run_scale "${scale}"
done

echo "finished_at=$(date -Is)"
echo "K2C_ALL_DONE"
