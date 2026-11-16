#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VM_HOST="${VM_HOST:-192.168.122.60}"
VM_USER="${VM_USER:-lxl}"
PROJECT_NAME="${PROJECT_NAME:-topology-291e63c1}"
LIMIT_CONTAINERS="${LIMIT_CONTAINERS:-0}"
REMOTE_ASSET_DIR="${REMOTE_ASSET_DIR:-/home/lxl/bird/delta-assets}"
DELTA_FIB_METRIC="${DELTA_FIB_METRIC:-232}"
DELTA_JOBS="${DELTA_JOBS:-1}"
DELTA_SERIAL_DELAY="${DELTA_SERIAL_DELAY:-0}"
DELTA_ENABLE_ON_RESTART="${DELTA_ENABLE_ON_RESTART:-1}"
DELTA_IGNORE_INITIAL_FEED="${DELTA_IGNORE_INITIAL_FEED:-0}"
DELTA_RESTART_LOAD_GATE_EVERY="${DELTA_RESTART_LOAD_GATE_EVERY:-0}"
DELTA_RESTART_LOAD_THRESHOLD="${DELTA_RESTART_LOAD_THRESHOLD:-50}"
DELTA_RESTART_LOAD_POLL_INTERVAL="${DELTA_RESTART_LOAD_POLL_INTERVAL:-30}"
DELTA_RESTART_LOAD_MAX_WAIT="${DELTA_RESTART_LOAD_MAX_WAIT:-1800}"
LOCAL_BIRD_BINARY="${LOCAL_BIRD_BINARY:-${SCRIPT_DIR}/bird-2.17.2-delta/bird-focal}"
if [[ ! -f "${LOCAL_BIRD_BINARY}" ]]; then
  LOCAL_BIRD_BINARY="${SCRIPT_DIR}/bird-2.17.2-delta/bird"
fi
LOCAL_BIRDC_BINARY="${LOCAL_BIRDC_BINARY:-}"

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
  -o ConnectTimeout=10
)

usage() {
  cat <<EOF
Usage:
  $0 sync
  $0 install
  $0 install-agent
  $0 probe-bird-state
  $0 start-clean-bird
  $0 restart-bird
  $0 enable-delta
  $0 write-initial
  $0 stop-agent
  $0 start-agent
  $0 status

Environment:
  VM_HOST=${VM_HOST}
  VM_USER=${VM_USER}
  PROJECT_NAME=${PROJECT_NAME}
  LIMIT_CONTAINERS=${LIMIT_CONTAINERS}
  DELTA_FIB_METRIC=${DELTA_FIB_METRIC}
  DELTA_JOBS=${DELTA_JOBS}
  DELTA_SERIAL_DELAY=${DELTA_SERIAL_DELAY}
  DELTA_ENABLE_ON_RESTART=${DELTA_ENABLE_ON_RESTART}
  DELTA_IGNORE_INITIAL_FEED=${DELTA_IGNORE_INITIAL_FEED}
  DELTA_RESTART_LOAD_GATE_EVERY=${DELTA_RESTART_LOAD_GATE_EVERY}
  DELTA_RESTART_LOAD_THRESHOLD=${DELTA_RESTART_LOAD_THRESHOLD}
  DELTA_RESTART_LOAD_POLL_INTERVAL=${DELTA_RESTART_LOAD_POLL_INTERVAL}
  DELTA_RESTART_LOAD_MAX_WAIT=${DELTA_RESTART_LOAD_MAX_WAIT}
  LOCAL_BIRD_BINARY=${LOCAL_BIRD_BINARY}
  LOCAL_BIRDC_BINARY=${LOCAL_BIRDC_BINARY:-<leave existing container birdc>}

Stages:
  sync          Copy patched BIRD binaries and delta_fib_agent.py to the VM.
  install       Copy patched BIRD and agent into running brdnode containers and
                add /etc/bird/conf/delta.conf.
  install-agent Copy only delta_fib_agent.py into running brdnode containers.
  probe-bird-state
                Probe one random brdnode and report whether BIRD is already
                reachable, only running as a process, or not started.
  start-clean-bird
                Start BIRD in brdnode containers that are expected to have no
                existing BIRD process. This avoids the heavier restart path.
  restart-bird  Restart BIRD in each brdnode using the patched binary. This
                truncates /run/bird/delta-events.tsv first.
  enable-delta   Add /etc/bird/conf/delta.conf and reconfigure running BIRD.
  write-initial Replay the current delta event file once, collapse to the final
                event per prefix, and write main FIB.
  stop-agent    Stop resident delta agents without changing BIRD or FIB.
  start-agent   Start a resident per-container agent from the current event-file
                end; only future BIRD deltas are written.
  status        Show delta protocol and agent status in sample containers.
EOF
}

ssh_vm() {
  ssh "${SSH_OPTS[@]}" "${VM_USER}@${VM_HOST}" "$@"
}

phase_sync() {
  ssh_vm "mkdir -p '${REMOTE_ASSET_DIR}'"
  rsync -a -e "ssh ${SSH_OPTS[*]}" \
    "${LOCAL_BIRD_BINARY}" \
    "${SCRIPT_DIR}/delta_fib_agent.py" \
    "${VM_USER}@${VM_HOST}:${REMOTE_ASSET_DIR}/"
  local remote_bird_name
  remote_bird_name="$(basename "${LOCAL_BIRD_BINARY}")"
  if [[ "${remote_bird_name}" != "bird" ]]; then
    ssh_vm "mv '${REMOTE_ASSET_DIR}/${remote_bird_name}' '${REMOTE_ASSET_DIR}/bird'"
  fi
  ssh_vm "rm -f '${REMOTE_ASSET_DIR}/birdc'"
  if [[ -n "${LOCAL_BIRDC_BINARY}" ]]; then
    rsync -a -e "ssh ${SSH_OPTS[*]}" "${LOCAL_BIRDC_BINARY}" "${VM_USER}@${VM_HOST}:${REMOTE_ASSET_DIR}/birdc"
  fi
}

remote_phase() {
  local phase="$1"
  ssh_vm "PROJECT_NAME='${PROJECT_NAME}' LIMIT_CONTAINERS='${LIMIT_CONTAINERS}' REMOTE_ASSET_DIR='${REMOTE_ASSET_DIR}' DELTA_FIB_METRIC='${DELTA_FIB_METRIC}' DELTA_JOBS='${DELTA_JOBS}' DELTA_SERIAL_DELAY='${DELTA_SERIAL_DELAY}' DELTA_ENABLE_ON_RESTART='${DELTA_ENABLE_ON_RESTART}' DELTA_IGNORE_INITIAL_FEED='${DELTA_IGNORE_INITIAL_FEED}' DELTA_RESTART_LOAD_GATE_EVERY='${DELTA_RESTART_LOAD_GATE_EVERY}' DELTA_RESTART_LOAD_THRESHOLD='${DELTA_RESTART_LOAD_THRESHOLD}' DELTA_RESTART_LOAD_POLL_INTERVAL='${DELTA_RESTART_LOAD_POLL_INTERVAL}' DELTA_RESTART_LOAD_MAX_WAIT='${DELTA_RESTART_LOAD_MAX_WAIT}' PHASE='${phase}' bash -s" <<'REMOTE'
set -euo pipefail
export PROJECT_NAME LIMIT_CONTAINERS REMOTE_ASSET_DIR DELTA_FIB_METRIC DELTA_JOBS DELTA_SERIAL_DELAY DELTA_ENABLE_ON_RESTART DELTA_IGNORE_INITIAL_FEED DELTA_RESTART_LOAD_GATE_EVERY DELTA_RESTART_LOAD_THRESHOLD DELTA_RESTART_LOAD_POLL_INTERVAL DELTA_RESTART_LOAD_MAX_WAIT PHASE

router_container_rows() {
  docker ps --filter "label=com.docker.compose.project=${PROJECT_NAME}" --format '{{.ID}} {{.Names}}' \
    | awk -v limit="${LIMIT_CONTAINERS}" '
        $2 ~ /^as[0-9]+brd-r/ {
          count++;
          if (limit == "0" || count <= limit) print $1, $2;
        }
      '
}

router_containers() {
  router_container_rows | awk '{print $1}'
}

install_one() {
  local cid="$1"
  docker cp "${REMOTE_ASSET_DIR}/bird" "${cid}:/tmp/bird.delta"
  docker cp "${REMOTE_ASSET_DIR}/delta_fib_agent.py" "${cid}:/tmp/delta_fib_agent.py"
  if [ -s "${REMOTE_ASSET_DIR}/birdc" ]; then
    docker cp "${REMOTE_ASSET_DIR}/birdc" "${cid}:/tmp/birdc.delta"
  fi
  docker exec "${cid}" sh -lc '
    set -e
    install -m 0755 /tmp/bird.delta /usr/sbin/bird
    if [ -s /tmp/birdc.delta ]; then install -m 0755 /tmp/birdc.delta /usr/sbin/birdc; fi
    install -m 0755 /tmp/delta_fib_agent.py /usr/local/bin/delta_fib_agent.py
    rm -f /tmp/bird.delta /tmp/birdc.delta /tmp/delta_fib_agent.py
    mkdir -p /etc/bird/conf /run/bird
    cat > /etc/bird/conf/delta.conf <<'"'"'EOF'"'"'
protocol delta delta_main {
    event "/run/bird/delta-events.tsv";
    metric '"${DELTA_FIB_METRIC}"';
    ignore initial feed '"${DELTA_IGNORE_INITIAL_FEED}"';
    merge paths on limit 16;
    ipv4 {
        table master4;
        export all;
    };
}
EOF
    sed -i "/\\/etc\\/bird\\/conf\\/delta.conf/d" /etc/bird/bird.conf
    grep -Eq "include[[:space:]]+\"/etc/bird/conf/\\*\\.conf\";" /etc/bird/bird.conf 2>/dev/null \
      || printf "\ninclude \"/etc/bird/conf/delta.conf\";\n" >> /etc/bird/bird.conf
  '
}

install_agent_one() {
  local cid="$1"
  docker cp "${REMOTE_ASSET_DIR}/delta_fib_agent.py" "${cid}:/tmp/delta_fib_agent.py"
  docker exec "${cid}" sh -lc '
    set -e
    install -m 0755 /tmp/delta_fib_agent.py /usr/local/bin/delta_fib_agent.py
    rm -f /tmp/delta_fib_agent.py
  '
}

start_clean_bird_one() {
  local cid="$1"
  docker exec -e DELTA_ENABLE_ON_RESTART="${DELTA_ENABLE_ON_RESTART}" "${cid}" sh -lc '
    set -e
    mkdir -p /run/bird
    if birdc -s /run/bird/bird.ctl show status >/dev/null 2>&1 || pgrep -x bird >/dev/null 2>&1; then
      echo "BIRD already appears to be running; use restart-bird for this project" >&2
      exit 3
    fi
    : > /run/bird/delta-events.tsv
    rm -f /run/bird/bird.pid /run/bird/bird.ctl
    sed -i "/\\/etc\\/bird\\/conf\\/delta.conf/d" /etc/bird/bird.conf
    if [ "${DELTA_ENABLE_ON_RESTART}" = "0" ]; then
      if [ -f /etc/bird/conf/delta.conf ]; then
        mv /etc/bird/conf/delta.conf /etc/bird/conf/delta.conf.disabled
      fi
    else
      if [ -f /etc/bird/conf/delta.conf.disabled ] && [ ! -f /etc/bird/conf/delta.conf ]; then
        mv /etc/bird/conf/delta.conf.disabled /etc/bird/conf/delta.conf
      fi
      grep -Eq "include[[:space:]]+\"/etc/bird/conf/\\*\\.conf\";" /etc/bird/bird.conf 2>/dev/null \
        || printf "\ninclude \"/etc/bird/conf/delta.conf\";\n" >> /etc/bird/bird.conf
    fi
    bird -p -c /etc/bird/bird.conf
    bird -s /run/bird/bird.ctl -c /etc/bird/bird.conf
    ready=0
    for _ in $(seq 1 50); do
      if birdc -s /run/bird/bird.ctl show status >/dev/null 2>&1; then
        ready=1
        break
      fi
      sleep 0.2
    done
    [ "${ready}" = "1" ]
  '
}

restart_bird_one() {
  local cid="$1"
  docker exec -e DELTA_ENABLE_ON_RESTART="${DELTA_ENABLE_ON_RESTART}" "${cid}" sh -lc '
    set -e
    mkdir -p /run/bird
    if [ -s /run/bird/delta-fib-agent.pid ]; then
      old_agent="$(cat /run/bird/delta-fib-agent.pid 2>/dev/null || true)"
      if [ -n "${old_agent}" ] && kill -0 "${old_agent}" >/dev/null 2>&1; then
        kill "${old_agent}" >/dev/null 2>&1 || true
      fi
      rm -f /run/bird/delta-fib-agent.pid
    fi
    : > /run/bird/delta-events.tsv
    birdc down >/dev/null 2>&1 || true
    pkill bird >/dev/null 2>&1 || true
    for _ in $(seq 1 50); do
      if ! pgrep -x bird >/dev/null 2>&1; then
        break
      fi
      sleep 0.2
    done
    if pgrep -x bird >/dev/null 2>&1; then
      pkill -9 bird >/dev/null 2>&1 || true
      sleep 0.2
    fi
    rm -f /run/bird/bird.pid /run/bird/bird.ctl
    sed -i "/\\/etc\\/bird\\/conf\\/delta.conf/d" /etc/bird/bird.conf
    if [ "${DELTA_ENABLE_ON_RESTART}" = "0" ]; then
      if [ -f /etc/bird/conf/delta.conf ]; then
        mv /etc/bird/conf/delta.conf /etc/bird/conf/delta.conf.disabled
      fi
    else
      if [ -f /etc/bird/conf/delta.conf.disabled ] && [ ! -f /etc/bird/conf/delta.conf ]; then
        mv /etc/bird/conf/delta.conf.disabled /etc/bird/conf/delta.conf
      fi
      grep -Eq "include[[:space:]]+\"/etc/bird/conf/\\*\\.conf\";" /etc/bird/bird.conf 2>/dev/null \
        || printf "\ninclude \"/etc/bird/conf/delta.conf\";\n" >> /etc/bird/bird.conf
    fi
    bird -p -c /etc/bird/bird.conf
    bird -s /run/bird/bird.ctl -c /etc/bird/bird.conf
    ready=0
    for _ in $(seq 1 50); do
      if birdc -s /run/bird/bird.ctl show status >/dev/null 2>&1; then
        ready=1
        break
      fi
      sleep 0.2
    done
    [ "${ready}" = "1" ]
  '
}

enable_delta_one() {
  local cid="$1"
  docker exec "${cid}" sh -lc '
    set -e
    mkdir -p /etc/bird/conf /run/bird
    cat > /etc/bird/conf/delta.conf <<'"'"'EOF'"'"'
protocol delta delta_main {
    event "/run/bird/delta-events.tsv";
    metric '"${DELTA_FIB_METRIC}"';
    ignore initial feed '"${DELTA_IGNORE_INITIAL_FEED}"';
    merge paths on limit 16;
    ipv4 {
        table master4;
        export all;
    };
}
EOF
    sed -i "/\\/etc\\/bird\\/conf\\/delta.conf/d" /etc/bird/bird.conf
    grep -Eq "include[[:space:]]+\"/etc/bird/conf/\\*\\.conf\";" /etc/bird/bird.conf 2>/dev/null \
      || printf "\ninclude \"/etc/bird/conf/delta.conf\";\n" >> /etc/bird/bird.conf
    birdc -s /run/bird/bird.ctl configure >/dev/null
    ready=0
    for _ in $(seq 1 50); do
      if birdc -s /run/bird/bird.ctl show protocols 2>/dev/null | awk "/delta_main/ && \$4 == \"up\" {found=1} END {exit found ? 0 : 1}"; then
        ready=1
        break
      fi
      sleep 0.2
    done
    [ "${ready}" = "1" ]
  '
}

write_initial_one() {
  local cid="$1"
  docker exec "${cid}" sh -lc '
    set -e
    ip -4 route flush proto bird metric '"${DELTA_FIB_METRIC}"' >/dev/null 2>&1 || true
    ip -6 route flush proto bird metric '"${DELTA_FIB_METRIC}"' >/dev/null 2>&1 || true
    /usr/local/bin/delta_fib_agent.py \
      --event-file /run/bird/delta-events.tsv \
      --stats-file /run/bird/delta-fib-initial.json \
      --final-collapse \
      --keep-going
  ' >/dev/null
}

stop_agent_one() {
  local cid="$1"
  docker exec "${cid}" sh -lc '
    set -e
    if [ -s /run/bird/delta-fib-agent.pid ]; then
      old_pid="$(cat /run/bird/delta-fib-agent.pid 2>/dev/null || true)"
      if [ -n "${old_pid}" ] && kill -0 "${old_pid}" >/dev/null 2>&1; then
        kill "${old_pid}" >/dev/null 2>&1 || true
      fi
    fi
    rm -f /run/bird/delta-fib-agent.pid
  '
}

start_agent_one() {
  local cid="$1"
  docker exec "${cid}" sh -lc '
    set -e
    mkdir -p /run/bird
    if [ -s /run/bird/delta-fib-agent.pid ]; then
      old_pid="$(cat /run/bird/delta-fib-agent.pid 2>/dev/null || true)"
      if [ -n "${old_pid}" ] && kill -0 "${old_pid}" >/dev/null 2>&1; then
        kill "${old_pid}" >/dev/null 2>&1 || true
      fi
    fi
    rm -f /run/bird/delta-fib-agent.json /run/bird/delta-fib-agent.log
    nohup /usr/local/bin/delta_fib_agent.py \
      --event-file /run/bird/delta-events.tsv \
      --from-end \
      --follow \
      --stats-file /run/bird/delta-fib-agent.json \
      --keep-going \
      >/run/bird/delta-fib-agent.log 2>&1 &
    echo $! > /run/bird/delta-fib-agent.pid
  '
}

status_one() {
  local cid="$1"
  docker exec "${cid}" sh -lc '
    printf "%s " "$(hostname)"
    birdc show protocols 2>/dev/null | awk "/delta_main/ {print \$1,\$4,\$5}" | tr "\n" " "
    printf "events="
    wc -l /run/bird/delta-events.tsv 2>/dev/null | awk "{print \$1}" | tr "\n" " "
    printf "agent="
    if [ -s /run/bird/delta-fib-agent.pid ] && kill -0 "$(cat /run/bird/delta-fib-agent.pid)" >/dev/null 2>&1; then echo 1; else echo 0; fi
  '
}

probe_bird_state() {
  local tmp
  tmp="$(mktemp)"
  router_container_rows > "${tmp}"
  local total
  total="$(wc -l < "${tmp}" | tr -d " ")"
  if [ "${total}" = "0" ]; then
    rm -f "${tmp}"
    python3 - "${PROJECT_NAME}" <<'PY'
import datetime as dt
import json
import sys

print(json.dumps({
    "project_name": sys.argv[1],
    "state": "no_brd_containers",
    "brdnode_count": 0,
    "sample_method": "random",
    "checked_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
}, sort_keys=True))
PY
    return
  fi

  local row
  if command -v shuf >/dev/null 2>&1; then
    row="$(shuf -n 1 "${tmp}")"
  else
    row="$(awk 'BEGIN { srand() } { rows[NR] = $0 } END { if (NR) print rows[int(rand() * NR) + 1] }' "${tmp}")"
  fi
  rm -f "${tmp}"

  local cid name raw rc
  cid="${row%% *}"
  name="${row#* }"
  raw="$(docker exec "${cid}" sh -lc '
    bird_process=0
    pgrep -x bird >/dev/null 2>&1 && bird_process=1
    birdc_status=0
    birdc_output="$(birdc -s /run/bird/bird.ctl show status 2>&1)" && birdc_status=1 || true
    ctl_socket=0
    [ -S /run/bird/bird.ctl ] && ctl_socket=1
    pid_file=0
    [ -s /run/bird/bird.pid ] && pid_file=1
    delta_conf=0
    [ -f /etc/bird/conf/delta.conf ] && delta_conf=1
    delta_conf_disabled=0
    [ -f /etc/bird/conf/delta.conf.disabled ] && delta_conf_disabled=1
    bird_version="$(bird --version 2>&1 | head -n 1 || true)"
    printf "bird_process=%s\n" "${bird_process}"
    printf "birdc_status=%s\n" "${birdc_status}"
    printf "ctl_socket=%s\n" "${ctl_socket}"
    printf "pid_file=%s\n" "${pid_file}"
    printf "delta_conf=%s\n" "${delta_conf}"
    printf "delta_conf_disabled=%s\n" "${delta_conf_disabled}"
    printf "bird_version=%s\n" "${bird_version}"
    printf "birdc_output=%s\n" "$(printf "%s" "${birdc_output}" | tr "\n" " " | head -c 300)"
  ' 2>&1)" || rc=$?
  rc="${rc:-0}"

  PROBE_RAW="${raw}" python3 - "${PROJECT_NAME}" "${total}" "${cid}" "${name}" "${rc}" <<'PY'
import datetime as dt
import json
import os
import sys

values = {}
raw = os.environ.get("PROBE_RAW", "")
for line in raw.splitlines():
    key, sep, value = line.partition("=")
    if sep:
        values[key] = value

bird_process = values.get("bird_process") == "1"
birdc_status = values.get("birdc_status") == "1"
if birdc_status:
    state = "bird_running"
elif bird_process:
    state = "bird_process_only"
else:
    state = "bird_not_started"

print(json.dumps({
    "project_name": sys.argv[1],
    "brdnode_count": int(sys.argv[2]),
    "container_id": sys.argv[3],
    "container_name": sys.argv[4],
    "docker_exec_rc": int(sys.argv[5]),
    "sample_method": "random",
    "state": state,
    "bird_process": bird_process,
    "birdc_status": birdc_status,
    "control_socket": values.get("ctl_socket") == "1",
    "pid_file": values.get("pid_file") == "1",
    "delta_conf": values.get("delta_conf") == "1",
    "delta_conf_disabled": values.get("delta_conf_disabled") == "1",
    "bird_version": values.get("bird_version", ""),
    "birdc_output": values.get("birdc_output", ""),
    "checked_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
}, sort_keys=True))
PY
}

restart_load_gate() {
  local count="$1"
  if { [ "${PHASE}" != "restart-bird" ] && [ "${PHASE}" != "start-clean-bird" ]; } || [ "${DELTA_RESTART_LOAD_GATE_EVERY}" = "0" ]; then
    return
  fi
  if [ $((count % DELTA_RESTART_LOAD_GATE_EVERY)) -ne 0 ]; then
    return
  fi

  local start_epoch elapsed load1
  start_epoch="$(date +%s)"
  while true; do
    load1="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 9999)"
    echo "${PHASE}: load gate count=${count} load1=${load1} threshold=${DELTA_RESTART_LOAD_THRESHOLD}"
    if awk -v loadval="${load1}" -v threshold="${DELTA_RESTART_LOAD_THRESHOLD}" 'BEGIN { exit !(loadval < threshold) }'; then
      return
    fi
    elapsed=$(( $(date +%s) - start_epoch ))
    if [ "${elapsed}" -ge "${DELTA_RESTART_LOAD_MAX_WAIT}" ]; then
      echo "${PHASE}: load gate timed out after ${elapsed}s at count=${count} load1=${load1}" >&2
      return 1
    fi
    sleep "${DELTA_RESTART_LOAD_POLL_INTERVAL}"
  done
}

map_phase() {
  local phase="$1"
  local tmp
  tmp="$(mktemp)"
  router_containers > "${tmp}"
  local total
  total="$(wc -l < "${tmp}" | tr -d " ")"
  if [ "${total}" = "0" ]; then
    rm -f "${tmp}"
    echo "${phase}: processed 0 containers"
    return
  fi

  if [ "${DELTA_JOBS}" != "1" ]; then
    export -f install_one install_agent_one start_clean_bird_one restart_bird_one enable_delta_one write_initial_one stop_agent_one start_agent_one status_one
    xargs -r -n 1 -P "${DELTA_JOBS}" bash -c '
      cid="$1"
      case "${PHASE}" in
        install) install_one "${cid}" ;;
        install-agent) install_agent_one "${cid}" ;;
        start-clean-bird) start_clean_bird_one "${cid}" ;;
        restart-bird) restart_bird_one "${cid}" ;;
        enable-delta) enable_delta_one "${cid}" ;;
        write-initial) write_initial_one "${cid}" ;;
        stop-agent) stop_agent_one "${cid}" ;;
        start-agent) start_agent_one "${cid}" ;;
        status) status_one "${cid}" ;;
        *) echo "unknown phase: ${PHASE}" >&2; exit 2 ;;
      esac
    ' _ < "${tmp}"
    rm -f "${tmp}"
    echo "${phase}: processed ${total} containers with jobs=${DELTA_JOBS}"
    return
  fi

  local count=0
  while read -r cid; do
    [ -n "${cid}" ] || continue
    count=$((count + 1))
    case "${phase}" in
      install) install_one "${cid}" ;;
      install-agent) install_agent_one "${cid}" ;;
      start-clean-bird) start_clean_bird_one "${cid}" ;;
      restart-bird) restart_bird_one "${cid}" ;;
      enable-delta) enable_delta_one "${cid}" ;;
      write-initial) write_initial_one "${cid}" ;;
      stop-agent) stop_agent_one "${cid}" ;;
      start-agent) start_agent_one "${cid}" ;;
      status) status_one "${cid}" ;;
      *) echo "unknown phase: ${phase}" >&2; exit 2 ;;
    esac
    if [ $((count % 100)) -eq 0 ]; then
      echo "${phase}: processed ${count} containers"
    fi
    if { [ "${phase}" = "restart-bird" ] || [ "${phase}" = "start-clean-bird" ]; } && [ "${DELTA_SERIAL_DELAY}" != "0" ]; then
      sleep "${DELTA_SERIAL_DELAY}"
    fi
    restart_load_gate "${count}"
  done < "${tmp}"
  rm -f "${tmp}"
  echo "${phase}: processed ${count} containers"
}

if [ "${PHASE}" = "probe-bird-state" ]; then
  probe_bird_state
else
  map_phase "${PHASE}"
fi
REMOTE
}

phase="${1:-}"
case "${phase}" in
  sync) phase_sync ;;
  install) phase_sync; remote_phase install ;;
  install-agent) phase_sync; remote_phase install-agent ;;
  probe-bird-state) remote_phase probe-bird-state ;;
  start-clean-bird) remote_phase start-clean-bird ;;
  restart-bird) remote_phase restart-bird ;;
  enable-delta) remote_phase enable-delta ;;
  write-initial) remote_phase write-initial ;;
  stop-agent) remote_phase stop-agent ;;
  start-agent) remote_phase start-agent ;;
  status) remote_phase status ;;
  -h|--help|help|"") usage ;;
  *) usage >&2; exit 2 ;;
esac
