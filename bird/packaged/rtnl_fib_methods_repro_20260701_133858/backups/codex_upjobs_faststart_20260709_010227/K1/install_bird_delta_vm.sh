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
  LOCAL_BIRD_BINARY=${LOCAL_BIRD_BINARY}
  LOCAL_BIRDC_BINARY=${LOCAL_BIRDC_BINARY:-<leave existing container birdc>}

Stages:
  sync          Copy patched BIRD binaries and delta_fib_agent.py to the VM.
  install       Copy patched BIRD and agent into running brdnode containers and
                add /etc/bird/conf/delta.conf.
  install-agent Copy only delta_fib_agent.py into running brdnode containers.
  probe-bird-state  Probe whether BIRD is already running in brdnodes.
  start-clean-bird  Start BIRD in fresh brdnodes without birdc down/pkill.
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
  ssh_vm "PROJECT_NAME='${PROJECT_NAME}' LIMIT_CONTAINERS='${LIMIT_CONTAINERS}' REMOTE_ASSET_DIR='${REMOTE_ASSET_DIR}' DELTA_FIB_METRIC='${DELTA_FIB_METRIC}' DELTA_JOBS='${DELTA_JOBS}' DELTA_SERIAL_DELAY='${DELTA_SERIAL_DELAY}' DELTA_ENABLE_ON_RESTART='${DELTA_ENABLE_ON_RESTART}' DELTA_IGNORE_INITIAL_FEED='${DELTA_IGNORE_INITIAL_FEED}' PHASE='${phase}' bash -s" <<'REMOTE'
set -euo pipefail
export PROJECT_NAME LIMIT_CONTAINERS REMOTE_ASSET_DIR DELTA_FIB_METRIC DELTA_JOBS DELTA_SERIAL_DELAY DELTA_ENABLE_ON_RESTART DELTA_IGNORE_INITIAL_FEED PHASE

router_containers() {
  docker ps --filter "label=com.docker.compose.project=${PROJECT_NAME}" --format '{{.ID}} {{.Names}}' \
    | awk -v limit="${LIMIT_CONTAINERS}" '
        $2 ~ /^as[0-9]+brd-r/ {
          count++;
          if (limit == "0" || count <= limit) print $1;
        }
      '
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

start_clean_bird_one() {
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

probe_bird_state_one() {
  local cid="$1"
  docker exec "${cid}" sh -lc '
    if birdc -s /run/bird/bird.ctl show status >/dev/null 2>&1 || birdc show status >/dev/null 2>&1; then
      echo bird_running
    elif pgrep -x bird >/dev/null 2>&1; then
      echo bird_running
    else
      echo bird_not_started
    fi
  ' 2>/dev/null || echo unknown
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

map_phase() {
  local phase="$1"
  local tmp
  tmp="$(mktemp)"
  router_containers > "${tmp}"
  local total
  total="$(wc -l < "${tmp}" | tr -d " ")"
  if [ "${phase}" = "probe-bird-state" ]; then
    local sampled=0
    local running=0
    local not_started=0
    local unknown=0
    while read -r cid; do
      [ -n "${cid}" ] || continue
      sampled=$((sampled + 1))
      case "$(probe_bird_state_one "${cid}")" in
        bird_running) running=$((running + 1)) ;;
        bird_not_started) not_started=$((not_started + 1)) ;;
        *) unknown=$((unknown + 1)) ;;
      esac
      [ "${sampled}" -ge 20 ] && break
    done < "${tmp}"
    rm -f "${tmp}"
    local state
    if [ "${total}" = "0" ]; then
      state="no_brd_containers"
    elif [ "${running}" -gt 0 ]; then
      state="bird_running"
    elif [ "${unknown}" -gt 0 ]; then
      state="unknown"
    else
      state="bird_not_started"
    fi
    printf '{"state":"%s","sampled":%s,"running":%s,"not_started":%s,"unknown":%s,"total":%s}\n' \
      "${state}" "${sampled}" "${running}" "${not_started}" "${unknown}" "${total}"
    return
  fi
  if [ "${total}" = "0" ]; then
    rm -f "${tmp}"
    echo "${phase}: processed 0 containers"
    return
  fi

  if [ "${DELTA_JOBS}" != "1" ]; then
    export -f install_one install_agent_one restart_bird_one start_clean_bird_one enable_delta_one write_initial_one stop_agent_one start_agent_one status_one
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
  done < "${tmp}"
  rm -f "${tmp}"
  echo "${phase}: processed ${count} containers"
}

map_phase "${PHASE}"
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
