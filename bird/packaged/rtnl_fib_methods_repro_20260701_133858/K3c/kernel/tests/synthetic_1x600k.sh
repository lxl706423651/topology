#!/usr/bin/env bash
set -euo pipefail

WRITER="${WRITER:-/home/lxl/bird/kernel/tools/kernel_bulk_writer.py}"
RUN_DIR="${RUN_DIR:-$HOME/bird/runs/k1_synthetic_1x600k_$(date +%Y%m%d_%H%M%S)}"
NETNS="${NETNS:-k1bulk-r000}"
ROUTES="${ROUTES:-600000}"
BATCH="${BATCH:-1024}"

mkdir -p "${RUN_DIR}"

cleanup() {
  ip netns del "${NETNS}" 2>/dev/null || true
}

cleanup
ip netns add "${NETNS}"
ip -n "${NETNS}" link set lo up

started="$(python3 -c 'import time; print(time.monotonic())')"
python3 "${WRITER}" \
  --netns-path "/var/run/netns/${NETNS}" \
  --routes "${ROUTES}" \
  --batch "${BATCH}" \
  --base 10.0.0.0 \
  --route-type dev \
  --dev lo \
  --op replace \
  --summary-path "${RUN_DIR}/summary.json" \
  | tee "${RUN_DIR}/stdout.log"
ended="$(python3 -c 'import time; print(time.monotonic())')"

ip netns exec "${NETNS}" sh -lc "ip route | wc -l" > "${RUN_DIR}/ip_route_wc.txt"
cat /proc/loadavg > "${RUN_DIR}/loadavg_after.txt"

cleanup

cat > "${RUN_DIR}/run_meta.json" <<EOF
{
  "case": "k1_synthetic_1x600k",
  "netns": "${NETNS}",
  "routes": ${ROUTES},
  "batch": ${BATCH},
  "wall_seconds_outer": $(python3 - <<PY
print(float("${ended}") - float("${started}"))
PY
)
}
EOF

echo "RUN_DIR=${RUN_DIR}"
