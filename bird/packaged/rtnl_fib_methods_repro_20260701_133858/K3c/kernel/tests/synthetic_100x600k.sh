#!/usr/bin/env bash
set -euo pipefail

WRITER="${WRITER:-/home/lxl/bird/kernel/tools/kernel_bulk_writer.py}"
RUN_DIR="${RUN_DIR:-$HOME/bird/runs/k1_synthetic_100x600k_$(date +%Y%m%d_%H%M%S)}"
PREFIX="${PREFIX:-k1bulk-r}"
ROUTERS="${ROUTERS:-100}"
ROUTES_PER_ROUTER="${ROUTES_PER_ROUTER:-600000}"
BATCH="${BATCH:-1024}"
JOBS="${JOBS:-4}"

mkdir -p "${RUN_DIR}/summaries" "${RUN_DIR}/logs"

cleanup() {
  ip netns list | awk '{print $1}' | while read -r ns; do
    case "${ns}" in
      "${PREFIX}"*) ip netns del "${ns}" ;;
    esac
  done
}

cleanup

for i in $(seq 0 $((ROUTERS - 1))); do
  ns="$(printf '%s%03d' "${PREFIX}" "${i}")"
  ip netns add "${ns}"
  ip -n "${ns}" link set lo up
done

run_one() {
  i="$1"
  ns="$(printf '%s%03d' "${PREFIX}" "${i}")"
  offset=$((i * ROUTES_PER_ROUTER))
  python3 "${WRITER}" \
    --netns-path "/var/run/netns/${ns}" \
    --routes "${ROUTES_PER_ROUTER}" \
    --batch "${BATCH}" \
    --base 10.0.0.0 \
    --start-offset "${offset}" \
    --route-type dev \
    --dev lo \
    --op replace \
    --summary-path "${RUN_DIR}/summaries/${ns}.json" \
    > "${RUN_DIR}/logs/${ns}.stdout" \
    2> "${RUN_DIR}/logs/${ns}.stderr"
}
export -f run_one
export WRITER RUN_DIR PREFIX ROUTES_PER_ROUTER BATCH

started="$(python3 -c 'import time; print(time.monotonic())')"
seq 0 $((ROUTERS - 1)) | xargs -n1 -P "${JOBS}" bash -lc 'run_one "$@"' _
ended="$(python3 -c 'import time; print(time.monotonic())')"

python3 - <<PY > "${RUN_DIR}/summary.json"
import json
from pathlib import Path
run = Path("${RUN_DIR}")
rows = []
for path in sorted((run / "summaries").glob("*.json")):
    rows.append(json.loads(path.read_text()))
success = sum(r.get("success_count", 0) for r in rows)
fail = sum(r.get("fail_count", 0) for r in rows)
wall = float("${ended}") - float("${started}")
print(json.dumps({
    "case": "k1_synthetic_100x600k",
    "routers": ${ROUTERS},
    "routes_per_router": ${ROUTES_PER_ROUTER},
    "batch": ${BATCH},
    "jobs": ${JOBS},
    "successful_routes": success,
    "failed_routes": fail,
    "wall_seconds_outer": wall,
    "aggregate_routes_per_second": success / wall if wall else None,
    "writer_summaries": len(rows),
}, indent=2, sort_keys=True))
PY

cat /proc/loadavg > "${RUN_DIR}/loadavg_after.txt"
cleanup

echo "RUN_DIR=${RUN_DIR}"
