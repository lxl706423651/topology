#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export K2C_KERNEL="${K2C_KERNEL:-6.8.12-birdrtbulk-k2c}"
export RUN_NAME="${RUN_NAME:-k2c-delta-mixed-3083_$(date +%Y%m%d_%H%M%S)}"
export RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/runs/${RUN_NAME}}"

# Sparse reconvergence deltas produce very small per-container command files.
# Mixed batches combine replace/delete ops for each netns and avoid two tiny
# rtbulk submissions per router.
export K1_DELTA_MIXED_BATCHES="${K1_DELTA_MIXED_BATCHES:-1}"

# Offset preparation is Docker exec heavy, not FIB-write heavy. Keep it
# independent from the rtbulk writer jobs so conservative writer sweeps do not
# serialize thousands of offset writes.
export OFFSET_JOBS="${OFFSET_JOBS:-16}"

# K2c full writes benefit from parallelism, but sparse delta writes can spend
# most time contending on fib_info_hash_mutex. Start conservatively.
export DELTA_WRITE_JOBS="${DELTA_WRITE_JOBS:-1}"
export K1_DELTA_BATCH="${K1_DELTA_BATCH:-2000}"

exec "${SCRIPT_DIR}/run_3083_k2c_delta_reconvergence.sh" "$@"
