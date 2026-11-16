#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export K1_KERNEL="${K2C_KERNEL:-6.8.12-birdrtbulk-k2c}"
export RUN_NAME="${RUN_NAME:-k2c-delta-reconv-3083_$(date +%Y%m%d_%H%M%S)}"
export RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/runs/${RUN_NAME}}"

exec "${SCRIPT_DIR}/run_3083_k1_delta_reconvergence.sh" "$@"
