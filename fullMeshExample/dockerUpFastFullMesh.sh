#!/usr/bin/env bash
#
# Purpose: start the shared-image fast compose for fullMeshExample in batches.
# Inputs: output/docker-compose-fast.yml and seedemu-fullmesh-node:1078.
# Outputs: running Docker containers and logs/dockerBuild/dockerUpFast_*.log.
# Side effects: starts brdnode and rs_ix Docker containers.
# Expected context: run from fullMeshExample after buildFastFullMesh.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_FILE="./output/docker-compose-fast.yml"
PARALLEL_JOBS=4
BATCH_SIZE=50
LOG_DIR="./logs/dockerBuild"
LOG_FILE="${LOG_DIR}/dockerUpFast_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "missing $COMPOSE_FILE; run ./buildFastFullMesh.sh first" >&2
    exit 1
fi

logOutput() {
    # $1=message to print and append to the dockerUpFast log.
    local message="$1"
    echo "$message"
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $message" >> "$LOG_FILE"
}

start_time="$(date +%s)"
logOutput "===== Docker Compose fast up start ====="
logOutput "compose file: $COMPOSE_FILE"
logOutput "parallel jobs: $PARALLEL_JOBS"
logOutput "batch size: $BATCH_SIZE"

services="$(docker compose -f "$COMPOSE_FILE" config --services)"
if [[ -z "$services" ]]; then
    logOutput "ERROR: no services found in $COMPOSE_FILE"
    exit 1
fi

echo "$services" | xargs -n "$BATCH_SIZE" -P "$PARALLEL_JOBS" \
    docker compose -f "$COMPOSE_FILE" up -d >> "$LOG_FILE" 2>&1

end_time="$(date +%s)"
duration="$((end_time - start_time))"
logOutput "Total runtime: $((duration / 3600))h $(((duration % 3600) / 60))m $((duration % 60))s"
logOutput "===== Docker Compose fast up end ====="
logOutput "Full log: $LOG_FILE"
