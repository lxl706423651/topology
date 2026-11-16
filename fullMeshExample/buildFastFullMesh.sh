#!/usr/bin/env bash
#
# Purpose: build one shared runtime image and prepare fast compose for fullMeshExample.
# Inputs: fastNode.Dockerfile and output/docker-compose.yml.
# Outputs: seedemu-fullmesh-node:1078 and output/docker-compose-fast.yml.
# Side effects: builds one Docker image.
# Expected context: run from fullMeshExample after compileFullMesh.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="${1:-seedemu-fullmesh-node:1078}"

if [[ ! -f ./output/docker-compose.yml ]]; then
    echo "missing ./output/docker-compose.yml; run ./compileFullMesh.sh first" >&2
    exit 1
fi

python3 prepareFastCompose.py --input ./output/docker-compose.yml --output ./output/docker-compose-fast.yml --image "$IMAGE_NAME"
docker build -f fastNode.Dockerfile -t "$IMAGE_NAME" .
