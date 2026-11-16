#!/usr/bin/env bash
#
# Purpose: generate the connected 1078 topology JSON and compile it with the
# local RR_1897.py copy.
# Inputs: optional $1=size suffix, optional $2=topology JSON path.
# Outputs: real_topology_<size>_connected.json, its report, and ./output.
# Side effects: replaces ./output through SeedEMU compile override=True.
# Expected context: run from fullMeshExample or by absolute path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SIZE="${1:-1078}"
TOPOLOGY_FILE="${2:-real_topology_${SIZE}_connected.json}"
SOURCE_FILE="real_topology_${SIZE}.txt"

if [[ ! -f "$TOPOLOGY_FILE" ]]; then
    python3 connectTopology.py "$SOURCE_FILE" -o "$TOPOLOGY_FILE"
fi

python3 RR_1897.py "$SIZE" --topology "$TOPOLOGY_FILE" --output-dir ./output --quiet
python3 patchRuntimeRouting.py --compose ./output/docker-compose.yml
