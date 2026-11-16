#!/usr/bin/env bash
# Purpose: compatibility entrypoint for the fast SeedEMU Docker image builder.
# Inputs: optional path to output/docker-compose.yml, plus any Python flags.
# Outputs: logs/dockerBuild/dockerBuild_*.log and output/.fastDockerBuild/*.
# Side effects: delegates to buildFastDockerImages.py, which may pull/build/load
# Docker images in the local Docker image store.
# Context: run from the topology directory after SeedEMU has compiled ./output.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/buildFastDockerImages.py" "$@"
