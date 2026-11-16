#!/usr/bin/env bash
# Purpose: start the compiled Compose topology in bounded parallel batches.
# Inputs: config.yaml; optional --compose, --project-name and --experiment-dir.
# Outputs: evaluationResult logs. Side effects: create/start Docker resources.
# Context: run after buildDockerImages.py with seedpy310 activated.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$SCRIPT_DIR/runExperiment.py" up "$@"
