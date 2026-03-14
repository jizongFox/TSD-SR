#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-tsd-sr:cu128}"
LOG_FILE="${2:-docker-build.log}"

docker build --progress=plain -t "$IMAGE_TAG" . 2>&1 | tee "$LOG_FILE"
